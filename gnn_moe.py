import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse

from transformers.activations import ACT2FN
from transformers import AutoModelForCausalLM, AutoConfig
from sundial.configuration_sundial import SundialConfig
from sundial.ts_generation_mixin import TSGenerationMixin

# ==============================
# Utils
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mape_adj(pred, target, eps: float = 1e-3):
    return (torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps)).mean()


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)


class SundialPatchEmbedding(nn.Module):
    def __init__(self, config: SundialConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.hidden_layer = nn.Linear(
            config.input_token_len * 2, config.intermediate_size
        )
        self.act = ACT2FN[config.hidden_act]
        self.output_layer = nn.Linear(config.intermediate_size, config.hidden_size)
        self.residual_layer = nn.Linear(config.input_token_len * 2, config.hidden_size)
        self.input_token_len = config.input_token_len

    def forward(self, x):
        mask = torch.ones_like(x, dtype=torch.float32)
        input_length = x.shape[-1]
        padding_length = (
            self.input_token_len - (input_length % self.input_token_len)
        ) % self.input_token_len
        x = F.pad(x, (padding_length, 0))
        mask = F.pad(mask, (padding_length, 0))
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        mask = mask.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len
        )

        x = torch.cat([x, mask], dim=-1)
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res
        return out


# ==============================
# Graph Build
# ==============================

class LearnableAdjacency(nn.Module):
    """
    可学习稠密邻接矩阵 A∈R^{D×D}：
      - 对称化 (A + A^T) / 2
      - softplus 保证非负
      - 对角线强制为 0
    """
    def __init__(self, D: int):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(D, D))
        nn.init.xavier_uniform_(self.raw)

    def forward(self):
        A = 0.5 * (self.raw + self.raw.t())
        A = F.softplus(A)
        A = A - torch.diag(torch.diag(A))
        return A  # (D,D), nonnegative


def pearson_graph_from_batch(X: torch.Tensor, thr: float = 0.3) -> torch.Tensor:
    """
    X:(B,L,D) -> Pearson |r|≥thr 连边的 0/1 邻接矩阵 A∈R^{D×D}
    用当前 batch 近似整体统计。
    """
    B, L, D = X.shape
    Xf = X.reshape(B * L, D)
    Xf = Xf - Xf.mean(dim=0, keepdim=True)
    denom = torch.clamp(Xf.std(dim=0, keepdim=True), min=1e-6)
    Xn = Xf / denom
    corr = (Xn.t() @ Xn) / (Xn.shape[0] - 1)
    A = (corr.abs() >= thr).float()
    A.fill_diagonal_(0.0)
    return A  # (D,D)


# ==============================
# PyG GNN（变量级）
# ==============================

class PyGGNNEstimator(nn.Module):
    """
    使用 PyG 的 GCN / GAT，在“变量图”上学习变量 embedding：
      - 每个变量一个节点，总共 D 个节点；
      - 节点特征为当前 batch 中该变量的均值 / 标准差，Fin=2；
      - graph_mode="static": Pearson 阈值图；
      - graph_mode="learnable": 学习型邻接 A∈R^{D×D}。
    输出：
      E ∈ R^{D×H} 变量 embedding，用于 MoE Gate。
    """
    def __init__(
        self,
        D: int,
        H: int = 64,
        K: int = 2,
        gnn_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.1,
        graph_mode: str = "static"
    ):
        super().__init__()
        self.D = D
        self.H = H
        self.K = K
        self.gnn_type = gnn_type
        self.graph_mode = graph_mode
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        convs = []
        for i in range(K):
            in_ch = 2 if i == 0 else H
            if gnn_type == "gcn":
                convs.append(GCNConv(in_ch, H, add_self_loops=True, normalize=True))
            else:
                # GATConv: out_dim = out_channels * heads (concat=True)
                # 为保持输出维度 H，设 heads=1, out_channels=H
                convs.append(
                    GATConv(
                        in_ch,
                        H,
                        heads=1,
                        dropout=dropout,
                        add_self_loops=True,
                        concat=True,
                    )
                )
        self.convs = nn.ModuleList(convs)
        self.act = nn.GELU()

        if graph_mode == "learnable":
            self.learnable_A = LearnableAdjacency(D)
        else:
            self.learnable_A = None

    def _build_graph(self, X: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        根据设置构建 PyG 所需的 edge_index / edge_weight。
        返回：
          edge_index: (2,E) [long]
          edge_weight: (E,) [float] 或 None
        """
        device = X.device
        D = self.D

        if self.graph_mode == "learnable":
            A = self.learnable_A().to(device)  # (D,D)
            dense = (A > 0) | torch.eye(D, device=device).bool()
            edge_index, _ = dense_to_sparse(dense.float())
            if self.gnn_type == "gcn":
                ew = A[edge_index[0], edge_index[1]].clamp(min=1e-6)
            else:
                ew = None
            return edge_index, ew
        else:
            with torch.no_grad():
                A = pearson_graph_from_batch(X).to(device)
            dense = (A > 0) | torch.eye(D, device=device).bool()
            edge_index, _ = dense_to_sparse(dense.float())
            return edge_index, None

    def forward(self, node_feats: torch.Tensor, X_for_graph: torch.Tensor):
        """
        node_feats: (B,D,2)  变量统计特征（均值/标准差）
        X_for_graph: (B,L,D) 用于静态图构建或作为上下文（不回传梯度）
        返回：
          E: (D,H)  变量 embedding
        """
        B, D, Fin = node_feats.shape
        assert D == self.D and Fin == 2

        edge_index, edge_weight = self._build_graph(X_for_graph)

        # batch 内样本平均作为图信号输入
        x = node_feats.mean(dim=0)  # (D,2)

        for conv in self.convs:
            if isinstance(conv, GCNConv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)

        return x  # (D,H)


# ==============================
# VSN（可选）
# ==============================

class GRN(nn.Module):
    def __init__(self, inp: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, inp)
        self.gate = nn.GLU()
        self.norm = nn.LayerNorm(inp)

    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = torch.cat([x, residual], dim=-1)
        x = self.gate(x)
        return self.norm(x + residual)


class VariableSelection(nn.Module):
    """
    与初版保持一致：可以对时间步 embedding 做上下文相关的变量权重。
    在本版本中，仅用于（可选）对输入特征 X 做 element-wise 加权，
    不再与 GNN 变量 embedding 融合。
    """
    def __init__(self, d_model: int, D: int):
        super().__init__()
        self.context = GRN(d_model, max(32, d_model // 2))
        self.proj = nn.Linear(d_model, D)

    def forward(self, H):  # H: (B,L,d_model)
        ctx = self.context(H)
        w = F.softmax(self.proj(ctx), dim=-1)  # (B,L,D)
        return w


# ==============================
# Positional Encoding
# ==============================

class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):  # x:(B,L,C)
        L = x.shape[1]
        return x + self.pe[:, :L, :]


class LearnablePosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe[:, :L, :]


# ==============================
# Transformer Backbone
# ==============================

class TransformerBackbone(nn.Module):
    """
    标准 TransformerEncoder：
      输入 Z_in:(B,L,d_in)
      线性投影到 d_model，加位置编码，再做 self-attention。
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        posenc: str = "learnable",
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.posenc = (
            LearnablePosEnc(d_model) if posenc == "learnable" else SinusoidalPosEnc(d_model)
        )

    def forward(self, Z):  # Z:(B,L,d_in)
        H = self.input_proj(Z)
        H = self.posenc(H)
        H = self.encoder(H)
        return H  # (B,L,d_model)


# ==============================
# Experts & Gating
# ==============================

class TrendExpert(nn.Module):
    """
    低频趋势专家：
      - depthwise 1D conv 下采样
      - 线性插值上采样
      - 输出 (B,L,1)
    """
    def __init__(self, d_model: int, stride: int = 4):
        super().__init__()
        self.stride = stride
        self.smoother = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=stride,
            stride=stride,
            groups=d_model,
            padding=0,
        )
        nn.init.constant_(self.smoother.weight, 1.0 / stride)
        if self.smoother.bias is not None:
            nn.init.zeros_(self.smoother.bias)
        self.out = nn.Linear(d_model, 1)

    def forward(self, H):  # H:(B,L,C)
        B, L, C = H.shape
        x = H.transpose(1, 2)            # (B,C,L)
        ds = self.smoother(x)            # (B,C,L')
        us = F.interpolate(ds, size=L, mode="linear", align_corners=False)
        us = us.transpose(1, 2)          # (B,L,C)
        return self.out(us)              # (B,L,1)

    def supervised_target(self, y):      # y:(B,L,1)
        B, L, _ = y.shape
        x = y.transpose(1, 2)            # (B,1,L)
        kernel = torch.ones(1, 1, self.stride, device=y.device) / self.stride
        ds = F.conv1d(x, kernel, stride=self.stride, padding=0)
        us = F.interpolate(ds, size=L, mode="linear", align_corners=False)
        return us.transpose(1, 2)        # (B,L,1)


class ZoneExpert(nn.Module):
    """
    区间专家：基于分桶预测每个时间步所属的值域区间
    """
    def __init__(self, d_model: int, num_bins: int = 5, hidden: int = 128):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_bins),
        )
        self.register_buffer(
            "bin_edges",
            torch.linspace(0.0, 1.0, steps=num_bins + 1),
            persistent=False,
        )
        self.momentum = 0.05

    def update_bins_from_labels(self, y: torch.Tensor):
        """用当前 batch 标签的分位数平滑更新 bin 边界"""
        y1 = y.detach().reshape(-1)
        if y1.numel() < self.num_bins * 4:
            return
        qs = torch.quantile(
            y1, torch.linspace(0, 1, self.num_bins + 1, device=y.device)
        )
        self.bin_edges = (1 - self.momentum) * self.bin_edges + self.momentum * qs

    def forward(self, H):  # H:(B,L,d_model)
        return self.net(H)  # logits:(B,L,num_bins)

    def labels_from_y(self, y: torch.Tensor):
        bins = torch.bucketize(
            y.detach(), self.bin_edges[1:-1], right=False
        )  # (B,L,1)
        return bins.squeeze(-1).long()  # (B,L)


class ValueExpert(nn.Module):
    """值专家：拟合残差 y - pred_M"""

    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, H):
        return self.net(H)  # (B,L,1)


class VariableGating(nn.Module):
    """
    变量级 Gate：
      输入：变量 embedding E_j ∈ R^{H_g}，以及来自目标 hidden states 的上下文 c ∈ R^{H_g}
      输出：4 维 logits -> softmax 概率：[p_none, p_trend, p_zone, p_value]
    """
    def __init__(self, h_gnn: int, num_states: int = 4, use_context: bool = True):
        super().__init__()
        self.use_context = use_context
        in_dim = h_gnn * (2 if use_context else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h_gnn),
            nn.GELU(),
            nn.Linear(h_gnn, num_states),
        )

    def forward(self, E, ctx: Optional[torch.Tensor] = None):
        """
        E:   (D,H_g)
        ctx: (H_g,) 或 None
        """
        if self.use_context and (ctx is not None):
            ctx_exp = ctx.unsqueeze(0).expand(E.size(0), -1)  # (D,H_g)
            x = torch.cat([E, ctx_exp], dim=-1)               # (D, 2H_g)
        else:
            x = E                                             # (D,H_g)

        logits = self.mlp(x)           # (D,4)
        p = F.softmax(logits, dim=-1)  # (D,4)
        return logits, p



# ==============================
# Full Model
# ==============================

@dataclass
class ModelFlags:
    use_gnn: bool = True
    use_vsn: bool = False
    use_trend: bool = True
    use_zone_soft: bool = False  # False: hard clamp; True: soft penalty
    use_value: bool = True
    use_moe: bool = True
    graph_mode: str = "static"   # ["static","learnable"]


class GNNMoETransformerRegressor(nn.Module):
    """
    整体模型：
      X:(B,L,D) ->
        Transformer 主干得到基准预测 pred_M；
        GNN 得到变量 embedding E；
        Gate(E) -> 4 状态概率，聚合为全局权重；
        Trend / Zone / Value 三专家 + 不作用状态；
        最终 pred = pred_M + g_trend * pred_trend + g_value * pred_value
        （Zone 通过区间约束/惩罚影响 loss）
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        gnn_type: str = "gat",
        gnn_hidden: int = 128,
        num_gnn_layers: int = 2,
        num_bins: int = 5,
        downsample_stride: int = 4,
        moe_hidden: int = 256,
        flags: Optional[ModelFlags] = None,
        posenc: str = "learnable",
    ):
        super().__init__()
        self.D = input_size
        self.flags = flags or ModelFlags()
        self.num_bins = num_bins
        self.gnn_hidden = gnn_hidden

        # GNN: 变量 embedding
        if self.flags.use_gnn:
            self.gnn = PyGGNNEstimator(
                self.D,
                H=gnn_hidden,
                K=num_gnn_layers,
                gnn_type=gnn_type,
                heads=max(1, num_heads // 2),
                dropout=dropout,
                graph_mode=self.flags.graph_mode,
            )
        else:
            self.gnn = None

        # self.y_embed = nn.Linear(1, d_model)
        self.backbone_config = AutoConfig.from_pretrained(
            'thuml/sundial-base-128m', trust_remote_code=True
        )
        # self.backbone_config.use_return_dict = True
        self.backbone = AutoModelForCausalLM.from_pretrained(
            'thuml/sundial-base-128m', trust_remote_code=True, config=self.backbone_config
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # self.y_embed = SundialPatchEmbedding(self.backbone_config)
        self.embedding = self.backbone.model.embed_layer
        for p in self.embedding.parameters():
            p.requires_grad = False
        self.token_hidden_size = self.backbone_config.hidden_size
        self.patch_len = self.backbone_config.input_token_len  # patch 长度
        
        # # 把 backbone 的标量预测 pred_M 映射到专家空间 (B,L_pred,d_model)
        # self.pred_proj = nn.Linear(1, d_model)

        # # VSN 可选
        # self.time_linear = nn.Linear(self.D, d_model)
        # self.vsn = VariableSelection(d_model, self.D) if self.flags.use_vsn else None

        # MoE：按变量独立的时序 encoder
        # 每个变量只用自己的标量序列作为输入(后面reshape 成 (B*D, L, 1))
        self.moe_input_proj = nn.Linear(self.token_hidden_size, d_model)
        self.moe_backbone = TransformerBackbone(
            d_in=d_model,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            posenc=posenc,
        )

        # MoE：三个专家
        self.trend_expert = TrendExpert(d_model, stride=downsample_stride)
        self.zone_expert = ZoneExpert(d_model, num_bins=num_bins, hidden=moe_hidden)
        self.value_expert = ValueExpert(d_model, hidden=moe_hidden)

        if self.flags.use_gnn:
            self.var_trend_proj = nn.Linear(gnn_hidden, d_model)
            self.var_value_proj = nn.Linear(gnn_hidden, d_model)
        else:
            self.var_trend_proj = None
            self.var_value_proj = None

        # gate
        if self.flags.use_gnn and self.flags.use_moe:
            self.var_gating = VariableGating(gnn_hidden, num_states=4, use_context=True)
            # H 的全局上下文 -> gate context（维度对齐 H_g）
            self.gate_ctx_proj = nn.Sequential(
                nn.Linear(self.backbone_config.hidden_size, d_model),
                nn.Linear(d_model, gnn_hidden)
            )
        else:
            self.var_gating = None
            self.gate_ctx_proj = None

    def _compute_variable_embeddings(self, X: torch.Tensor):
        """
        X:(B,L,D) -> E:(D,H_g)
        节点特征为每个变量在时间维上的均值 / 标准差。
        """
        if self.gnn is None:
            return None

        mean = X.mean(dim=1)  # (B,D)
        std = X.std(dim=1)    # (B,D)
        node_feats = torch.stack([mean, std], dim=-1)  # (B,D,2)

        E = self.gnn(node_feats, X)  # (D,H_g)
        return E
    
    def _build_moe_hidden_per_var(self, X_tokens: torch.Tensor, B: int, D: int):
        """
        X_tokens: (B*D, L_tok, token_hidden_size) — 经过 Sundial patch embedding 后的 token 序列
        返回:
          H_var: (B, D, L_tok, d_model)  # 每个变量自己的 MoE hidden 序列（在 patch 维度上）
        """
        B_D, L_tok, _ = X_tokens.shape
        assert B_D == B * D

        Z_flat = self.moe_input_proj(X_tokens)         # (B*D,L_tok,d_model)
        H_flat = self.moe_backbone(Z_flat)             # (B*D,L_tok,d_model)

        H_var = H_flat.view(B, D, L_tok, -1)           # (B,D,L_tok,d_model)
        return H_var

    def forward(
        self,
        X: torch.Tensor,              # (B, L_pred, D)  未来协变量
        y_hist: torch.Tensor,         # (B, L_hist, 1)  过去目标变量
        y_future: Optional[torch.Tensor] = None,  # (B, L_pred, 1)  未来目标变量
        return_aux: bool = False,
    ):
        """
        语义：
          输入：未来协变量 X_future + 过去 y_hist
          （训练时）提供未来 y_future 作为监督和区间/趋势 target
        输出：
          pred: (B,L_pred,1)  对未来窗口的预测
        """
        B, L_pred, D = X.shape
        assert D == self.D
        B_y, L_hist, C_y = y_hist.shape
        assert B_y == B and C_y == 1
        y_hist = y_hist.squeeze()

        device = X.device
        dtype = X.dtype
        
        # backbone 预测
        with torch.no_grad():
            pred_M = self.backbone.generate(
                y_hist,
                max_new_tokens=self.backbone_config.output_token_lens[0],
            ).mean(dim=1)
        if pred_M.dim() == 2:
            pred_M = pred_M.unsqueeze(-1)         # (B, L_pred, 1)
        
        # y 的 embedding 参与 GNN
        y_ctx_emb = self.embedding(y_hist)            # (B, L_hist_tok, hidden)
        # 全局上下文：先对时间，再对 batch 求平均
        gate_ctx = y_ctx_emb.mean(dim=1).mean(dim=0)   # (hidden,)
        gate_ctx = self.gate_ctx_proj(gate_ctx)         # (H_g)
        
        X_var = X.permute(0, 2, 1).contiguous()          # (B,D,L_pred)
        X_flat = X_var.view(B * D, L_pred)               # (B*D,L_pred)
        X_tokens = self.embedding(X_flat)                # (B*D,L_tok,token_hidden_size)
        L_tok = X_tokens.size(1)
        
        # MoE：按变量构建时序 hidden
        H_var = self._build_moe_hidden_per_var(X_tokens, B, D)  # (B,D,L_tok,d_model)

        # GNN + 变量级 Gate + var_hidden
        # 这里用未来窗口的协变量 X 估计变量 embedding
        E = self._compute_variable_embeddings(X)       # (D,H_g)
        gate_logits, p_vars = self.var_gating(E, gate_ctx)  # (D,4)

        var_labels = torch.argmax(p_vars, dim=-1)      # (D,)

        # Gate 权重（按变量区分），利用 none 抑制不重要变量
        p_none  = p_vars[:, 0:1]                       # (D,1)
        p_trend = p_vars[:, 1:2]                       # (D,1)
        p_zone  = p_vars[:, 2:3]                       # (D,1)
        p_value = p_vars[:, 3:4]                       # (D,1)

        w_trend = p_trend * (1.0 - p_none)             # (D,1)
        w_value = p_value * (1.0 - p_none)             # (D,1)
        w_zone  = p_zone  * (1.0 - p_none)             # (D,1)
        
        
        
        # 变量级 hidden states：embedding -> expert 空间
        var_h_trend = self.var_trend_proj(E)           # (D,d_model)
        var_h_value = self.var_value_proj(E)           # (D,d_model)
        v_trend = var_h_trend.view(1, D, 1, -1)   # (1,D,1,d_model)
        v_value = var_h_value.view(1, D, 1, -1)   # (1,D,1,d_model)
        
        H_trend = H_var + v_trend                        # (B,D,L_tok,d_model)
        H_value = H_var + v_value                        # (B,D,L_tok,d_model)
        H_trend_flat = H_trend.view(B * D, L_tok, -1)    # (B*D,L_tok,d_model)
        H_value_flat = H_value.view(B * D, L_tok, -1)    # (B*D,L_tok,d_model)
        
        pred_trend_var = self.trend_expert(H_trend_flat).view(B, D, L_tok, 1)  # (B,D,L_tok,1)
        pred_value_var = self.value_expert(H_value_flat).view(B, D, L_tok, 1)  # (B,D,L_tok,1)

        # 归一化
        w_trend = w_trend / (w_trend.sum() + 1e-6)  # (D,1)
        w_value = w_value / (w_value.sum() + 1e-6)  # (D,1)
        w_zone = w_zone / (w_zone.sum() + 1e-6)     # (D,1)

        contrib_trend_patch = (w_trend.view(1, D, 1, 1) * pred_trend_var).sum(dim=1)  # (B,L_tok,1)
        contrib_value_patch = (w_value.view(1, D, 1, 1) * pred_value_var).sum(dim=1)  # (B,L_tok,1)
        
        # pred_raw = pred_M + contrib_trend + contrib_value   # (B,L,1)

        # # 区间专家, 变量加权
        # H_zone = (w_zone.view(1, D, 1, 1) * H_var).sum(dim=1)   # (B,L,d_model)
        # if self.training and (y_future is not None):
        #     self.zone_expert.update_bins_from_labels(y_future)
        # logits_bins = self.zone_expert(H_zone)          # (B,L,K)
        # p_bins = F.softmax(logits_bins, dim=-1)         # (B,L,K)
        # label_bins = self.zone_expert.labels_from_y(y_future) if y_future is not None else None


        p_global = p_vars.mean(dim=0)                  # (4,)
        g_none  = p_global[0]
        g_trend = p_global[1]
        g_zone  = p_global[2]
        g_value = p_global[3]

        aux = {
            "pred_M": pred_M,
            "pred_trend": contrib_trend,
            "pred_value": contrib_value,
            "logits_bins": logits_bins,
            "p_bins": p_bins,
            "label_bins": label_bins,
            "E": E,
            "gate_logits": gate_logits,
            "p_vars": p_vars,
            "p_global": p_global,
            "g_none": g_none,
            "g_trend": g_trend,
            "g_zone": g_zone,
            "g_value": g_value,
            "pred_raw": pred_raw,
            "var_labels": var_labels,
        }

        return (pred_raw, aux) if return_aux else pred_raw


# ==============================
# Losses & Interval
# ==============================

@dataclass
class LossWeights:
    lv: float = 0.5   # value expert
    lz: float = 0.5   # zone classification
    lt: float = 0.2   # trend
    lc: float = 0.1   # clamp soft penalty
    lu: float = 0.1   # unused/none regularization


def interval_from_bins(bin_edges: torch.Tensor, p_bins: torch.Tensor):
    """
    由分桶概率 p_bins 得到每个时间步的区间 [lower, upper] 以及中点 mid。
    这里使用 argmax bucket 作为当前区间。
    """
    B, L, K = p_bins.shape
    device = p_bins.device
    edges = bin_edges.to(device)  # (K+1,)

    idx = torch.argmax(p_bins, dim=-1)  # (B,L)
    lower = edges[:-1].gather(0, idx.reshape(-1)).reshape(B, L, 1)
    upper = edges[1:].gather(0, idx.reshape(-1)).reshape(B, L, 1)
    mid = 0.5 * (lower + upper)
    return lower, upper, mid


def clamp_with_bins(pred, p_bins, bin_edges, hard: bool = True):
    lower, upper, mid = interval_from_bins(bin_edges, p_bins)
    if hard:
        return torch.clamp(pred, min=lower, max=upper), lower, upper, mid
    else:
        return pred, lower, upper, mid


def compute_losses(
    y: torch.Tensor,
    pred: torch.Tensor,
    aux: dict,
    weights: LossWeights,
    hard_clamp: bool = False,
    bin_edges: Optional[torch.Tensor] = None,
    trend_target: Optional[torch.Tensor] = None,
):
    """
      y: (B,L,1)
      pred: 模型 raw 输出（尚未区间裁剪）
      aux: model forward 返回的辅助字典
    """
    pred_M = aux["pred_M"]
    pred_trend = aux["pred_trend"]
    pred_value = aux["pred_value"]
    logits_bins = aux["logits_bins"]
    p_bins = aux["p_bins"]
    label_bins = aux["label_bins"]
    p_vars = aux.get("p_vars", None)
    p_global = aux.get("p_global", None)
    g_zone = aux.get("g_zone", None)

    # --------- 区间限制 ---------
    if bin_edges is None:
        with torch.no_grad():
            flat = y.reshape(-1)
            bin_edges = torch.quantile(
                flat,
                torch.linspace(0, 1, p_bins.shape[-1] + 1, device=y.device),
            )

    pred_adj, lower, upper, _ = clamp_with_bins(pred, p_bins, bin_edges, hard=hard_clamp)
    
    loss_main = F.mse_loss(pred_adj, y)
    loss_value = F.mse_loss(pred_value, y - pred_M)
    if label_bins is not None:
        loss_zone = F.cross_entropy(
            logits_bins.reshape(-1, logits_bins.shape[-1]),
            label_bins.reshape(-1),
        )
    else:
        loss_zone = logits_bins.mean() * 0.0

    # 用全局 g_zone 让区间相关 loss 也能影响 Gate / GNN
    if (g_zone is not None) and torch.is_tensor(g_zone):
        loss_zone_eff = g_zone * loss_zone
    else:
        loss_zone_eff = loss_zone

    if trend_target is None:
        trend_target = pred_trend.detach()
    loss_trend = F.mse_loss(pred_trend, trend_target)

    if not hard_clamp:
        over = F.relu(pred - upper)
        under = F.relu(lower - pred)
        loss_clamp = (over ** 2 + under ** 2).mean()
        if (g_zone is not None) and torch.is_tensor(g_zone):
            loss_clamp = g_zone * loss_clamp
    else:
        loss_clamp = pred.mean() * 0.0

    # 不作用
    if p_vars is not None:
        p_none = p_vars[:, 0]         # (D,)
        loss_unused = (1.0 - p_none).mean()
    else:
        loss_unused = pred.mean() * 0.0

    total = (
        loss_main
        + weights.lv * loss_value
        + weights.lz * loss_zone_eff
        + weights.lt * loss_trend
        + weights.lc * loss_clamp
        + weights.lu * loss_unused
    )

    metrics = {
        "loss_main": loss_main.detach().item(),
        "loss_value": loss_value.detach().item(),
        "loss_zone": loss_zone.detach().item(),
        "loss_trend": loss_trend.detach().item(),
        "loss_clamp": loss_clamp.detach().item(),
        "loss_unused": loss_unused.detach().item(),
        "mape_adj": mape_adj(pred_adj.detach(), y.detach()).item(),
    }

    if p_global is not None:
        metrics.update(
            {
                "g_none": aux["g_none"].detach().item(),
                "g_trend": aux["g_trend"].detach().item(),
                "g_zone": aux["g_zone"].detach().item(),
                "g_value": aux["g_value"].detach().item(),
            }
        )

    return total, metrics, (lower.detach(), upper.detach())
