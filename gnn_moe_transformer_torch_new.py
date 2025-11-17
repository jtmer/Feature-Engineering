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


# ==============================
# Utils
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mape_adj(pred, target, eps: float = 1e-3):
    """Adjusted MAPE（避免分母过小）"""
    return (torch.abs(pred - target) / torch.clamp(torch.abs(target), min=eps)).mean()


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)


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
    区间专家：基于分桶预测每个时间步所属的值域区间。
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
        """用当前 batch 标签的分位数平滑更新 bin 边界。"""
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
    """值专家：拟合残差 y - pred_M。"""

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
      输入：变量 embedding E_j ∈ R^{H_g}
      输出：4 维 logits -> softmax 概率：
        [p_none, p_trend, p_zone, p_value]
    """
    def __init__(self, h_gnn: int, num_states: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_gnn, h_gnn),
            nn.GELU(),
            nn.Linear(h_gnn, num_states),
        )

    def forward(self, E):  # E:(D,H_g)
        logits = self.mlp(E)           # (D,4)
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
        （Zone 通过区间约束/惩罚影响 loss）。
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        gnn_type: str = "gat",
        gnn_hidden: int = 64,
        num_gnn_layers: int = 2,
        num_bins: int = 5,
        downsample_stride: int = 4,
        moe_hidden: int = 128,
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

        # VSN 可选
        self.time_linear = nn.Linear(self.D, d_model)
        self.vsn = VariableSelection(d_model, self.D) if self.flags.use_vsn else None

        # Transformer 主干
        self.backbone = TransformerBackbone(
            d_in=d_model,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            posenc=posenc,
        )

        # 基准预测头
        self.base_out = nn.Linear(d_model, 1)

        # MoE：三个专家
        self.trend_expert = TrendExpert(d_model, stride=downsample_stride)
        self.zone_expert = ZoneExpert(d_model, num_bins=num_bins, hidden=moe_hidden)
        self.value_expert = ValueExpert(d_model, hidden=moe_hidden)

        # gate
        if self.flags.use_gnn and self.flags.use_moe:
            self.var_gating = VariableGating(gnn_hidden, num_states=4)
        else:
            self.var_gating = None  # 退化为无门控

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

    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        X: (B,L,D)
        y: (B,L,1) 
        返回：
          pred: (B,L,1)
          aux:  辅助输出，用于 loss 计算
        """
        B, L, D = X.shape
        assert D == self.D

        Xt_raw = self.time_linear(X)  # (B,L,d_model)
        if self.vsn is not None:
            w_vsn = self.vsn(Xt_raw)    # (B,L,D)
            X_weighted = X * w_vsn      # element-wise 特征选择
            Xt = self.time_linear(X_weighted)
        else:
            w_vsn = None
            Xt = Xt_raw

        H = self.backbone(Xt)          # (B,L,d_model)
        pred_M = self.base_out(H)      # (B,L,1)

        # --------- 区间专家（zone） ---------
        if self.training and (y is not None):
            self.zone_expert.update_bins_from_labels(y)
        logits_bins = self.zone_expert(H)               # (B,L,K)
        p_bins = F.softmax(logits_bins, dim=-1)         # (B,L,K)
        label_bins = self.zone_expert.labels_from_y(y) if y is not None else None

        # --------- Trend & Value 专家 ---------
        pred_trend = self.trend_expert(H) if self.flags.use_trend else torch.zeros_like(pred_M)
        pred_value = self.value_expert(H) if self.flags.use_value else torch.zeros_like(pred_M)

        # --------- GNN + 变量级 Gate（4 状态） ---------
        if self.flags.use_gnn and self.flags.use_moe:
            E = self._compute_variable_embeddings(X)     # (D,H_g)
            gate_logits, p_vars = self.var_gating(E)     # (D,4)
            p_global = p_vars.mean(dim=0)                # (4,)

            # 顺序：[none, trend, zone, value]
            g_none = p_global[0]
            g_trend = p_global[1]
            g_zone = p_global[2]
            g_value = p_global[3]

            # broadcast 为 (1,1,1) 用于缩放专家输出
            g_trend_b = g_trend.view(1, 1, 1)
            g_value_b = g_value.view(1, 1, 1)
        else:
            # 无 GNN 或禁用 MoE 时，退化为普通 Transformer + 专家全开
            E = None
            gate_logits = None
            p_vars = None
            p_global = None
            g_none = torch.tensor(0.0, device=X.device)
            g_trend = torch.tensor(1.0, device=X.device)
            g_zone = torch.tensor(1.0, device=X.device)
            g_value = torch.tensor(1.0, device=X.device)
            g_trend_b = g_value_b = torch.ones(1, 1, 1, device=X.device)

        pred_raw = pred_M + g_trend_b * pred_trend + g_value_b * pred_value  # (B,L,1)

        aux = {
            "pred_M": pred_M,
            "pred_trend": pred_trend,
            "pred_value": pred_value,
            "logits_bins": logits_bins,
            "p_bins": p_bins,
            "label_bins": label_bins,
            "E": E,
            "gate_logits": gate_logits,
            "p_vars": p_vars,          # (D,4)
            "p_global": p_global,      # (4,)
            "g_none": g_none,
            "g_trend": g_trend,
            "g_zone": g_zone,
            "g_value": g_value,
            "pred_raw": pred_raw,
            "w_vsn": w_vsn,
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
