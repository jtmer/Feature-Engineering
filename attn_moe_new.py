import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN

from sundial.configuration_sundial import SundialConfig
from sundial.ts_generation_mixin import TSGenerationMixin

def print_mem(label):
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{label}] allocated = {alloc:.1f} MB, reserved = {reserved:.1f} MB")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L)
        return: (B, N_patch, hidden_size)
        """
        assert x.dim() == 2, f"expect (B,L), got {x.shape}"
        B, L = x.shape
        mask = torch.ones_like(x, dtype=torch.float32)

        padding_length = (self.input_token_len - (L % self.input_token_len)) % self.input_token_len
        if padding_length > 0:
            x = F.pad(x, (padding_length, 0))
            mask = F.pad(mask, (padding_length, 0))

        # (B, N_patch, patch_len)
        x = x.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)
        mask = mask.unfold(dimension=-1, size=self.input_token_len, step=self.input_token_len)

        x_cat = torch.cat([x, mask], dim=-1)  # (B,N_patch,2*patch_len)

        hid = self.act(self.hidden_layer(x_cat))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x_cat)
        out = out + res
        return out  # (B,N_patch,H)


# 数据处理专家：按 patch 处理
class DataProcessExpert(nn.Module):
    """
      1) 给定一组 patch 级 μ/σ，对 y_hist 做 patch 级仿射变换 -> y_hist_proc
      2) 用同一组 μ/σ（通过 patch 维度插值扩展到未来）把 backbone 输出 pred_norm 反变换回原空间
    """
    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = patch_len

    def _patchify(self, y: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        y: (B, L)
        return:
          patches: (B, N_patch, P)
          pad_left: int
        """
        B, L = y.shape
        P = self.patch_len
        pad_left = (P - (L % P)) % P
        if pad_left > 0:
            y = F.pad(y, (pad_left, 0))
        patches = y.unfold(-1, P, P)  # (B,Np,P)
        return patches, pad_left

    def compute_patch_stats(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅用于辅助监督（用未来 y 的真实 patch 均值/方差
        y: (B, L)
        return:
          mu_true:(B,Np), sigma_true:(B,Np)
        """
        patches, _ = self._patchify(y)
        mu = patches.mean(dim=-1)
        sigma = patches.std(dim=-1) + 1e-3
        return mu, sigma

    def normalize_with_params(
        self,
        y_hist: torch.Tensor,     # (B,L_hist)
        mu_hist: torch.Tensor,    # (B,N_hist)
        sigma_hist: torch.Tensor  # (B,N_hist)
    ) -> torch.Tensor:
        """
        使用值域专家给出的 μ/σ对历史目标做 patch 级变换
        """
        patches, pad_left = self._patchify(y_hist)      # (B,Np,P)
        B, Np, P = patches.shape
        assert Np == mu_hist.shape[1] == sigma_hist.shape[1], \
            f"N_patch mismatch: {Np}, mu:{mu_hist.shape[1]}, sigma:{sigma_hist.shape[1]}"

        z_patches = (patches - mu_hist.unsqueeze(-1)) / sigma_hist.unsqueeze(-1)
        y_norm_all = z_patches.reshape(B, Np * P)
        if pad_left > 0:
            y_norm_all = y_norm_all[:, pad_left:]
        return y_norm_all  # (B,L_hist)

    def denormalize_future_with_params(
        self,
        y_future_norm: torch.Tensor,  # (B,L_pred)
        mu_hist: torch.Tensor,        # (B,N_hist)
        sigma_hist: torch.Tensor,     # (B,N_hist)
        L_pred: int,
    ) -> torch.Tensor:
        """
        对 backbone 输出的标准化预测做逆处理，得到最终预测：
          y_pred = z * σ_future + μ_future
        其中 μ_future/σ_future 由 μ_hist/σ_hist 在 patch 维度上线性插值得到。
        """
        B, L = y_future_norm.shape
        assert L == L_pred

        patches_norm, pad_left = self._patchify(y_future_norm)  # (B,N_pred,P)
        B2, N_pred, P = patches_norm.shape
        assert B2 == B

        # μ_hist, σ_hist 在 patch 维度插值到 N_pred
        if N_pred != mu_hist.shape[1]:
            mu_future = F.interpolate(
                mu_hist.unsqueeze(1), size=N_pred, mode="linear", align_corners=False
            ).squeeze(1)  # (B,N_pred)
            sigma_future = F.interpolate(
                sigma_hist.unsqueeze(1), size=N_pred, mode="linear", align_corners=False
            ).squeeze(1)  # (B,N_pred)
        else:
            mu_future = mu_hist
            sigma_future = sigma_hist

        y_denorm_patches = patches_norm * sigma_future.unsqueeze(-1) + mu_future.unsqueeze(-1)
        y_denorm_all = y_denorm_patches.reshape(B, N_pred * P)
        if pad_left > 0:
            y_denorm_all = y_denorm_all[:, pad_left:]
        return y_denorm_all[:, :L_pred]  # (B,L_pred)


# 值域专家：逐协变量 + 逐 patch 输出 (mu_j, sigma_j)
class ValueRangeExpertVarWise(nn.Module):
    """
    H_var:(B,D,N_hist,H)  ->  mu_var,sigma_var:(B,D,N_hist)
    """
    def __init__(self, hidden_size: int, hidden_mlp: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, hidden_mlp),
            nn.GELU(),
        )
        self.out_mu = nn.Linear(hidden_mlp, 1)
        self.out_logsigma = nn.Linear(hidden_mlp, 1)
        
        nn.init.zeros_(self.out_mu.weight)
        nn.init.zeros_(self.out_mu.bias)
        nn.init.zeros_(self.out_logsigma.weight)
        nn.init.zeros_(self.out_logsigma.bias)

    def forward(self, H_var: torch.Tensor):
        B, D, Np, H = H_var.shape
        x = H_var.reshape(B * D * Np, H)
        z = self.net(x)
        delta_mu = self.out_mu(z).view(B, D, Np)         # (B,D,N_hist)
        delta_logsigma = self.out_logsigma(z).view(B, D, Np)
        return delta_mu, delta_logsigma


# 变量级 Gate：给每个协变量一个权重
class VariableGatingSimple(nn.Module):
    """
    E_var:(B,D,H), ctx:(B,H) -> w_vars:(B,D)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, E_var: torch.Tensor, ctx: torch.Tensor):
        B, D, H = E_var.shape
        ctx_exp = ctx.unsqueeze(1).expand(-1, D, -1)  # (B,D,H)
        x = torch.cat([E_var, ctx_exp], dim=-1)       # (B,D,2H)
        logits = self.mlp(x).squeeze(-1)             # (B,D)
        w = F.softmax(logits, dim=-1)
        return logits, w


# =========================================================
# 主模型：逐协变量的 Patch-MoE + Sundial backbone
# =========================================================

@dataclass
class PatchMoEFlags:
    use_value_expert: bool = True
    freeze_backbone: bool = True
    mu_delta_scale: float = 0.1
    sigma_delta_scale: float = 0.05 


class CovAwareVarWisePatchMoERegressor(nn.Module):
    """
    时序流程（重点）：

      1) 对历史目标 y_hist 做 patch embedding: H_y:(B,N_hist,H)
      2) 对每个协变量 j 的未来序列 X_future[...,j] 做 patch embedding: H_x_var:(B,D,N_future,H)
      3) Cross-Attention：
           Q = H_y（目标 patch）
           K,V = 每个变量自己的 H_x_var_j（未来协变量 patch）
         得到 H_var_attn:(B,D,N_hist,H)，即“每个协变量对每个历史 patch 的影响表示”
      4) 值域专家（逐变量）：
           H_var_attn -> mu_var_hist,sigma_var_hist:(B,D,N_hist)
      5) 变量 Gate：
           w_vars:(B,D)  -> 聚合得到全局 mu_hist,sigma_hist:(B,N_hist)
      6) DataProcessExpert 用 mu_hist/sigma_hist 对 y_hist 按 patch 变换：
           y_hist_norm = (y_hist - mu_hist)/sigma_hist
      7) Sundial backbone 在 y_hist_norm 上生成未来标准化预测 pred_norm:(B,L_pred)
      8) DataProcessExpert 用“同一组 mu_hist/sigma_hist（在 patch 维度插值到未来）”
         把 pred_norm 反变换成最终预测 y_pred:(B,L_pred)
      9) 辅助损失：用 y_future 的真实 patch 均值/方差监督 “插值到未来”的 mu_pred_future/sigma_pred_future
    """
    def __init__(
        self,
        input_size: int,
        sundial_name: str = "thuml/sundial-base-128m",
        flags: Optional[PatchMoEFlags] = None,
    ):
        super().__init__()
        self.D = input_size
        self.flags = flags or PatchMoEFlags()

        self.backbone_config = AutoConfig.from_pretrained(
            sundial_name, trust_remote_code=True
        )
        self.backbone: TSGenerationMixin = AutoModelForCausalLM.from_pretrained(
            sundial_name, trust_remote_code=True, config=self.backbone_config
        )
        if self.flags.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.patch_len = self.backbone_config.input_token_len
        self.hidden_size = self.backbone_config.hidden_size

        self.patch_embed_y = self.backbone.model.embed_layer
        self.patch_embed_x = self.backbone.model.embed_layer

        self.data_proc = DataProcessExpert(self.patch_len)

        # Q = y_hist patches, K/V = 各变量的未来协变量 patches
        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=self.hidden_size,
        #     num_heads=4,
        #     dropout=0.1,
        #     batch_first=True,
        # )
        self.num_heads = 4
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_dropout = 0.1

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.value_expert = ValueRangeExpertVarWise(self.hidden_size, hidden_mlp=128)
        self.var_gating = VariableGatingSimple(self.hidden_size)

    def forward(
        self,
        X_future: torch.Tensor,          # (B,L_pred,D)
        y_hist: torch.Tensor,            # (B,L_hist,1)
        y_future: Optional[torch.Tensor] = None,  # (B,L_pred,1)
        return_aux: bool = False,
    ):
        device = X_future.device
        B, L_pred, D = X_future.shape
        assert D == self.D
        y_hist = y_hist.squeeze(-1)      # (B,L_hist)
        if y_future is not None:
            y_future = y_future.squeeze(-1)  # (B,L_pred)

        H_y = self.patch_embed_y(y_hist)          # (B,N_hist,H)
        N_hist = H_y.shape[1]
        ctx_y = H_y.mean(dim=1)                   # (B,H)，给 gate 用
        
        # 作为 base stats,专家只在此基础上学残差
        mu_base_hist, sigma_base_hist = self.data_proc.compute_patch_stats(y_hist)  # (B,N_hist_base)

        X_var = X_future.permute(0, 2, 1).contiguous()  # (B,D,L_pred)
        X_flat = X_var.view(B * D, L_pred)              # (B*D,L_pred)
        H_x_flat = self.patch_embed_x(X_flat)           # (B*D,N_future,H)
        N_future = H_x_flat.shape[1]
        
        H_y_rep = H_y.repeat_interleave(D, dim=0)       # (B*D, N_hist, H)
        Q = H_y_rep                                     # (B*D, N_hist, H)
        K = H_x_flat                                    # (B*D, N_future, H)
        V = H_x_flat

        q = self.q_proj(Q)  # (B*D, N_hist, H)
        k = self.k_proj(K)  # (B*D, N_future, H)
        v = self.v_proj(V)  # (B*D, N_future, H)

        BD, N_hist, H = q.shape
        _, N_future, _ = k.shape

        # (B*D, N, H) -> (B*D, num_heads, N, head_dim)
        q = q.view(BD, N_hist, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(BD, N_future, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(BD, N_future, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )  # (B*D, num_heads, N_hist, head_dim)

        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous().view(BD, N_hist, H)
        H_attn_flat = self.out_proj(attn_out)          # (B*D, N_hist, H)
        H_var_attn = H_attn_flat.view(B, D, N_hist, self.hidden_size)

        # 值域专家
        delta_mu_var_hist, delta_logsigma_var_hist = self.value_expert(H_var_attn)    # (B,D,N_hist)

        # 变量级 gate，得到全局 μ_hist/σ_hist
        E_var = H_var_attn.mean(dim=2)                # (B,D,H)
        gate_logits, w_vars = self.var_gating(E_var, ctx_y)  # w_vars:(B,D)

        w_exp = w_vars.unsqueeze(-1)                  # (B,D,1)
        delta_mu_hist = (w_exp * delta_mu_var_hist).sum(dim=1)       # (B,N_hist)
        delta_logsigma_hist = (w_exp * delta_logsigma_var_hist).sum(dim=1) # (B,N_hist)
        
        mu_hist = mu_base_hist + self.flags.mu_delta_scale * delta_mu_hist
        log_sigma_base = torch.log(sigma_base_hist + 1e-6)           # (B,N_hist)
        log_sigma_hist = log_sigma_base + self.flags.sigma_delta_scale * delta_logsigma_hist
        sigma_hist = torch.exp(log_sigma_hist) + 1e-3       # 保证非负！

        # DataProcessExpert
        y_hist_norm = self.data_proc.normalize_with_params(
            y_hist, mu_hist, sigma_hist
        )                                              # (B,L_hist)

        # Sundial backbone 在处理后的 y_hist_norm上预测
        with torch.no_grad() if self.flags.freeze_backbone else torch.enable_grad():
            pred_norm = self.backbone.generate(
                y_hist_norm,
                max_new_tokens=self.backbone_config.output_token_lens[0],
            ).mean(dim=1)                                    # (B,L_pred)
        if pred_norm.dim() == 1:
            pred_norm = pred_norm.unsqueeze(0)
        pred_norm = pred_norm[:, :L_pred]             # (B,L_pred)

        # 逆处理
        y_pred = self.data_proc.denormalize_future_with_params(
            y_future_norm=pred_norm,
            mu_hist=mu_hist,
            sigma_hist=sigma_hist,
            L_pred=L_pred,
        )                                              # (B,L_pred)

        aux = {
            "H_y": H_y,
            "H_var_attn": H_var_attn,
            "delta_mu_var_hist": delta_mu_var_hist,
            "delta_logsigma_var_hist": delta_logsigma_var_hist,
            "delta_mu_hist": delta_mu_hist,
            "delta_logsigma_hist": delta_logsigma_hist,
            "mu_base_hist": mu_base_hist,
            "sigma_base_hist": sigma_base_hist,
            "mu_hist": mu_hist,
            "sigma_hist": sigma_hist,
            "w_vars": w_vars,
            "gate_logits": gate_logits,
            "y_hist_norm": y_hist_norm,
            "pred_norm": pred_norm,
        }

        aux_loss = None
        if self.training and (y_future is not None):
            # y_future 的真实 patch 统计
            mu_true_fut, sigma_true_fut = self.data_proc.compute_patch_stats(y_future)  # (B,N_true)
            N_true = mu_true_fut.shape[1]

            # 把 μ_hist/σ_hist 在 patch 维度插值到未来 patch 数 N_true
            mu_pred_fut = F.interpolate(
                mu_hist.unsqueeze(1), size=N_true, mode="linear", align_corners=False
            ).squeeze(1)  # (B,N_true)
            sigma_pred_fut = F.interpolate(
                sigma_hist.unsqueeze(1), size=N_true, mode="linear", align_corners=False
            ).squeeze(1)

            loss_mu = F.mse_loss(mu_pred_fut, mu_true_fut)
            loss_sigma = F.mse_loss(sigma_pred_fut, sigma_true_fut)
            aux_loss = loss_mu + loss_sigma

            aux.update({
                "mu_true_fut": mu_true_fut,
                "sigma_true_fut": sigma_true_fut,
                "mu_pred_fut": mu_pred_fut,
                "sigma_pred_fut": sigma_pred_fut,
                "aux_loss_mu": loss_mu.detach().item(),
                "aux_loss_sigma": loss_sigma.detach().item(),
            })

        if return_aux:
            return y_pred.unsqueeze(-1), aux, aux_loss
        else:
            return y_pred.unsqueeze(-1)
        
        
def param_size_mb(model: torch.nn.Module):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / 1024**2
        
        
if __name__ == '__main__':
    from torch.profiler import profile, record_function, ProfilerActivity
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先清一遍缓存，避免历史占用干扰
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ===== 1) 构建一个“小模型 + 小输入” =====
    # 这里的 input_size 要等于你的协变量个数 D
    input_size =  128
    flags = PatchMoEFlags(use_value_expert=True, freeze_backbone=True)

    print_mem("before model")

    model = CovAwareVarWisePatchMoERegressor(
        input_size=input_size,
        sundial_name='thuml/sundial-base-128m',
        flags=flags,
    ).to(device)

    print_mem("after model.to(cuda)")
    
    print(f"Model parameters memory ~= {param_size_mb(model):.1f} MB")

    # 用比正常训练小很多的 window 做一次测试
    B_small = 64
    L_hist_small = 2880   # 小历史窗口
    L_pred_small = 720    # 小预测窗口
    D = input_size

    X_small = torch.randn(B_small, L_pred_small, D, device=device)
    y_hist_small = torch.randn(B_small, L_hist_small, 1, device=device)
    y_future_small = torch.randn(B_small, L_pred_small, 1, device=device)

    torch.cuda.reset_peak_memory_stats()
    print_mem("before forward")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("model_inference"):
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred, aux, aux_loss = model(
                    X_small, y_hist_small, y_future=y_future_small, return_aux=True
                )

    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=20
    ))

    print_mem("after forward (no grad)")
    print("peak allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")
