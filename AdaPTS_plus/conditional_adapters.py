# conditional_adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


def revin_patch_stats_target(target_series: torch.Tensor, patch_size: int, eps: float = 1e-5):
    """
    y: (B,1,T)
    return:
      mu_t:   (B,1,T)   expanded per timestep
      std_t:  (B,1,T)
      mu_p:   (B,1,Np)  per patch
      std_p:  (B,1,Np)
    """
    assert target_series.dim() == 3 and target_series.size(1) == 1
    B, _, T = target_series.shape
    assert T % patch_size == 0, f"T={T} must be divisible by patch_size={patch_size}"
    Np = T // patch_size

    y4 = target_series.view(B, 1, Np, patch_size)             # (B,1,Np,P)
    mu_p = y4.mean(dim=-1)                                    # (B,1,Np)
    std_p = y4.std(dim=-1)                                    # (B,1,Np)

    mu_t = mu_p.unsqueeze(-1).repeat(1, 1, 1, patch_size).view(B, 1, T)
    std_t = std_p.unsqueeze(-1).repeat(1, 1, 1, patch_size).view(B, 1, T)

    std_t = std_t + eps
    std_p = std_p + eps
    return mu_t, std_t, mu_p, std_p

def revin_patch_norm_target(target_series: torch.Tensor, patch_size: int, eps: float = 1e-5):
    """
    target_series: (B,1,T)
    returns:
      target_norm: (B,1,T)
      mu_t/std_t:  (B,1,T)
      mu_p/std_p:  (B,1,Np)
    """
    mu_t, std_t, mu_p, std_p = revin_patch_stats_target(target_series, patch_size, eps=eps)
    target_norm = (target_series - mu_t) / std_t
    return target_norm, mu_t, std_t, mu_p, std_p


def revin_patch_denorm_target(target_norm: torch.Tensor, mu_t: torch.Tensor, std_t: torch.Tensor):
    return target_norm * std_t + mu_t


def expand_patch_to_time(patch_vals: torch.Tensor, patch_size: int, T: int):
    """
    patch_vals: (B,1,Np)
    return:     (B,1,T)
    """
    B, C, Np = patch_vals.shape
    assert C == 1
    assert Np * patch_size == T
    return patch_vals.unsqueeze(-1).repeat(1, 1, 1, patch_size).view(B, 1, T)

# def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
#     """
#     x: (B, C, L)
#     return: (B, C*patch_size, Np) where Np = L // patch_size
#     """
#     if patch_size <= 1:
#         return x
#     B, C, L = x.shape
#     assert L % patch_size == 0, f"L={L} must be divisible by patch_size={patch_size}"
#     Np = L // patch_size
#     x = x.view(B, C, Np, patch_size)                 # (B, C, Np, P)
#     x = x.permute(0, 2, 1, 3).contiguous()           # (B, Np, C, P)
#     x = x.view(B, Np, C * patch_size)                # (B, Np, C*P)
#     x = x.permute(0, 2, 1).contiguous()              # (B, C*P, Np)
#     return x


# def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
#     """
#     Inverse of patchify when you want to recover per-timestep outputs.
#     Here we use a simple "repeat within patch" strategy for per-timestep y.
#     x: (B, Cy, Np)
#     return: (B, Cy, L= Np*patch_size)
#     """
#     if patch_size <= 1:
#         return x
#     B, Cy, Np = x.shape
#     x = x.unsqueeze(-1).repeat(1, 1, 1, patch_size)  # (B, Cy, Np, P)
#     x = x.reshape(B, Cy, Np * patch_size)            # (B, Cy, L)
#     return x

class BaseEncoder(nn.Module):
    def forward(self, target_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def forward(self, latent_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    """
    No-op encoder:
      z := y_norm (and keep channels = 1)
    """
    def __init__(self, latent_dim: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim != 1:
            raise ValueError("IdentityEncoder only supports latent_dim=1 (y itself as latent).")

    def forward(self, target_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return target_series
    
class IdentityDecoder(BaseDecoder):
    """
    No-op decoder:
      y_hat_norm := z
    """
    def __init__(self, latent_dim: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim != 1:
            raise ValueError("IdentityDecoder only supports latent_dim=1.")

    def forward(self, latent_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return latent_series
    
class BaseNormalizer(nn.Module):
    """
    All normalization semantics are centralized here to avoid ambiguity.
    """
    def norm_past(self, y_past: torch.Tensor, *, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def future_true_stats(self, y_future: torch.Tensor, *, patch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def norm_future_with_true_stats(self, y_future: torch.Tensor, *, mu_t_true: torch.Tensor, std_t_true: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def mix_future_time_stats(
        self,
        *,
        mu_t_true: Optional[torch.Tensor],
        std_t_true: Optional[torch.Tensor],
        mu_t_hat: torch.Tensor,
        std_t_hat: torch.Tensor,
        alpha_stats: float,
        std_floor: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def denorm_future_pred(self, *, y_future_pred_norm: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target_norm_ref(self, *, y_future: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def future_pred_loss(
        self,
        *,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityNormalizer(BaseNormalizer):
    """
    No normalization:
      y_norm = y
      mu = 0, std = 1
    No stats mixing: mu_ref=0 std_ref=1 always (ignore predictor) -> removes ambiguity.
    """
    def norm_past(self, y_past: torch.Tensor, *, patch_size: int):
        mu = torch.zeros_like(y_past)
        std = torch.ones_like(y_past)
        return y_past, mu, std

    def future_true_stats(self, y_future: torch.Tensor, *, patch_size: int):
        # provide "true stats" for completeness (patch moment loss may still use target_norm_ref=y_future)
        mu_t = torch.zeros_like(y_future)
        std_t = torch.ones_like(y_future)

        # patch stats from raw y (still well-defined)
        B, _, T = y_future.shape
        assert T % patch_size == 0
        Np = T // patch_size
        y4 = y_future.view(B, 1, Np, patch_size)
        mu_p = y4.mean(dim=-1)
        std_p = y4.std(dim=-1) + 1e-5
        return mu_t, std_t, mu_p, std_p

    def norm_future_with_true_stats(self, y_future: torch.Tensor, *, mu_t_true: torch.Tensor, std_t_true: torch.Tensor):
        return y_future

    def mix_future_time_stats(
        self,
        *,
        mu_t_true: Optional[torch.Tensor],
        std_t_true: Optional[torch.Tensor],
        mu_t_hat: torch.Tensor,
        std_t_hat: torch.Tensor,
        alpha_stats: float,
        std_floor: float = 0.05,
    ):
        # ignore all inputs: identity means "no re-scaling"
        mu_ref = torch.zeros_like(mu_t_hat)
        std_ref = torch.ones_like(std_t_hat)
        return mu_ref, std_ref

    def denorm_future_pred(self, *, y_future_pred_norm: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor):
        return y_future_pred_norm

    def target_norm_ref(self, *, y_future: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor):
        return y_future

    def future_pred_loss(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, y_pred_norm: torch.Tensor, y_true_norm: torch.Tensor):
        return F.mse_loss(y_pred_norm, y_true_norm)

class RevINPatchNormalizer(BaseNormalizer):
    """
    Patch-level RevIN:
      - past: revin_patch_norm_target
      - future: true stats = revin_patch_stats_target
      - mixing: interpolate in log-std space, (1-alpha)*true + alpha*hat
      - denorm: use mu_ref/std_ref
      - loss: compare in raw space for stability & interpretability
    """
    def norm_past(self, y_past: torch.Tensor, *, patch_size: int):
        y_norm, mu_t, std_t, _, _ = revin_patch_norm_target(y_past, patch_size=patch_size)
        return y_norm, mu_t, std_t

    def future_true_stats(self, y_future: torch.Tensor, *, patch_size: int):
        mu_t_true, std_t_true, mu_p_true, std_p_true = revin_patch_stats_target(y_future, patch_size=patch_size)
        return mu_t_true, std_t_true, mu_p_true, std_p_true

    def norm_future_with_true_stats(self, y_future: torch.Tensor, *, mu_t_true: torch.Tensor, std_t_true: torch.Tensor):
        return (y_future - mu_t_true) / (std_t_true + 1e-5)

    def mix_future_time_stats(
        self,
        *,
        mu_t_true: Optional[torch.Tensor],
        std_t_true: Optional[torch.Tensor],
        mu_t_hat: torch.Tensor,
        std_t_hat: torch.Tensor,
        alpha_stats: float,
        std_floor: float = 0.05,
    ):
        assert mu_t_true is not None and std_t_true is not None, "RevINPatchNormalizer requires true stats."
        mu_mix = (1.0 - alpha_stats) * mu_t_true + alpha_stats * mu_t_hat

        logstd_true = torch.log(std_t_true + 1e-5)
        logstd_hat = torch.log(std_t_hat + 1e-5)
        logstd_mix = (1.0 - alpha_stats) * logstd_true + alpha_stats * logstd_hat
        std_mix = torch.exp(logstd_mix)
        std_mix = torch.clamp(std_mix, min=std_floor)
        return mu_mix, std_mix

    def denorm_future_pred(self, *, y_future_pred_norm: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor):
        return revin_patch_denorm_target(y_future_pred_norm, mu_ref, std_ref)

    def target_norm_ref(self, *, y_future: torch.Tensor, mu_ref: torch.Tensor, std_ref: torch.Tensor):
        return (y_future - mu_ref) / (std_ref + 1e-5)

    def future_pred_loss(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, y_pred_norm: torch.Tensor, y_true_norm: torch.Tensor):
        return F.mse_loss(y_pred, y_true)
    

# class MLP(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.0):
#         super().__init__()
#         layers = []
#         d = in_dim
#         for _ in range(max(1, num_layers - 1)):
#             layers += [nn.Linear(d, hidden_dim), nn.GELU()]
#             if dropout > 0:
#                 layers += [nn.Dropout(dropout)]
#             d = hidden_dim
#         layers += [nn.Linear(d, out_dim)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)


# @dataclass
# class AdapterOutput:
#     z: torch.Tensor                 # (B, Cz, Lz or Np)
#     kl: torch.Tensor                # scalar

class PatchStatsPredictor(nn.Module):
    def __init__(
        self,
        cov_dim: int,
        patch_size: int,
        hidden_dim: int = 128,
        context_layers: int = 2,
        stats_hidden_dim: int | None = None,
        logstd_min: float = -4.0,
        logstd_max: float = 2.0,
    ):
        super().__init__()
        self.cov_dim = cov_dim
        self.patch_size = patch_size
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        if stats_hidden_dim is None:
            stats_hidden_dim = hidden_dim // 2

        # (B,Cx,T) -> (B,hidden,Np)
        self.down = nn.Conv1d(
            in_channels=cov_dim,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.act = nn.GELU()

        # 对每个 patch 内做 mean/std/max/min
        # (B,Cx,T) -> (B,4*Cx,Np) -> (B,stats_hidden_dim,Np)
        self.stats_proj = nn.Conv1d(
            in_channels=4 * cov_dim,
            out_channels=stats_hidden_dim,
            kernel_size=1,
        )

        # 在 Np 维度上做 kernel=3 的 conv
        context_channels = hidden_dim + stats_hidden_dim
        context_blocks = []
        in_ch = context_channels
        for _ in range(max(1, context_layers)):
            context_blocks.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=context_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            context_blocks.append(nn.GELU())
            in_ch = context_channels
        self.context_net = nn.Sequential(*context_blocks)

        self.mu_head = nn.Conv1d(context_channels, 1, kernel_size=1)
        self.logstd_head = nn.Conv1d(context_channels, 1, kernel_size=1)

    def _compute_patch_stats(self, covariates_series: torch.Tensor, eps: float = 1e-5):
        """
        covariates_series: (B,Cx,T)
        返回 per-patch 的 mean/std/max/min 拼在 channel 维:
          (B,4*Cx,Np)
        """
        B, Cx, T = covariates_series.shape
        P = self.patch_size
        assert T % P == 0, f"T={T} must be divisible by patch_size={P}"
        Np = T // P

        # (B,Cx,T) -> (B,Cx,Np,P)
        x4 = covariates_series.view(B, Cx, Np, P)

        mean_p = x4.mean(dim=-1)                 # (B,Cx,Np)
        std_p  = x4.std(dim=-1) + eps            # (B,Cx,Np)
        max_p  = x4.max(dim=-1).values           # (B,Cx,Np)
        min_p  = x4.min(dim=-1).values           # (B,Cx,Np)

        # 堆成 (B,4,Cx,Np) 再合并成 channel 维 (B,4*Cx,Np)
        stats = torch.stack([mean_p, std_p, max_p, min_p], dim=2)  # (B,4,Cx,Np)
        stats = stats.view(B, 4 * Cx, Np)                          # (B,4*Cx,Np)
        return stats

    def forward(self, covariates_series: torch.Tensor, eps: float = 1e-5):
        """
        return:
          mu_patch:      (B,1,Np)
          std_patch:     (B,1,Np)  > 0
          logstd_patch:  (B,1,Np)  = log(std_patch)
        """
        B, Cx, T = covariates_series.shape
        P = self.patch_size
        assert T % P == 0, f"T={T} must be divisible by patch_size={P}"
        Np = T // P

        h_down = self.act(self.down(covariates_series))  # (B,hidden,Np)

        stats_raw = self._compute_patch_stats(covariates_series, eps=eps)  # (B,4*Cx,Np)
        stats_feat = self.act(self.stats_proj(stats_raw))                  # (B,stats_hidden_dim,Np)

        h = torch.cat([h_down, stats_feat], dim=1)  # (B,hidden+stats_hidden_dim,Np)
        h = self.context_net(h)                     # (B,context_channels,Np)

        mu_patch = self.mu_head(h)                 # (B,1,Np)
        logstd_raw = self.logstd_head(h)           # (B,1,Np)
        logstd_clamped = torch.clamp(logstd_raw, min=self.logstd_min, max=self.logstd_max)

        std_patch = torch.exp(logstd_clamped) + eps  # (B,1,Np), > 0
        return mu_patch, std_patch, logstd_clamped
    
# class ResTemporalBlock(nn.Module):
#     def __init__(self, ch: int, k: int = 5, dropout: float = 0.0):
#         super().__init__()
#         self.dw = nn.Conv1d(ch, ch, kernel_size=k, padding=k//2, groups=ch)
#         self.pw = nn.Conv1d(ch, ch, kernel_size=1)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x):
#         h = self.dw(x)
#         h = self.act(h)
#         h = self.pw(h)
#         h = self.drop(h)
#         return x + h
    
class NeuralLatentEncoder(nn.Module):
    """
    - encode: (y, x) -> z
    """
    def __init__(
        self,
        covariates_dim: int,
        latent_dim: int,
        revin_patch_size_past: int,
        hidden_dim: int = 256,
        encoder_layers: int = 2,
        dropout: float = 0.0,
        normalize_latents: bool = False,
    ):
        super().__init__()
        self.covariates_dim = covariates_dim
        self.latent_dim = latent_dim
        self.revin_patch_size_past = revin_patch_size_past

        covariates_dim_aug = covariates_dim
        encoder_input_channels = 1 + covariates_dim_aug

        def build_temporal_encoder(input_channels: int, output_channels: int, num_layers: int) -> nn.Sequential:
            layers = []
            ch = input_channels
            for i in range(max(1, num_layers)):
                layers.append(nn.Conv1d(ch, hidden_dim, kernel_size=1))
                layers.append(nn.GELU())

                # layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim))
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
                layers.append(nn.GELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                ch = hidden_dim

            layers.append(nn.Conv1d(ch, output_channels, kernel_size=1))
            return nn.Sequential(*layers)

        # (B, 1+Cx, L) -> (B, z_dim, L)
        self.latent_encoder = build_temporal_encoder(
            input_channels=encoder_input_channels,
            output_channels=latent_dim,
            num_layers=encoder_layers,
        )

        # self.normalize_latents = normalize_latents
        self.normalize_latents = False
        self.latent_normalizer = (
            nn.GroupNorm(num_groups=1, num_channels=latent_dim) if normalize_latents else nn.Identity()
        )

    def forward(self, target_series: torch.Tensor, covariates_series: torch.Tensor):
        """
        target_series:     (B,1,L)
        covariates_series: (B,Cx,L)
        returns latents:   (B,Cz,L)
        """
        device = next(self.parameters()).device
        target_series = target_series.to(device)
        covariates_series = covariates_series.to(device)

        encoder_inputs = torch.cat([target_series, covariates_series], dim=1)  # (B,1+Cx,L)
        latent_series = self.latent_encoder(encoder_inputs)                    # (B,Cz,L)
        latent_series = self.latent_normalizer(latent_series)
        return latent_series


class NeuralTargetDecoder(nn.Module):
    """
    - decode: (z, x) -> y_hat
    """
    def __init__(
        self,
        covariates_dim: int,
        revin_patch_size_future: int,
        latent_dim: int,
        hidden_dim: int = 256,
        decoder_layers: int = 2,
        dropout: float = 0.0,
        normalize_latents: bool = False,
    ):
        super().__init__()
        self.covariates_dim = covariates_dim
        self.latent_dim = latent_dim
        
        self.revin_patch_size_future = revin_patch_size_future

        covariates_dim_aug = covariates_dim
        decoder_input_channels = latent_dim + covariates_dim_aug

        def build_temporal_decoder(input_channels: int, output_channels: int, num_layers: int) -> nn.Sequential:
            layers = []
            ch = input_channels
            for i in range(max(1, num_layers)):
                layers.append(nn.Conv1d(ch, hidden_dim, kernel_size=1))
                layers.append(nn.GELU())

                # 时域混合（depthwise）+ 残差
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim))
                # layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1))
                layers.append(nn.GELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                ch = hidden_dim

            layers.append(nn.Conv1d(ch, output_channels, kernel_size=1))
            return nn.Sequential(*layers)

        # (B, z_dim+Cx, L) -> (B, 1, L)
        self.target_decoder = build_temporal_decoder(
            input_channels=decoder_input_channels,
            output_channels=1,
            num_layers=decoder_layers,
        )

        # self.normalize_latents = normalize_latents
        self.normalize_latents = False
        self.latent_normalizer = (
            nn.GroupNorm(num_groups=1, num_channels=latent_dim) if normalize_latents else nn.Identity()
        )

    # def decode(self, latent_series: torch.Tensor, covariates_series: torch.Tensor, logstd_t: torch.Tensor):
    def forward(self, latent_series: torch.Tensor, covariates_series: torch.Tensor):
        """
        latent_series:     (B,Cz,L)
        covariates_series: (B,Cx,L)
        returns target_hat:(B,1,L)
        """
        device = next(self.parameters()).device
        latent_series = latent_series.to(device)
        covariates_series = covariates_series.to(device)
        latent_series = self.latent_normalizer(latent_series)

        decoder_inputs = torch.cat([latent_series, covariates_series], dim=1)  # (B,Cz+Cx,L)
        target_hat = self.target_decoder(decoder_inputs)                       # (B,1,L)
        return target_hat
        # out = self.target_decoder(decoder_inputs)   # (B,2,T)
        # base = out[:, 0:1, :]
        # res  = torch.tanh(out[:, 1:2, :])           # [-1,1] 限幅
        # # return base, res

        # logstd_ref = logstd_t.detach()
        # amp = 0.2 + 0.8 * torch.sigmoid(logstd_ref) # (B,1,H)，范围 [0.2,1.0]
        # target_pred_norm = base + amp * res
        # return target_pred_norm


class NeuralEncoder(BaseEncoder):
    def __init__(self, core: NeuralLatentEncoder):
        super().__init__()
        self.core = core

    def forward(self, target_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return self.core(target_series, covariates_series)


class NeuralDecoder(BaseDecoder):
    def __init__(self, core: NeuralTargetDecoder):
        super().__init__()
        self.core = core

    def forward(self, latent_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return self.core(latent_series, covariates_series)
    
class PipelineAdapter(nn.Module):
    def __init__(
        self,
        *,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        normalizer: BaseNormalizer,
        latent_dim: int,
        revin_patch_size_past: int,
        revin_patch_size_future: int,
        future_stats_predictor: Optional[PatchStatsPredictor] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.normalizer = normalizer

        self.latent_dim = latent_dim
        self.revin_patch_size_past = revin_patch_size_past
        self.revin_patch_size_future = revin_patch_size_future

        self.future_stats_predictor = future_stats_predictor

        # capability flags (used by main.py to skip stages)
        self.has_stats_predictor = (self.future_stats_predictor is not None) and isinstance(self.normalizer, RevINPatchNormalizer)
        self.is_trainable = any(p.requires_grad for p in self.parameters())
        self.can_reconstruct_past = True  # even identity can, but training may be skipped by is_trainable
        if isinstance(self.encoder, IdentityEncoder) and isinstance(self.decoder, IdentityDecoder):
            self.is_trainable = False
            self.can_reconstruct_past = False

    def encode(self, target_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return self.encoder(target_series, covariates_series)

    def decode(self, latent_series: torch.Tensor, covariates_series: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_series, covariates_series)

    def predict_future_stats(self, future_covariates: torch.Tensor):
        # If no predictor or identity normalizer, return zeros/ones to avoid ambiguity.
        B, _, H = future_covariates.shape
        device = future_covariates.device
        mu_t_hat = torch.zeros((B, 1, H), device=device, dtype=future_covariates.dtype)
        std_t_hat = torch.ones((B, 1, H), device=device, dtype=future_covariates.dtype)

        P = self.revin_patch_size_future
        Np = H // P
        mu_p_hat = torch.zeros((B, 1, Np), device=device, dtype=future_covariates.dtype)
        std_p_hat = torch.ones((B, 1, Np), device=device, dtype=future_covariates.dtype)
        logstd_p_hat = torch.zeros((B, 1, Np), device=device, dtype=future_covariates.dtype)

        if self.future_stats_predictor is None:
            return mu_t_hat, std_t_hat, mu_p_hat, std_p_hat, logstd_p_hat

        mu_p_hat, std_p_hat, logstd_p_hat = self.future_stats_predictor(future_covariates)
        mu_t_hat = expand_patch_to_time(mu_p_hat, P, H)
        std_t_hat = expand_patch_to_time(std_p_hat, P, H)
        return mu_t_hat, std_t_hat, mu_p_hat, std_p_hat, logstd_p_hat


# =========================
# Factory
# =========================

def build_pipeline_adapter(
    *,
    cov_dim: int,
    z_dim: int,
    patch_size: int,
    pred_window: int,
    encoder_type: str = "neural",
    decoder_type: str = "neural",
    normalizer: str = "identity",
) -> PipelineAdapter:
    if normalizer == "identity":
        norm = IdentityNormalizer()
        stats_pred = None
    elif normalizer == "revin_patch":
        norm = RevINPatchNormalizer()
        stats_pred = PatchStatsPredictor(cov_dim=cov_dim, patch_size=patch_size, hidden_dim=128)
    else:
        raise ValueError(f"Unknown normalizer={normalizer}")

        encoder_core = None
    decoder_core = None

    if encoder_type == "neural":
        encoder_core = NeuralLatentEncoder(
            covariates_dim=cov_dim,
            latent_dim=z_dim,
            revin_patch_size_past=patch_size,
            hidden_dim=256,
            encoder_layers=2,
            dropout=0.0,
            normalize_latents=False,
        )

    if decoder_type == "neural":
        decoder_core = NeuralTargetDecoder(
            covariates_dim=cov_dim,
            latent_dim=z_dim,
            revin_patch_size_future=patch_size,
            hidden_dim=256,
            decoder_layers=2,
            dropout=0.0,
            normalize_latents=False,
        )

    if encoder_type == "none":
        enc = IdentityEncoder(latent_dim=z_dim)
    elif encoder_type == "neural":
        assert encoder_core is not None
        enc = NeuralEncoder(encoder_core)
    else:
        raise ValueError(f"Unknown encoder_type={encoder_type}")

    if decoder_type == "none":
        dec = IdentityDecoder(latent_dim=z_dim)
    elif decoder_type == "neural":
        assert decoder_core is not None
        dec = NeuralDecoder(decoder_core)
    else:
        raise ValueError(f"Unknown decoder_type={decoder_type}")

    adapter = PipelineAdapter(
        encoder=enc,
        decoder=dec,
        normalizer=norm,
        latent_dim=z_dim,
        revin_patch_size_past=patch_size,
        revin_patch_size_future=patch_size,
        future_stats_predictor=stats_pred,
    )
    return adapter