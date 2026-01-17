from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B, C, L)
    return: (B, C*patch_size, Np) where Np = L // patch_size
    """
    if patch_size <= 1:
        return x
    B, C, L = x.shape
    assert L % patch_size == 0, f"L={L} must be divisible by patch_size={patch_size}"
    Np = L // patch_size
    x = x.view(B, C, Np, patch_size)                 # (B, C, Np, P)
    x = x.permute(0, 2, 1, 3).contiguous()           # (B, Np, C, P)
    x = x.view(B, Np, C * patch_size)                # (B, Np, C*P)
    x = x.permute(0, 2, 1).contiguous()              # (B, C*P, Np)
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Inverse of patchify when you want to recover per-timestep outputs.
    Here we use a simple "repeat within patch" strategy for per-timestep y.
    x: (B, Cy, Np)
    return: (B, Cy, L= Np*patch_size)
    """
    if patch_size <= 1:
        return x
    B, Cy, Np = x.shape
    x = x.unsqueeze(-1).repeat(1, 1, 1, patch_size)  # (B, Cy, Np, P)
    x = x.reshape(B, Cy, Np * patch_size)            # (B, Cy, L)
    return x


class SmoothPatchAdapter(nn.Module):
    """
    逐 patch encoder-decoder，不显式用均值/方差。
    - encode:  (y, x) -> z_patch  (B, Cz, Np)
    - decode:  (z_patch, x) -> y_hat  (B, 1, T)
    同时提供 latent_smoothness_loss，用于约束 patch 间的 z 统计更平滑。
    """
    def __init__(
        self,
        cov_dim: int,       # Cx
        z_dim: int,         # Cz
        patch_size: int = 1,
        hidden_dim: int = 256,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cov_dim = cov_dim
        self.z_dim = z_dim
        self.patch_size = patch_size

        # encoder: 每个 patch 的输入维度
        enc_in_dim = (1 + cov_dim) * patch_size if patch_size > 1 else (1 + cov_dim)
        dec_cov_dim = cov_dim * patch_size if patch_size > 1 else cov_dim

        self.encoder = MLP(
            in_dim=enc_in_dim,
            out_dim=z_dim,
            hidden_dim=hidden_dim,
            num_layers=enc_layers,
            dropout=dropout,
        )

        self.decoder = MLP(
            in_dim=z_dim + dec_cov_dim,
            out_dim=patch_size,
            hidden_dim=hidden_dim,
            num_layers=dec_layers,
            dropout=dropout,
        )

    # -------- 编码 / 解码 --------
    def encode(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        y: (B,1,L)
        x: (B,Cx,L)
        return:
          z: (B,Cz,Np)  Np=L//patch_size
        """
        assert y.dim() == 3 and x.dim() == 3
        B, _, L = y.shape
        assert x.shape[0] == B and x.shape[2] == L

        if self.patch_size > 1:
            y_tok = patchify(y, self.patch_size)        # (B,1*P,Np)
            x_tok = patchify(x, self.patch_size)        # (B,Cx*P,Np)
            Np = y_tok.shape[-1]
            inp = torch.cat([y_tok, x_tok], dim=1)      # (B,(1+Cx)*P,Np)
        else:
            Np = L
            inp = torch.cat([y, x], dim=1)              # (B,1+Cx,L)

        # (B,Cin,Np) -> (B*Np,Cin)
        B, Cin, Np = inp.shape
        inp2 = inp.permute(0, 2, 1).contiguous().view(B * Np, Cin)
        h = self.encoder(inp2)                          # (B*Np,Cz)
        z = h.view(B, Np, self.z_dim).permute(0, 2, 1).contiguous()  # (B,Cz,Np)
        return z

    def decode(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        z: (B,Cz,Np)
        x: (B,Cx,T)  T = Np*patch_size
        return:
          y_hat: (B,1,T)
        """
        B, Cz, Np = z.shape
        P = self.patch_size
        T = x.shape[-1]
        assert T % P == 0 and T // P == Np, f"T={T}, P={P}, Np={Np} mismatch"

        if P > 1:
            x_tok = patchify(x, P)                      # (B,Cx*P,Np)
            cov2 = x_tok.permute(0, 2, 1).contiguous().view(B * Np, -1) # （B*Np,Cx*P）
        else:
            cov2 = x.permute(0, 2, 1).contiguous().view(B * Np, -1)

        z2 = z.permute(0, 2, 1).contiguous().view(B * Np, Cz)   # (B*Np,Cz)
        dec_in = torch.cat([z2, cov2], dim=-1)
        h = self.decoder(dec_in)                                # (B*Np,P)
        y_hat = h.view(B, Np, P).reshape(B, 1, Np * P)          # (B,1,T)
        return y_hat

    # -------- latent 平滑约束 --------
    def latent_smoothness_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,Cz,Np)

        做两层约束：
        1) 相邻 patch 的 z 差值要小（L2）
        2) 以 patch 为单位看：每个 patch 的 mean/std（在 channel 维上）在时间轴上的方差要小
        """
        if z.size(-1) <= 1:
            return z.new_tensor(0.0)

        # 1) 相邻 patch 的 L2 平滑
        diff = z[:, :, 1:] - z[:, :, :-1]              # (B,Cz,Np-1)
        loss_adj = (diff ** 2).mean()

        # 2) patch 级 mean/std 在时间维上方差小
        #   对每个 patch t: 先在 channel 维上算 mean/std -> (B,Np)
        mu_patch = z.mean(dim=1)                       # (B,Np)
        std_patch = z.std(dim=1)                       # (B,Np)

        # 再对时间维求方差（越小越说明各 patch 统计相近）
        var_mu = mu_patch.var(dim=-1, unbiased=False).mean()
        var_std = std_patch.var(dim=-1, unbiased=False).mean()

        return loss_adj + var_mu + var_std
