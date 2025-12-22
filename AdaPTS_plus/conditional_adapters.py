# conditional_adapters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


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


@dataclass
class AdapterOutput:
    z: torch.Tensor                 # (B, Cz, Lz or Np)
    kl: torch.Tensor                # scalar


class ConditionalPatchVAEAdapter(nn.Module):
    """
    Adapter = encoder(y_past, x_past) -> z_past
           + decoder(z, x_cov) -> y_hat (past recon OR future forecast)
    - 支持 patch 粒度：先 patchify，再在 patch token 维度上建模。
    - z 是“代理变量序列”，将喂给 foundation model（FM）。
    """

    def __init__(
        self,
        cov_dim: int,            # Cx
        z_dim: int,              # Cz (代理通道数)
        patch_size: int = 1,
        hidden_dim: int = 256,
        enc_layers: int = 2,
        dec_layers: int = 2,
        beta_kl: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cov_dim = cov_dim
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.beta_kl = beta_kl

        # encoder 输入： [y(1), x(Cx)] -> (1+Cx) channels
        self.enc_in_dim = (1 + cov_dim) * patch_size if patch_size > 1 else (1 + cov_dim)
        self.dec_cov_dim = cov_dim * patch_size if patch_size > 1 else cov_dim

        # token-wise VAE: q(z|token) = N(mu, sigma)
        self.encoder = MLP(self.enc_in_dim, hidden_dim, hidden_dim=hidden_dim, num_layers=enc_layers, dropout=dropout)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

        # decoder: p(y|z, cov)
        self.decoder = MLP(z_dim + self.dec_cov_dim, hidden_dim, hidden_dim=hidden_dim, num_layers=dec_layers, dropout=dropout)
        self.y_head = nn.Linear(hidden_dim, patch_size)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def encode(self, y_past: torch.Tensor, x_past: torch.Tensor) -> AdapterOutput:
        """
        y_past: (B, 1, L)
        x_past: (B, Cx, L)
        return z: (B, Cz, Lz) where Lz = L//patch_size if patch_size>1 else L
        """
        assert y_past.dim() == 3 and x_past.dim() == 3
        B, _, L = y_past.shape
        assert x_past.shape[0] == B and x_past.shape[2] == L

        if self.patch_size > 1:
            y_tok = patchify(y_past, self.patch_size)        # (B, 1*P, Np)
            x_tok = patchify(x_past, self.patch_size)        # (B, Cx*P, Np)
            Np = y_tok.shape[-1]
            inp = torch.cat([y_tok, x_tok], dim=1)           # (B, (1+Cx)*P, Np)
        else:
            Np = L
            inp = torch.cat([y_past, x_past], dim=1)         # (B, 1+Cx, L)

        # token-wise MLP: (B, Cin, Np) -> (B, Np, Cin) -> (B*Np, Cin)
        inp2 = inp.permute(0, 2, 1).contiguous().view(B * Np, -1)
        h = self.encoder(inp2)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)                  # (B*Np, Cz)

        # KL per token
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl = self.beta_kl * kl

        z = z.view(B, Np, self.z_dim).permute(0, 2, 1).contiguous()   # (B, Cz, Np)
        return AdapterOutput(z=z, kl=kl)

    def decode(self, z: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        """
        z:     (B, Cz, Np)  (如果 patch_size==1，则 Np=L)
        x_cov: (B, Cx, L)   (协变量在时间维仍是原始 L)
        return y_hat: (B, 1, L)
        """
        B, Cz, Np = z.shape
        assert Cz == self.z_dim

        if self.patch_size > 1:
            x_tok = patchify(x_cov, self.patch_size)         # (B, Cx*P, Np)
            assert x_tok.shape[-1] == Np
            cov2 = x_tok.permute(0, 2, 1).contiguous().view(B * Np, -1)  # (B*Np, Cx*P)
        else:
            # patch_size=1: x_cov token数 = L = Np
            assert x_cov.shape[-1] == Np
            cov2 = x_cov.permute(0, 2, 1).contiguous().view(B * Np, -1)  # (B*Np, Cx)

        z2 = z.permute(0, 2, 1).contiguous().view(B * Np, Cz)            # (B*Np, Cz)
        dec_in = torch.cat([z2, cov2], dim=-1)
        h = self.decoder(dec_in)
        y_hat = self.y_head(h).view(B, Np, self.patch_size).reshape(B, 1, Np * self.patch_size)  # (B,1,L)

        # # token -> timestep
        # y_hat = unpatchify(y_tok, self.patch_size)  # (B,1,L)
        return y_hat

class ConditionalPatchPCAAdapter(nn.Module):
    """
    非 VAE 版本：PCA encoder + 条件 decoder
    encode(y_past,x_past) -> z_past（patch token 级）
    decode(z, x_cov) -> y_hat（token -> timestep）
    """
    def __init__(self, cov_dim: int, z_dim: int, patch_size: int = 1,
                 hidden_dim: int = 256, dec_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.cov_dim = cov_dim
        self.z_dim = z_dim
        self.patch_size = patch_size

        self.enc_in_dim = (1 + cov_dim) * patch_size if patch_size > 1 else (1 + cov_dim)
        self.dec_cov_dim = cov_dim * patch_size if patch_size > 1 else cov_dim

        self.pca = PCA(n_components=z_dim)
        self._pca_fitted = False

        self.decoder = MLP(z_dim + self.dec_cov_dim, hidden_dim, hidden_dim=hidden_dim, num_layers=dec_layers, dropout=dropout)
        self.y_head = nn.Linear(hidden_dim, patch_size)

    def fit_pca(self, y_past_np: np.ndarray, x_past_np: np.ndarray):
        """
        y_past_np: (N,1,L)
        x_past_np: (N,Cx,L)
        PCA 在 token 样本上 fit：样本数 = N * Np
        """
        y_t = torch.tensor(y_past_np, dtype=torch.float32)
        x_t = torch.tensor(x_past_np, dtype=torch.float32)

        if self.patch_size > 1:
            y_tok = patchify(y_t, self.patch_size)  # (N, 1*P, Np)
            x_tok = patchify(x_t, self.patch_size)  # (N, Cx*P, Np)
            inp = torch.cat([y_tok, x_tok], dim=1)  # (N, (1+Cx)*P, Np)
        else:
            inp = torch.cat([y_t, x_t], dim=1)      # (N, 1+Cx, L)

        N, Cin, Np = inp.shape
        X = inp.permute(0,2,1).reshape(N*Np, Cin).numpy()  # (N*Np, Cin)

        self.pca.fit(X)
        self._pca_fitted = True

    def encode(self, y_past: torch.Tensor, x_past: torch.Tensor) -> AdapterOutput:
        assert self._pca_fitted, "Call fit_pca(...) before encode"
        B, _, L = y_past.shape

        if self.patch_size > 1:
            y_tok = patchify(y_past, self.patch_size)
            x_tok = patchify(x_past, self.patch_size)
            inp = torch.cat([y_tok, x_tok], dim=1)   # (B, Cin, Np)
        else:
            inp = torch.cat([y_past, x_past], dim=1) # (B, Cin, L)

        B, Cin, Np = inp.shape
        X = inp.permute(0,2,1).reshape(B*Np, Cin).detach().cpu().numpy()
        Z = self.pca.transform(X).astype(np.float32)        # (B*Np, z_dim)

        z = torch.tensor(Z, device=y_past.device).view(B, Np, self.z_dim).permute(0,2,1).contiguous()
        return AdapterOutput(z=z, kl=torch.tensor(0.0, device=y_past.device))

    def decode(self, z: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        B, Cz, Np = z.shape
        assert Cz == self.z_dim

        if self.patch_size > 1:
            x_tok = patchify(x_cov, self.patch_size)  # (B, Cx*P, Np)
            cov2 = x_tok.permute(0,2,1).reshape(B*Np, -1)
        else:
            assert x_cov.shape[-1] == Np
            cov2 = x_cov.permute(0,2,1).reshape(B*Np, -1)

        z2 = z.permute(0,2,1).reshape(B*Np, Cz)
        dec_in = torch.cat([z2, cov2], dim=-1)

        h = self.decoder(dec_in)
        y_hat = self.y_head(h).view(B, Np, self.patch_size).reshape(B, 1, Np * self.patch_size)  # (B,1,L)
        # y_hat = unpatchify(y_tok, self.patch_size)  # (B,1,L)
        return y_hat