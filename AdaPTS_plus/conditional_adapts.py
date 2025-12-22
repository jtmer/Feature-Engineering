# conditional_adapts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@dataclass
class PredPack:
    mean: np.ndarray   # (B,1,H)
    lb:   np.ndarray
    ub:   np.ndarray


class ConditionalAdaPTS:
    """
    - 训练 adapter: y_past,x_past -> z_past -> decode(z_past,x_past) 重建 y_past
    - 推理:  y_past,x_past -> z_past -> FM(z_past)-> z_future_pred -> decode(z_future_pred, x_future) -> y_future_pred
    """

    def __init__(
        self,
        adapter: nn.Module,          # ConditionalPatchVAEAdapter
        iclearner,                   # adapts.icl.iclearner.* (MomentICLTrainer / MoiraiICLTrainer)
        device: str = "cpu",
    ):
        self.adapter = adapter
        self.iclearner = iclearner
        self.device = torch.device(device)
        self.adapter.to(self.device)

    @torch.no_grad()
    def _fm_predict_z(self, z_past: torch.Tensor, horizon_tokens: int, batch_size: int = 64, **kwargs) -> torch.Tensor:
        self.iclearner.update_context(
            time_series=copy.copy(z_past),
            context_length=z_past.shape[-1],
        )
        icl_objs = self.iclearner.predict_long_horizon(
            prediction_horizon=horizon_tokens,
            batch_size=batch_size,
            verbose=0,
            **kwargs,
        )

        preds = []
        Cz = z_past.shape[1]
        for k in range(Cz):
            p = icl_objs[k].predictions  # 你 trainer 保证是 torch.Tensor (B,1,H)
            p = p.to(self.device, dtype=torch.float32)

            # 对齐长度（保险）
            if p.shape[-1] > horizon_tokens:
                p = p[..., :horizon_tokens]
            elif p.shape[-1] < horizon_tokens:
                pad = horizon_tokens - p.shape[-1]
                p = torch.nn.functional.pad(p, (0, pad))

            preds.append(p)

        z_future = torch.cat(preds, dim=1)  # (B,Cz,Htok)
        return z_future

    def train_adapter_reconstruct_past(
        self,
        y_past: np.ndarray,   # (N,1,L)
        x_past: np.ndarray,   # (N,Cx,L)
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        coeff_recon: float = 1.0,
        coeff_kl: float = 1.0,    # 这里 adapter 内部已有 beta_kl；再乘一层方便你扫参
        verbose: int = 1,
    ):
        ds = TensorDataset(
            torch.tensor(y_past, dtype=torch.float32),
            torch.tensor(x_past, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.adapter.train()
        opt = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        for ep in range(n_epochs):
            total = 0.0
            for yb, xb in dl:
                yb = yb.to(self.device)
                xb = xb.to(self.device)

                out = self.adapter.encode(yb, xb)
                y_hat = self.adapter.decode(out.z, xb)

                recon = F.mse_loss(y_hat, yb)
                loss = coeff_recon * recon + coeff_kl * out.kl

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * yb.size(0)

            if verbose:
                print(f"[adapter-pretrain] epoch={ep:03d} loss={total/len(ds):.6f}")

    def predict(
        self,
        y_past: np.ndarray,          # (B,1,L)
        x_past: np.ndarray,          # (B,Cx,L)
        x_future: np.ndarray,        # (B,Cx,H)  (如果没有未来协变量，你可以先用 cov forecaster 产出)
        pred_horizon: int,           # H (timestep)
        fm_batch_size: int = 128,
        n_samples: int = 20,
        **fm_kwargs,
    ) -> PredPack:

        self.adapter.train()  # 为了让 VAE 的采样产生不确定性（类似你原版用 dropout 采样）:contentReference[oaicite:3]{index=3}
        self.iclearner.eval()

        y_past_t = torch.tensor(y_past, dtype=torch.float32, device=self.device)
        x_past_t = torch.tensor(x_past, dtype=torch.float32, device=self.device)
        x_future_t = torch.tensor(x_future, dtype=torch.float32, device=self.device)

        # token horizon：如果 patch_size>1，FM 预测的是 token 数 = H//P
        patch_size = getattr(self.adapter, "patch_size", 1)
        if patch_size > 1:
            assert pred_horizon % patch_size == 0
            horizon_tokens = pred_horizon // patch_size
        else:
            horizon_tokens = pred_horizon

        all_y = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = self.adapter.encode(y_past_t, x_past_t)              # z_past
                z_future = self._fm_predict_z(out.z, horizon_tokens, batch_size=fm_batch_size, **fm_kwargs)
                
                if _ == 0:  # 第一次采样打印
                    print("\n==== z stats ====")
                    print("z_past  min/mean/max:", out.z.min().item(), out.z.mean().item(), out.z.max().item())
                    print("z_past  std:", out.z.std().item())
                    print("x_future min/mean/max:", x_future_t.min().item(), x_future_t.mean().item(), x_future_t.max().item())
                    print("z_future min/mean/max:", z_future.min().item(), z_future.mean().item(), z_future.max().item())
                    print("z_future std:", z_future.std().item())
                
                
                y_future_hat = self.adapter.decode(z_future, x_future_t)   # (B,1,H)
                all_y.append(y_future_hat.detach().cpu().numpy())

        all_y = np.stack(all_y, axis=0)  # (S,B,1,H)
        mean = all_y.mean(axis=0)
        std = all_y.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
