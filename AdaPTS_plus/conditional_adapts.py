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
    def __init__(
        self,
        adapter: nn.Module,
        iclearner,
        device: str = "cpu",
    ):
        self.adapter = adapter
        self.iclearner = iclearner
        self.device = torch.device(device)
        self.adapter.to(self.device)

    @torch.no_grad()
    def _predict_future_latents_with_ltm(
        self,
        past_latents: torch.Tensor,          # (B, z_dim, L)
        future_horizon: int,                # H
        ltm_batch_size: int = 64,
        **ltm_kwargs,
    ) -> torch.Tensor:
        """
        return: future_latents_pred (B, z_dim, H)
        """
        self.iclearner.update_context(
            time_series=copy.copy(past_latents),
            context_length=past_latents.shape[-1],
        )

        per_channel_outputs = self.iclearner.predict_long_horizon(
            prediction_horizon=future_horizon,
            batch_size=ltm_batch_size,
            verbose=0,
            **ltm_kwargs,
        )

        predicted_channels = []
        num_latent_channels = past_latents.shape[1]

        for channel_idx in range(num_latent_channels):
            channel_pred = per_channel_outputs[channel_idx].predictions  # torch.Tensor (B,1,H) in your trainer
            channel_pred = channel_pred.to(self.device, dtype=torch.float32)

            # safety: force length match
            if channel_pred.shape[-1] > future_horizon:
                channel_pred = channel_pred[..., :future_horizon]
            elif channel_pred.shape[-1] < future_horizon:
                pad_len = future_horizon - channel_pred.shape[-1]
                channel_pred = F.pad(channel_pred, (0, pad_len))

            predicted_channels.append(channel_pred)

        future_latents_pred = torch.cat(predicted_channels, dim=1)  # (B, z_dim, H)
        return future_latents_pred

    # def train_adapter_reconstruct_past(
    #     self,
    #     y_past: np.ndarray,   # (N,1,L)
    #     x_past: np.ndarray,   # (N,Cx,L)
    #     n_epochs: int = 50,
    #     batch_size: int = 64,
    #     lr: float = 1e-3,
    #     weight_decay: float = 0.0,
    #     coeff_recon: float = 1.0,
    #     coeff_kl: float = 1.0,
    #     verbose: int = 1,
    # ):
    #     ds = TensorDataset(
    #         torch.tensor(y_past, dtype=torch.float32),
    #         torch.tensor(x_past, dtype=torch.float32),
    #     )
    #     dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    #     self.adapter.train()
    #     opt = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

    #     for ep in range(n_epochs):
    #         total = 0.0
    #         for yb, xb in dl:
    #             yb = yb.to(self.device)
    #             xb = xb.to(self.device)

    #             out = self.adapter.encode(yb, xb)
    #             y_hat = self.adapter.decode(out.z, xb)

    #             recon = F.mse_loss(y_hat, yb)
    #             loss = coeff_recon * recon + coeff_kl * out.kl

    #             opt.zero_grad()
    #             loss.backward()
    #             opt.step()
    #             total += loss.item() * yb.size(0)

    #         if verbose:
    #             print(f"[adapter-pretrain] epoch={ep:03d} loss={total/len(ds):.6f}")
    
    def train_adapter(
        self,
        past_target: np.ndarray,        # (N,1,L)   y_past
        past_covariates: np.ndarray,    # (N,Cx,L)  x_past
        future_target: np.ndarray,      # (N,1,H)   y_future
        future_covariates: np.ndarray,  # (N,Cx,H)  x_future
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lambda_past_recon: float = 1.0,
        lambda_ltm_consistency: float = 1.0,
        ltm_batch_size: int = 1,
        verbose: bool = True,
    ):
        """
        1) 重建:  y_past ≈ decode( encode(y_past,x_past), x_past )
        2) 分布一致:  encode(y_future,x_future) ≈ LTM( encode(y_past,x_past) )
        """

        dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.adapter.train()
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        # LTM frozen
        self.iclearner.eval()

        for epoch in range(n_epochs):
            epoch_loss_sum = 0.0

            last_loss_past = 0.0
            last_loss_future_teacher = 0.0
            last_loss_ltm = 0.0

            for (
                batch_past_target,
                batch_past_covariates,
                batch_future_target,
                batch_future_covariates,
            ) in dataloader:

                batch_past_target = batch_past_target.to(self.device)
                batch_past_covariates = batch_past_covariates.to(self.device)

                past_latents = self.adapter.encode(batch_past_target, batch_past_covariates)
                future_latents_true = self.adapter.encode(batch_future_target, batch_future_covariates)

                # past reconstruction
                past_target_recon = self.adapter.decode(past_latents, batch_past_covariates)
                loss_past_recon = F.mse_loss(past_target_recon, batch_past_target)

                # LTM consistency ----
                with torch.no_grad():
                    future_latents_pred = self._predict_future_latents_with_ltm(
                        past_latents=past_latents,
                        future_horizon=batch_future_target.shape[-1],
                        ltm_batch_size=ltm_batch_size,
                    )
                    
                # TODO 这里mse点对点的loss是否不太对？改成对均值、方差等统计量的约束怎么样？
                loss_ltm_consistency = F.mse_loss(future_latents_pred, future_latents_true)

                total_loss = (
                    lambda_past_recon * loss_past_recon
                    + lambda_ltm_consistency * loss_ltm_consistency
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss_sum += total_loss.item() * batch_past_target.size(0)

                last_loss_past = loss_past_recon.item()
                last_loss_ltm = loss_ltm_consistency.item()

            if verbose:
                epoch_loss = epoch_loss_sum / len(dataset)
                print(
                    f"[adapter-train] epoch={epoch:03d} "
                    f"loss={epoch_loss:.6f} "
                    f"past_recon={last_loss_past:.4f} "
                    f"ltm_consistency={last_loss_ltm:.4f}"
                )

    def predict(
        self,
        past_target: np.ndarray,        # (B,1,L)
        past_covariates: np.ndarray,    # (B,Cx,L)
        future_covariates: np.ndarray,  # (B,Cx,H)
        pred_horizon: int,              # H
        ltm_batch_size: int = 128,
        n_samples: int = 20,
        **ltm_kwargs,
    ) -> PredPack:

        self.adapter.train()   # keep stochasticity if adapter has dropout etc.
        self.iclearner.eval()

        past_target_tensor = torch.tensor(past_target, dtype=torch.float32, device=self.device)
        past_covariates_tensor = torch.tensor(past_covariates, dtype=torch.float32, device=self.device)
        future_covariates_tensor = torch.tensor(future_covariates, dtype=torch.float32, device=self.device)

        all_future_target_samples = []

        for sample_idx in range(n_samples):
            with torch.no_grad():
                past_latent_out = self.adapter.encode(past_target_tensor, past_covariates_tensor)
                past_latents = past_latent_out

                future_latents_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )

                if sample_idx == 0:
                    print("\n==== latent stats ====")
                    print("past_latents   min/mean/max:", past_latents.min().item(), past_latents.mean().item(), past_latents.max().item())
                    print("past_latents   std:", past_latents.std().item())
                    print("future_cov     min/mean/max:", future_covariates_tensor.min().item(), future_covariates_tensor.mean().item(), future_covariates_tensor.max().item())
                    print("future_latents min/mean/max:", future_latents_pred.min().item(), future_latents_pred.mean().item(), future_latents_pred.max().item())
                    print("future_latents std:", future_latents_pred.std().item())

                future_target_pred = self.adapter.decode(future_latents_pred, future_covariates_tensor)  # (B,1,H)
                all_future_target_samples.append(future_target_pred.detach().cpu().numpy())

        all_future_target_samples = np.stack(all_future_target_samples, axis=0)  # (S,B,1,H)
        mean = all_future_target_samples.mean(axis=0)
        std = all_future_target_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
