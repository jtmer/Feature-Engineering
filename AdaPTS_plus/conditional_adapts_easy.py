from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PredPack:
    mean: np.ndarray   # (B,1,H)
    lb:   np.ndarray
    ub:   np.ndarray


class SimplePatchAdaPTS:
    """
      - 逐 patch encode (y,x) -> z_past
      - LTM 在 latent 空间做 long-horizon 预测 -> z_future_pred
      - 逐 patch decode (z_future_pred, x_future) -> y_future_pred
    只在 latent 上加：
      1) 重建 y_past
      2) y_future_pred 的 MSE
      3) latent_smoothness_loss(past) (+ 可选 future_true)
      4) 可选 latent 对齐：z_future_pred ~ z_future_true
    """
    def __init__(
        self,
        adapter: nn.Module,   # SmoothPatchAdapter
        iclearner,
        device: str = "cuda",
    ):
        self.adapter = adapter
        self.iclearner = iclearner
        self.device = torch.device(device)
        self.adapter.to(self.device)

    @torch.no_grad()
    def _predict_future_latents_with_ltm(
        self,
        past_latents: torch.Tensor,  # (B,Cz,Np_past)
        future_horizon_tokens: int,  # Np_future
        ltm_batch_size: int = 64,
        **ltm_kwargs,
    ) -> torch.Tensor:
        """
        返回: future_latents_pred (B,Cz,Np_future)
        注意：这里的 horizon 是 patch token 的长度，而不是时间点个数。
        """
        past_latents_ctx = past_latents.detach().contiguous().clone()

        # Sundial ICL：把每个 channel 当作一条独立序列
        self.iclearner.update_context(
            time_series=past_latents_ctx,
            context_length=past_latents_ctx.shape[-1],
        )

        per_channel_outputs = self.iclearner.predict_long_horizon(
            prediction_horizon=future_horizon_tokens,
            batch_size=ltm_batch_size,
            verbose=0,
            **ltm_kwargs,
        )

        predicted_channels = []
        num_latent_channels = past_latents.shape[1]
        for ch in range(num_latent_channels):
            channel_pred = per_channel_outputs[ch].predictions  # (B,1,Np_future)
            channel_pred = channel_pred.to(self.device, dtype=torch.float32)

            if channel_pred.shape[-1] > future_horizon_tokens:
                channel_pred = channel_pred[..., :future_horizon_tokens]
            elif channel_pred.shape[-1] < future_horizon_tokens:
                pad_len = future_horizon_tokens - channel_pred.shape[-1]
                channel_pred = F.pad(channel_pred, (0, pad_len))

            predicted_channels.append(channel_pred)

        future_latents_pred = torch.cat(predicted_channels, dim=1)  # (B,Cz,Np_future)
        return future_latents_pred

    # ---------- 一些辅助 loss ----------
    def _latent_smoothness_loss(self, z: torch.Tensor) -> torch.Tensor:
        return self.adapter.latent_smoothness_loss(z)

    def _latent_alignment_loss(self, z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
        """
        让 LTM 预测到的 latent 分布与直接 encode(y_future,x_future) 的 latent 接近。
        """
        return F.mse_loss(z_pred, z_true)

    # ---------- evaluate ----------
    @torch.no_grad()
    def evaluate(
        self,
        past_target: np.ndarray,
        past_covariates: np.ndarray,
        future_target: np.ndarray,
        future_covariates: np.ndarray,
        batch_size: int = 128,
        ltm_batch_size: int = 64,
        lambda_past_recon: float = 1.0,
        lambda_future_pred: float = 1.0,
        lambda_latent_smooth: float = 0.1,
        lambda_latent_align: float = 0.1,
    ) -> Dict[str, float]:

        self.adapter.eval()
        self.iclearner.eval()

        dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metric_sums = {
            "loss": 0.0,
            "past_recon": 0.0,
            "future_pred": 0.0,
            "latent_smooth": 0.0,
            "latent_align": 0.0,
        }
        total_count = 0

        P = self.adapter.patch_size

        for batch_past_y, batch_past_x, batch_future_y, batch_future_x in dataloader:
            batch_past_y = batch_past_y.to(self.device)      # (B,1,L)
            batch_past_x = batch_past_x.to(self.device)      # (B,Cx,L)
            batch_future_y = batch_future_y.to(self.device)  # (B,1,H)
            batch_future_x = batch_future_x.to(self.device)  # (B,Cx,H)

            B, _, L = batch_past_y.shape
            H = batch_future_y.shape[-1]
            assert L % P == 0 and H % P == 0
            Np_past = L // P
            Np_future = H // P

            # encode
            z_past = self.adapter.encode(batch_past_y, batch_past_x)          # (B,Cz,Np_past)
            z_future_true = self.adapter.encode(batch_future_y, batch_future_x)  # (B,Cz,Np_future)

            # 1) past reconstruction
            y_past_recon = self.adapter.decode(z_past, batch_past_x)         # (B,1,L)
            loss_past = F.mse_loss(y_past_recon, batch_past_y)

            # 2) LTM -> future_latents_pred
            z_future_pred = self._predict_future_latents_with_ltm(
                past_latents=z_past,
                future_horizon_tokens=Np_future,
                ltm_batch_size=ltm_batch_size,
            )

            # 3) decode future
            y_future_pred = self.adapter.decode(z_future_pred, batch_future_x)  # (B,1,H)
            loss_future = F.mse_loss(y_future_pred, batch_future_y)

            # 4) latent smoothness（past + future_true）
            loss_smooth_past = self._latent_smoothness_loss(z_past)
            loss_smooth_future = self._latent_smoothness_loss(z_future_true)
            loss_smooth = 0.5 * (loss_smooth_past + loss_smooth_future)

            # 5) latent 对齐：预测 latent vs 真正 encode 出来的 future latent
            loss_align = self._latent_alignment_loss(z_future_pred, z_future_true)

            total_loss = (
                lambda_past_recon * loss_past
                + lambda_future_pred * loss_future
                + lambda_latent_smooth * loss_smooth
                + lambda_latent_align * loss_align
            )

            bsz = batch_past_y.size(0)
            total_count += bsz
            metric_sums["loss"]          += total_loss.item() * bsz
            metric_sums["past_recon"]    += loss_past.item() * bsz
            metric_sums["future_pred"]   += loss_future.item() * bsz
            metric_sums["latent_smooth"] += loss_smooth.item() * bsz
            metric_sums["latent_align"]  += loss_align.item() * bsz

        return {k: v / total_count for k, v in metric_sums.items()}

    # ---------- 训练 ----------
    def train_adapter(
        self,
        past_target: np.ndarray,
        past_covariates: np.ndarray,
        future_target: np.ndarray,
        future_covariates: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lambda_past_recon: float = 1.0,
        lambda_future_pred: float = 1.0,
        lambda_latent_smooth: float = 0.1,
        lambda_latent_align: float = 0.1,
        ltm_batch_size: int = 64,
        verbose: bool = True,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        use_swanlab: bool = False,
        swanlab_run = None,
    ):
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

        best_val = float("inf")
        best_epoch = -1
        best_state = None
        bad_count = 0
        patience = 10

        P = self.adapter.patch_size

        for epoch in range(n_epochs):
            self.adapter.train()

            metric_sums = {
                "loss": 0.0,
                "past_recon": 0.0,
                "future_pred": 0.0,
                "latent_smooth": 0.0,
                "latent_align": 0.0,
            }
            total_count = 0

            last_past = last_future = last_smooth = last_align = 0.0

            for batch_past_y, batch_past_x, batch_future_y, batch_future_x in dataloader:
                batch_past_y = batch_past_y.to(self.device)
                batch_past_x = batch_past_x.to(self.device)
                batch_future_y = batch_future_y.to(self.device)
                batch_future_x = batch_future_x.to(self.device)

                B, _, L = batch_past_y.shape
                H = batch_future_y.shape[-1]
                assert L % P == 0 and H % P == 0
                Np_past = L // P
                Np_future = H // P

                # encode
                z_past = self.adapter.encode(batch_past_y, batch_past_x)
                z_future_true = self.adapter.encode(batch_future_y, batch_future_x)

                # 1) past reconstruction
                y_past_recon = self.adapter.decode(z_past, batch_past_x)
                loss_past = F.mse_loss(y_past_recon, batch_past_y)

                # 2) LTM in latent space
                with torch.no_grad():
                    z_future_pred = self._predict_future_latents_with_ltm(
                        past_latents=z_past,
                        future_horizon_tokens=Np_future,
                        ltm_batch_size=ltm_batch_size,
                    )

                # 3) decode future
                y_future_pred = self.adapter.decode(z_future_pred, batch_future_x)
                loss_future = F.mse_loss(y_future_pred, batch_future_y)

                # 4) latent smooth
                loss_smooth_past = self._latent_smoothness_loss(z_past)
                loss_smooth_future = self._latent_smoothness_loss(z_future_true)
                loss_smooth = 0.5 * (loss_smooth_past + loss_smooth_future)

                # 5) latent alignment
                loss_align = self._latent_alignment_loss(z_future_pred.detach(), z_future_true)

                total_loss = (
                    lambda_past_recon * loss_past
                    + lambda_future_pred * loss_future
                    + lambda_latent_smooth * loss_smooth
                    + lambda_latent_align * loss_align
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                bsz = batch_past_y.size(0)
                total_count += bsz
                metric_sums["loss"]          += total_loss.item() * bsz
                metric_sums["past_recon"]    += loss_past.item() * bsz
                metric_sums["future_pred"]   += loss_future.item() * bsz
                metric_sums["latent_smooth"] += loss_smooth.item() * bsz
                metric_sums["latent_align"]  += loss_align.item() * bsz

                last_past   = loss_past.item()
                last_future = loss_future.item()
                last_smooth = loss_smooth.item()
                last_align  = loss_align.item()

            train_metrics = {k: v / total_count for k, v in metric_sums.items()}

            val_metrics = None
            if val_data is not None:
                val_metrics = self.evaluate(
                    past_target=val_data["past_target"],
                    past_covariates=val_data["past_covariates"],
                    future_target=val_data["future_target"],
                    future_covariates=val_data["future_covariates"],
                    batch_size=batch_size,
                    ltm_batch_size=ltm_batch_size,
                    lambda_past_recon=lambda_past_recon,
                    lambda_future_pred=lambda_future_pred,
                    lambda_latent_smooth=lambda_latent_smooth,
                    lambda_latent_align=lambda_latent_align,
                )
                current_val = val_metrics["loss"]
            else:
                current_val = train_metrics["loss"]

            improved = (best_val - current_val) > 1e-4
            if improved:
                best_val = current_val
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}
                bad_count = 0
            else:
                bad_count += 1

            if use_swanlab and swanlab_run is not None:
                log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
                if val_metrics is not None:
                    log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_dict["epoch"] = epoch
                swanlab_run.log(log_dict)

            if verbose:
                msg = (
                    f"[simple-adapter] epoch={epoch:03d} "
                    f"loss={train_metrics['loss']:.6f} "
                    f"past={last_past:.4f} "
                    f"future={last_future:.4f} "
                    f"smooth={last_smooth:.4f} "
                    f"align={last_align:.4f}"
                )
                if val_metrics is not None:
                    msg += (
                        f" | val_loss={val_metrics['loss']:.6f} "
                        f"val_past={val_metrics['past_recon']:.4f} "
                        f"val_future={val_metrics['future_pred']:.4f}"
                    )
                print(msg)
                print(f"[earlystop] best={best_val:.6f} (epoch={best_epoch}) bad={bad_count}/{patience}")

            if bad_count >= patience:
                if verbose:
                    print(f"[earlystop] stop at epoch={epoch}, best_epoch={best_epoch}, best_loss={best_val:.6f}")
                break

        if best_state is not None:
            self.adapter.load_state_dict(best_state)
            if verbose:
                print(f"[earlystop] restored best adapter weights from epoch={best_epoch}")

    # ---------- 推理 ----------
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

        self.adapter.train()   # 保持 dropout 等随机性
        self.iclearner.eval()

        past_y = torch.tensor(past_target, dtype=torch.float32, device=self.device)
        past_x = torch.tensor(past_covariates, dtype=torch.float32, device=self.device)
        future_x = torch.tensor(future_covariates, dtype=torch.float32, device=self.device)

        P = self.adapter.patch_size
        H = pred_horizon
        assert H % P == 0
        Np_future = H // P

        all_samples = []

        for s in range(n_samples):
            with torch.no_grad():
                z_past = self.adapter.encode(past_y, past_x)  # (B,Cz,Np_past)

                z_future_pred = self._predict_future_latents_with_ltm(
                    past_latents=z_past,
                    future_horizon_tokens=Np_future,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )

                y_future_pred = self.adapter.decode(z_future_pred, future_x)  # (B,1,H)

                if s == 0:
                    print("---------------- SimplePatchAdaPTS.predict ----------------")
                    print("z_past   std:", z_past.std().item())
                    print("z_future std:", z_future_pred.std().item())
                    print("y_pred   std:", y_future_pred.std().item())

                all_samples.append(y_future_pred.detach().cpu().numpy())

        all_samples = np.stack(all_samples, axis=0)  # (S,B,1,H)
        mean = all_samples.mean(axis=0)
        std = all_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
