# conditional_adapts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from conditional_adapters import (
    revin_patch_norm_target,
    revin_patch_denorm_target,
    revin_patch_stats_target,
    expand_patch_to_time,
)

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
        
    def _revin_norm_target(self, target_series: torch.Tensor, eps: float = 1e-5):
        # target_series: (B,1,T)
        mu = target_series.mean(dim=-1, keepdim=True)
        std = target_series.std(dim=-1, keepdim=True)
        target_norm = (target_series - mu) / (std + eps)
        return target_norm, mu, std

    def _revin_denorm_target(self, target_norm: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        return target_norm * std + mu

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
        past_latents_ctx = past_latents.detach()
        past_latents_ctx = past_latents_ctx.contiguous().clone()
        
        self.iclearner.update_context(
            time_series=past_latents_ctx,
            context_length=past_latents_ctx.shape[-1],
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
    
    def _latent_stats_loss(self, past_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
        """
        past_latents/future_latents: (B, z_dim, T)
        weak distribution alignment: match per-channel mean/std over time axis
        """
        past_mu = past_latents.mean(dim=-1)
        future_mu = future_latents.mean(dim=-1)
        past_std = past_latents.std(dim=-1)
        future_std = future_latents.std(dim=-1)
        return F.mse_loss(past_mu, future_mu) + F.mse_loss(past_std, future_std)

    def _augment_covariates_with_revin_stats(
        self,
        covariates: torch.Tensor,   # (B,Cx,T)
        past_mu: torch.Tensor,      # (B,1,1)
        past_std: torch.Tensor,     # (B,1,1)
    ):
        # repeat to time length
        T = covariates.shape[-1]
        mu_feat = past_mu.repeat(1, 1, T)    # (B,1,T)
        std_feat = past_std.repeat(1, 1, T)  # (B,1,T)
        return torch.cat([covariates, mu_feat, std_feat], dim=1)  # (B, Cx+2, T)
    
    def pretrain_stats_predictor(
        self,
        future_target: np.ndarray,      # (N,1,H)
        future_covariates: np.ndarray,  # (N,Cx,H)
        n_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        patience: int = 10,
        verbose: bool = True,
        use_swanlab: bool = False,
        swanlab_run = None,
    ):
        """
        只训练 adapter.future_stats_predictor:
          x_future -> (mu_patch_hat, std_patch_hat)
        目标: 贴近 revin_patch_stats_target 得到的真实 patch 均值/方差
        """
        self.adapter.train()
        
        # 只优化 stats predictor 的参数
        optimizer = torch.optim.Adam(
            self.adapter.future_stats_predictor.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        P_fut = self.adapter.revin_patch_size_future
        
        train_dataset = TensorDataset(
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if val_data is not None:
            val_future_target = val_data["future_target"]
            val_future_covariates = val_data["future_covariates"]
            val_dataset = TensorDataset(
                torch.tensor(val_future_target, dtype=torch.float32),
                torch.tensor(val_future_covariates, dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
            
        best_val_loss = float("inf")
        best_epoch = -1
        best_state_dict = None
        bad_count = 0
        
        def _run_one_epoch(dataloader, is_train: bool):
            """
            返回平均 loss_stats_pred (即 patch 级 mu/logstd 的拟合误差)
            """
            if is_train:
                self.adapter.train()
            else:
                self.adapter.eval()
            
            total_loss = 0.0
            total_count = 0
            
            for batch_future_target, batch_future_cov in dataloader:
                batch_future_target = batch_future_target.to(self.device)   # (B,1,H)
                batch_future_cov    = batch_future_cov.to(self.device)      # (B,Cx,H)
                
                # 真实 patch 统计
                _, _, mu_p_true, std_p_true = revin_patch_stats_target(
                    batch_future_target, patch_size=P_fut
                )  # mu/std_p_true: (B,1,Np)
                logstd_p_true = torch.log(std_p_true + 1e-5)
                
                # 预测 patch 统计
                mu_p_hat, std_p_hat, logstd_p_hat = self.adapter.future_stats_predictor(batch_future_cov)
                
                # 和 train_adapter 中保持一致的损失形式
                logstd_hat = torch.log(std_p_hat + 1e-5)
                logstd_true = torch.log(std_p_true + 1e-5)
                loss_mu    = F.mse_loss(mu_p_hat, mu_p_true)
                # loss_std   = F.mse_loss(std_p_hat, std_p_true)
                loss_std = F.mse_loss(logstd_hat, logstd_true)
                
                # print('std:true vs pred', logstd_true, logstd_hat)
                
                loss = loss_mu + 2*loss_std
                
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                bsz = batch_future_target.size(0)
                total_loss  += loss.item() * bsz
                total_count += bsz
            
            return total_loss / max(1, total_count)
        
        for epoch in range(n_epochs):
            train_loss = _run_one_epoch(train_loader, is_train=True)
            
            val_loss = None
            if val_loader is not None:
                with torch.no_grad():
                    val_loss = _run_one_epoch(val_loader, is_train=False)
            else:
                # 如果没有 val，就用 train_loss 监控 early stop（不建议，但保底）
                val_loss = train_loss
                
            improved = (best_val_loss - val_loss) > 1e-4
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.adapter.future_stats_predictor.state_dict().items()
                }
                bad_count = 0
            else:
                bad_count += 1
                
            if use_swanlab and swanlab_run is not None:
                log_dict = {
                    "stats_pretrain/train_loss": train_loss,
                    "stats_pretrain/val_loss":   val_loss,
                    "stats_pretrain/epoch":      epoch,
                }
                swanlab_run.log(log_dict)
            
            if verbose:
                msg = (
                    f"[stats-pretrain] epoch={epoch:03d} "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} "
                    f"best_val={best_val_loss:.6f} "
                    f"(best_epoch={best_epoch}) "
                    f"bad_count={bad_count}/{patience}"
                )
                print(msg)
            
            if bad_count >= patience:
                if verbose:
                    print(f"[stats-pretrain] early stop at epoch={epoch}, "
                          f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                break
        
        if best_state_dict is not None:
            self.adapter.future_stats_predictor.load_state_dict(best_state_dict)
            if verbose:
                print(f"[stats-pretrain] restored best future_stats_predictor "
                      f"from epoch={best_epoch} (val_loss={best_val_loss:.6f})")
        else:
            if verbose:
                print("[stats-pretrain] WARNING: no best_state_dict saved (maybe no epochs ran?)")
    
    @torch.no_grad()
    def evaluate(
        self,
        past_target: np.ndarray,
        past_covariates: np.ndarray,
        future_target: np.ndarray,
        future_covariates: np.ndarray,
        alpha_stats: float,
        batch_size: int = 128,
        ltm_batch_size: int = 64,
        lambda_past_recon: float = 1.0,
        lambda_future_pred: float = 1.0,
        lambda_latent_stats: float = 0.01,
        lambda_stats_pred: float = 0.1,   # stats predictor
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

        metric_sums = {"loss": 0.0, "past_recon": 0.0, "future_pred": 0.0, "latent_stats": 0.0, "stats_pred": 0.0,}
        total_count = 0
        
        P_past = self.adapter.revin_patch_size_past
        P_fut  = self.adapter.revin_patch_size_future

        for (
            batch_past_target,
            batch_past_covariates,
            batch_future_target,
            batch_future_covariates,
        ) in dataloader:

            batch_past_target = batch_past_target.to(self.device)
            batch_past_covariates = batch_past_covariates.to(self.device)
            batch_future_target = batch_future_target.to(self.device)
            batch_future_covariates = batch_future_covariates.to(self.device)
            
            # batch_past_target_norm, past_mu, past_std = self._revin_norm_target(batch_past_target)
            # batch_future_target_norm = (batch_future_target - past_mu) / (past_std + 1e-5)
            
            # batch_past_covariates_aug = self._augment_covariates_with_revin_stats(
            #     batch_past_covariates, past_mu, past_std
            # )
            # batch_future_covariates_aug = self._augment_covariates_with_revin_stats(
            #     batch_future_covariates, past_mu, past_std
            # )
            batch_past_target_norm, past_mu_t, past_std_t, past_mu_p, past_std_p = revin_patch_norm_target(
                batch_past_target, patch_size=P_past
            )
            # batch_past_covariates_aug = torch.cat([batch_past_covariates, past_mu_t.detach(), past_std_t.detach()], dim=1)  # (B,Cx+2,L)
            batch_past_covariates_aug = batch_past_covariates

            future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = revin_patch_stats_target(
                batch_future_target, patch_size=P_fut
            )
            future_mu_t_hat, future_std_t_hat, future_mu_p_hat, future_std_p_hat, future_logstd_p_hat = self.adapter.predict_future_stats(
                batch_future_covariates
            )
            
            future_mu_t_mix = (1.0 - alpha_stats) * future_mu_t_true + alpha_stats * future_mu_t_hat
            future_logstd_t_true = torch.log(future_std_t_true + 1e-5)
            future_logstd_t_hat  = torch.log(future_std_t_hat + 1e-5)
            future_logstd_t_mix  = (1.0 - alpha_stats) * future_logstd_t_true + alpha_stats * future_logstd_t_hat
            future_std_t_mix = torch.exp(future_logstd_t_mix)
            
            # batch_future_covariates_aug = torch.cat([batch_future_covariates, future_mu_t_hat, future_std_t_hat], dim=1)  # (B,Cx+2,H)
            # batch_future_covariates_aug = torch.cat([batch_future_covariates, future_mu_t_mix.detach(), future_std_t_mix.detach()], dim=1)
            batch_future_covariates_aug = batch_future_covariates

            # past_latents = self.adapter.encode(batch_past_target_norm, batch_past_covariates_aug)
            # future_latents_true = self.adapter.encode(batch_future_target_norm, batch_future_covariates_aug)
            past_latents = self.adapter.encode(batch_past_target_norm, batch_past_covariates_aug)
            batch_future_target_norm_true = (batch_future_target - future_mu_t_true) / (future_std_t_true + 1e-5)
            # batch_future_covariates_aug_true = torch.cat([batch_future_covariates, future_mu_t_true.detach(), future_std_t_true.detach()], dim=1)
            batch_future_covariates_aug_true = batch_future_covariates
            future_latents_true = self.adapter.encode(batch_future_target_norm_true, batch_future_covariates_aug_true)

            # past recon
            past_target_recon_norm = self.adapter.decode(past_latents, batch_past_covariates_aug)

            past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
            loss_past_recon = F.mse_loss(past_target_recon, batch_past_target)

            # future pred via LTM -> decoder
            future_latents_pred = self._predict_future_latents_with_ltm(
                past_latents=past_latents,
                future_horizon=batch_future_target.shape[-1],
                ltm_batch_size=ltm_batch_size,
            )
        
            #! mix还是hat还是true?    
            future_target_pred_norm = self.adapter.decode(future_latents_pred, batch_future_covariates_aug)
            mu_ref  = future_mu_t_mix.detach()
            std_ref = future_std_t_mix.detach()
            std_floor = 0.05
            std_ref = torch.clamp(std_ref, min=std_floor)
            batch_future_target_norm = (batch_future_target - mu_ref) / std_ref
            loss_future_pred = F.mse_loss(future_target_pred_norm, batch_future_target_norm)
            # future_target_pred = revin_patch_denorm_target(future_target_pred_norm, future_mu_t_mix, future_std_t_mix)
            # loss_future_pred = F.mse_loss(future_target_pred, batch_future_target)
            future_target_pred = revin_patch_denorm_target(
                future_target_pred_norm, future_mu_t_mix, std_ref
            )
            
            future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
            loss_stats_pred =  F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5*F.mse_loss(future_logstd_p_hat, future_logstd_p_true)
            
            print('evl: std pred vs true', future_std_p_hat[0], future_std_p_true[0])

            # weak latent stats alignment
            loss_latent_stats = self._latent_stats_loss(past_latents, future_latents_true)

            total_loss = (
                lambda_past_recon * loss_past_recon
                + lambda_future_pred * loss_future_pred
                + lambda_latent_stats * loss_latent_stats
                + lambda_stats_pred * loss_stats_pred
            )

            bsz = batch_past_target.size(0)
            total_count += bsz
            metric_sums["loss"] += total_loss.item() * bsz
            metric_sums["past_recon"] += loss_past_recon.item() * bsz
            metric_sums["future_pred"] += loss_future_pred.item() * bsz
            metric_sums["latent_stats"] += loss_latent_stats.item() * bsz
            metric_sums["stats_pred"] += loss_stats_pred.item() * bsz
            
            
            if 'printed_patch_stats' not in locals():
                printed_patch_stats = True

                P = P_fut
                B, _, H = future_target_pred.shape
                assert H % P == 0
                Np = H // P

                true4 = batch_future_target.view(B, 1, Np, P)         # (B,1,Np,P)
                pred4 = future_target_pred.view(B, 1, Np, P)

                mu_true_p = true4.mean(dim=-1)                        # (B,1,Np)
                mu_pred_p = pred4.mean(dim=-1)
                std_true_p = true4.std(dim=-1)
                std_pred_p = pred4.std(dim=-1)

                mu_diff = (mu_pred_p - mu_true_p).detach()
                std_diff = (std_pred_p - std_true_p).detach()

                print("\n===== Patch-level TARGET statistics (Evaluation) =====")
                print("true mean  (first seq first 5 patches):", mu_true_p[0,0,:5].cpu().numpy())
                print("pred mean  (first seq first 5 patches):", mu_pred_p[0,0,:5].cpu().numpy())
                print("mean abs diff  min/mean/max:",
                    mu_diff.abs().min().item(),
                    mu_diff.abs().mean().item(),
                    mu_diff.abs().max().item())

                print("\ntrue std   (first seq first 5 patches):", std_true_p[0,0,:5].cpu().numpy())
                print("pred std   (first seq first 5 patches):", std_pred_p[0,0,:5].cpu().numpy())
                print("std abs diff   min/mean/max:",
                    std_diff.abs().min().item(),
                    std_diff.abs().mean().item(),
                    std_diff.abs().max().item())
                print("=======================================================\n")


        return {k: v / total_count for k, v in metric_sums.items()}
    
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
        # lambda_ltm_consistency: float = 1.0,
        lambda_future_pred: float = 1.0,
        lambda_latent_stats: float = 0.1,
        lambda_stats_pred: float = 0.1,
        lambda_y_patch_std: float = 0.2,
        ltm_batch_size: int = 1,
        verbose: bool = True,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        use_swanlab: bool = True,
        swanlab_run = None,
    ):
        """
        1) 重建:  y_past ≈ decode( encode(y_past,x_past), x_past )
        2) 分布一致:  encode(y_future,x_future) ≈ LTM( encode(y_past,x_past) )
        3) y_future ≈ decode( LTM( encode(y_past,x_past) ), x_future )

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
        
        best_monitor_value = float("inf")
        best_epoch = -1
        best_state_dict = None
        bad_count = 0

        for epoch in range(n_epochs):
            self.adapter.train()

            metric_sums = {"loss": 0.0, "past_recon": 0.0, "future_pred": 0.0, "latent_stats": 0.0, "stats_pred": 0.0}
            total_count = 0

            last_loss_past = 0.0
            last_loss_future = 0.0
            last_loss_stats = 0.0
            
            P_past = self.adapter.revin_patch_size_past
            P_fut  = self.adapter.revin_patch_size_future
            
            # μmix​=(1−α)μ_true ​+ αμ_hat​, logσmix​=(1−α)logσ_true ​+ αlogσ_hat​从真实逐渐变成推理分布
            progress = epoch / max(1, n_epochs - 1)
            alpha_stats = float(np.clip((progress - 0.1) / 0.4, 0.0, 1.0))

            for (
                batch_past_target,
                batch_past_covariates,
                batch_future_target,
                batch_future_covariates,
            ) in dataloader:

                batch_past_target = batch_past_target.to(self.device)
                batch_past_covariates = batch_past_covariates.to(self.device)
                batch_future_target = batch_future_target.to(self.device)
                batch_future_covariates = batch_future_covariates.to(self.device)
                
                batch_past_target_norm, past_mu_t, past_std_t, past_mu_p, past_std_p = revin_patch_norm_target(
                    batch_past_target, patch_size=P_past
                )
                # batch_past_covariates_aug = torch.cat([batch_past_covariates, past_mu_t.detach(), past_std_t.detach()], dim=1)  # (B,Cx+2,L)
                batch_past_covariates_aug = batch_past_covariates
                future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = revin_patch_stats_target(
                    batch_future_target, patch_size=P_fut
                )
                future_mu_t_hat, future_std_t_hat, future_mu_p_hat, future_std_p_hat, future_logstd_p_hat = self.adapter.predict_future_stats(
                    batch_future_covariates
                )
                
                future_mu_t_mix = (1.0 - alpha_stats) * future_mu_t_true + alpha_stats * future_mu_t_hat.detach()
                future_logstd_t_true = torch.log(future_std_t_true + 1e-5)
                future_logstd_t_hat  = torch.log(future_std_t_hat + 1e-5)
                future_logstd_t_mix  = (1.0 - alpha_stats) * future_logstd_t_true + alpha_stats * future_logstd_t_hat
                future_std_t_mix = torch.exp(future_logstd_t_mix)
                
                # batch_future_covariates_aug = torch.cat(
                #     [batch_future_covariates, future_mu_t_hat, future_std_t_hat], dim=1
                # )  # (B,Cx+2,H)
                # batch_future_covariates_aug = torch.cat([batch_future_covariates, future_mu_t_mix.detach(), future_std_t_mix.detach()], dim=1)
                batch_future_covariates_aug = batch_future_covariates

                past_latents = self.adapter.encode(batch_past_target_norm, batch_past_covariates_aug)
                # future_latents_true = self.adapter.encode(batch_future_target_norm, batch_future_covariates_aug)
                batch_future_target_norm_true = (batch_future_target - future_mu_t_true) / (future_std_t_true + 1e-5)
                # batch_future_covariates_aug_true = torch.cat([batch_future_covariates, future_mu_t_true.detach(), future_std_t_true.detach()], dim=1)
                batch_future_covariates_aug_true = batch_future_covariates
                future_latents_true = self.adapter.encode(batch_future_target_norm_true, batch_future_covariates_aug_true)

                # past reconstruction
                past_target_recon_norm = self.adapter.decode(past_latents, batch_past_covariates_aug)
                loss_past_recon = F.mse_loss(past_target_recon_norm, batch_past_target_norm)

                # LTM consistency
                with torch.no_grad():
                    future_latents_pred = self._predict_future_latents_with_ltm(
                        past_latents=past_latents,
                        future_horizon=batch_future_target.shape[-1],
                        ltm_batch_size=ltm_batch_size,
                    )
                    
                print("past_latents std over time:", past_latents.std(dim=-1).mean().item())
                print("future_latents_pred std over time:", future_latents_pred.std(dim=-1).mean().item())

                #! mix还是hat?
                future_target_pred_norm = self.adapter.decode(future_latents_pred, batch_future_covariates_aug)
                mu_ref  = future_mu_t_mix.detach()
                std_ref = future_std_t_mix.detach()
                std_floor = 0.05
                std_ref = torch.clamp(std_ref, min=std_floor)
                # batch_future_target_norm = (batch_future_target - mu_ref) / std_ref
                # loss_future_pred = F.mse_loss(future_target_pred_norm, batch_future_target_norm)
                future_target_pred = revin_patch_denorm_target(future_target_pred_norm, mu_ref, std_ref)
                loss_future_pred = F.mse_loss(future_target_pred, batch_future_target)
                
                future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
                loss_stats_pred = F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5* F.mse_loss(future_logstd_p_hat, future_logstd_p_true)
                
                loss_latent_stats = self._latent_stats_loss(past_latents.detach(), future_latents_true)
                
                # 计算预测 y 的 patch 级统计
                P = P_fut
                B, _, H = future_target_pred.shape
                assert H % P == 0
                Np = H // P
                true4 = batch_future_target.view(B, 1, Np, P)
                pred4 = future_target_pred.view(B, 1, Np, P)

                std_true_p = true4.std(dim=-1)          # (B,1,Np)
                std_pred_p = pred4.std(dim=-1)

                loss_y_patch_std = F.mse_loss(std_pred_p, std_true_p)

                total_loss = (
                    lambda_past_recon * loss_past_recon
                    + lambda_future_pred * loss_future_pred
                    + lambda_latent_stats * loss_latent_stats
                    + lambda_stats_pred * loss_stats_pred
                    + lambda_y_patch_std * loss_y_patch_std
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                bsz = batch_past_target.size(0)
                total_count += bsz
                metric_sums["loss"] += total_loss.item() * bsz
                metric_sums["past_recon"] += loss_past_recon.item() * bsz
                metric_sums["future_pred"] += loss_future_pred.item() * bsz
                metric_sums["latent_stats"] += loss_latent_stats.item() * bsz
                metric_sums["stats_pred"] += loss_stats_pred.item() * bsz

                last_loss_past = loss_past_recon.item()
                last_loss_future = loss_future_pred.item()
                last_loss_stats = loss_latent_stats.item()
                last_loss_stats_pred = loss_stats_pred.item()
                
            with torch.no_grad():
                P = P_fut
                B, _, H = future_target_pred_norm.shape
                Np = H // P
                pred4 = future_target_pred_norm.view(B, 1, Np, P)
                std_pred_norm_patch = pred4.std(dim=-1)  # (B,1,Np)
                print("future_target_pred_norm patch std (mean over batch):",
                    std_pred_norm_patch.mean().item())

            train_metrics = {k: v / total_count for k, v in metric_sums.items()}
            
            val_metrics = None
            if val_data is not None:
                val_metrics = self.evaluate(
                    past_target=val_data["past_target"],
                    past_covariates=val_data["past_covariates"],
                    future_target=val_data["future_target"],
                    future_covariates=val_data["future_covariates"],
                    alpha_stats=alpha_stats, 
                    batch_size=batch_size,
                    ltm_batch_size=ltm_batch_size,
                    lambda_past_recon=lambda_past_recon,
                    lambda_future_pred=lambda_future_pred,
                    lambda_latent_stats=lambda_latent_stats,
                    lambda_stats_pred=lambda_stats_pred
                )
                
            current_value = val_metrics["loss"]
            improved = (best_monitor_value - current_value) > 1e-4
            if improved:
                best_monitor_value = current_value
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in self.adapter.state_dict().items()}
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
                    f"[adapter-train] epoch={epoch:03d} "
                    f"loss={train_metrics['loss']:.6f} "
                    f"past_recon={last_loss_past:.4f} "
                    f"future_pred={last_loss_future:.4f} "
                    f"latent_stats={last_loss_stats:.4f}"
                    f"stats_pred={last_loss_stats_pred:.4f}"
                )
                if val_metrics is not None:
                    msg += (
                        f" | val_loss={val_metrics['loss']:.6f} "
                        f"val_past={val_metrics['past_recon']:.4f} "
                        f"val_future={val_metrics['future_pred']:.4f} "
                        f"val_stats={val_metrics['latent_stats']:.4f}"
                    )
                print(msg)
                print(f"[earlystop] best={best_monitor_value:.6f} "
                      f"(epoch={best_epoch}) bad_count={bad_count}/{20}")
                
                if bad_count >= 20:
                    print(f"[earlystop] stop at epoch={epoch}, best_epoch={best_epoch}, best_loss={best_monitor_value:.6f}")
                    break
                
        self.adapter.load_state_dict(best_state_dict)
        if verbose:
            print(f"[earlystop] restored best adapter weights from epoch={best_epoch}")

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
        
        P_past = self.adapter.revin_patch_size_past
        P_fut  = self.adapter.revin_patch_size_future

        for sample_idx in range(n_samples):
            with torch.no_grad():
                
                past_target_norm, past_mu_t, past_std_t, _, _ = revin_patch_norm_target(
                    past_target_tensor, patch_size=P_past
                )
                # past_covariates_aug = torch.cat([past_covariates_tensor, past_mu_t, past_std_t], dim=1)
                past_covariates_aug = past_covariates_tensor

                future_mu_t_hat, future_std_t_hat, _, _, _ = self.adapter.predict_future_stats(future_covariates_tensor)
                # future_covariates_aug = torch.cat([future_covariates_tensor, future_mu_t_hat, future_std_t_hat], dim=1)
                future_covariates_aug = future_covariates_tensor
        
                # past_latents = self.adapter.encode(past_target_tensor, past_covariates_tensor)
                past_latents = self.adapter.encode(past_target_norm, past_covariates_aug)

                future_latents_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )


                future_target_pred_norm = self.adapter.decode(future_latents_pred, future_covariates_aug)
                # future_target_pred = self._revin_denorm_target(future_target_pred_norm, past_mu, past_std)
                future_target_pred = revin_patch_denorm_target(future_target_pred_norm, future_mu_t_hat, torch.clamp(future_std_t_hat,min=0.05))

                
                all_future_target_samples.append(future_target_pred.detach().cpu().numpy())
                
                if sample_idx == 0:
                    print("------------------------predict----------------------")
                    print("\n==== latent stats ====")
                    print("past_latents   min/mean/max:", past_latents.min().item(), past_latents.mean().item(), past_latents.max().item())
                    print("past_latents   std:", past_latents.std().item())
                    print("future_cov     min/mean/max:", future_covariates_tensor.min().item(), future_covariates_tensor.mean().item(), future_covariates_tensor.max().item())
                    print("future_latents min/mean/max:", future_latents_pred.min().item(), future_latents_pred.mean().item(), future_latents_pred.max().item())
                    print("future_latents std:", future_latents_pred.std().item())
                    print("future_target_pred_norm std:", future_target_pred_norm.std().item()) # 如果很小说明decoder在输出保守解
                    
                    mu_patch_hat, std_patch_hat, _ = self.adapter.future_stats_predictor(future_covariates_tensor)
                    print("mu_patch_hat min/mean/max:", mu_patch_hat.min().item(), mu_patch_hat.mean().item(), mu_patch_hat.max().item())
                    print("std_patch_hat min/mean/max:", std_patch_hat.min().item(), std_patch_hat.mean().item(), std_patch_hat.max().item())

        all_future_target_samples = np.stack(all_future_target_samples, axis=0)  # (S,B,1,H)
        mean = all_future_target_samples.mean(axis=0)
        std = all_future_target_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
