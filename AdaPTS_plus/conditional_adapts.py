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

import os
import matplotlib.pyplot as plt

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
    
    def _identity_norm(self, y: torch.Tensor):
        # y: (B,1,T)
        mu = torch.zeros_like(y)
        std = torch.ones_like(y)
        return y, mu, std

    def _identity_stats(self, y: torch.Tensor):
        mu = torch.zeros_like(y)
        std = torch.ones_like(y)
        return mu, std

    def _identity_denorm(self, y_norm: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        return y_norm

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
        # past_latents_ctx = past_latents.detach()
        # past_latents_ctx = past_latents_ctx.contiguous().clone()
        past_latents_ctx = past_latents
        
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
        use_patch_revin: bool = False,
    ) -> Dict[str, float]:

        self.adapter.eval()
        self.iclearner.eval()

        def _patch_stats_1d(x: torch.Tensor, patch: int):
            B, C, T = x.shape
            assert T % patch == 0
            Np = T // patch
            x4 = x.view(B, C, Np, patch)
            mean = x4.mean(dim=-1)
            std  = x4.std(dim=-1) + 1e-5
            return mean, std

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
            
            # batch_past_covariates_aug = self._augment_covariates_with_revin_stats(
            #     batch_past_covariates, past_mu, past_std
            # )
            # batch_future_covariates_aug = self._augment_covariates_with_revin_stats(
            #     batch_future_covariates, past_mu, past_std
            # )
            if use_patch_revin:
                batch_past_target_norm, past_mu_t, past_std_t, _, _ = revin_patch_norm_target(
                    batch_past_target, patch_size=P_past
                )
            else:
                batch_past_target_norm, past_mu_t, past_std_t = self._identity_norm(batch_past_target)
            # batch_past_covariates_aug = torch.cat([batch_past_covariates, past_mu_t.detach(), past_std_t.detach()], dim=1)  # (B,Cx+2,L)
            batch_past_covariates_aug = batch_past_covariates

            if use_patch_revin:
                future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = revin_patch_stats_target(
                    batch_future_target, patch_size=P_fut
                )
            else:
                future_mu_t_true, future_std_t_true = self._identity_stats(batch_future_target)
                future_mu_p_true, future_std_p_true = _patch_stats_1d(batch_future_target, P_fut)  # (B,1,Np)
            future_mu_t_hat, future_std_t_hat, future_mu_p_hat, future_std_p_hat, future_logstd_p_hat = self.adapter.predict_future_stats(
                batch_future_covariates
            )
            
            future_mu_t_mix = (1.0 - alpha_stats) * future_mu_t_true + alpha_stats * future_mu_t_hat
            future_logstd_t_true = torch.log(future_std_t_true + 1e-5)
            future_logstd_t_hat  = torch.log(future_std_t_hat + 1e-5)
            future_logstd_t_mix  = (1.0 - alpha_stats) * future_logstd_t_true + alpha_stats * future_logstd_t_hat
            future_std_t_mix = torch.exp(future_logstd_t_mix)
            
            # batch_future_covariates_aug = torch.cat([batch_future_covariates, future_mu_t_mix.detach(), future_std_t_mix.detach()], dim=1)
            batch_future_covariates_aug = batch_future_covariates

            # past_latents = self.adapter.encode(batch_past_target_norm, batch_past_covariates_aug)
            # future_latents_true = self.adapter.encode(batch_future_target_norm, batch_future_covariates_aug)
            past_latents = self.adapter.encode(batch_past_target_norm, batch_past_covariates_aug)
            if past_latents.size(1) != 1:
                raise ValueError(f"[evaluate] latent_dim must be 1 for proxy, got {past_latents.size(1)}")
            past_proxy = batch_past_target_norm - past_latents
            
            batch_future_target_norm_true = (batch_future_target - future_mu_t_true) / (future_std_t_true + 1e-5)
            # batch_future_covariates_aug_true = torch.cat([batch_future_covariates, future_mu_t_true.detach(), future_std_t_true.detach()], dim=1)
            batch_future_covariates_aug_true = batch_future_covariates
            future_latents_true = self.adapter.encode(batch_future_target_norm_true, batch_future_covariates_aug_true)
            if future_latents_true.size(1) != 1:
                raise ValueError(f"[evaluate] latent_dim must be 1 for proxy, got {future_latents_true.size(1)}")
            future_proxy_true = batch_future_target_norm_true - future_latents_true

            # past recon
            # past_target_recon_norm = self.adapter.decode(past_latents, batch_past_covariates_aug)
            # past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
            # loss_past_recon = F.mse_loss(past_target_recon_norm, batch_past_target_norm)
            
            past_resid_recon_norm = self.adapter.decode(past_proxy, batch_past_covariates_aug)   # residual_hat
            past_target_recon_norm = past_latents + past_resid_recon_norm                        # y_hat_norm

            if use_patch_revin:
                past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
                loss_past_recon = F.mse_loss(past_target_recon, batch_past_target)
            else:
                loss_past_recon = F.mse_loss(past_target_recon_norm, batch_past_target_norm)

            # future pred via LTM -> decoder
            future_proxy_pred = self._predict_future_latents_with_ltm(
                # past_latents=past_latents,
                past_latents=past_proxy,
                future_horizon=batch_future_target.shape[-1],
                ltm_batch_size=ltm_batch_size,
            )
            future_trend_pred = self._predict_future_latents_with_ltm(
                past_latents=past_latents,
                future_horizon=batch_future_target.shape[-1],
                ltm_batch_size=ltm_batch_size,
            )
        
        
            future_resid_hat_norm = self.adapter.decode(future_proxy_pred, batch_future_covariates_aug)
            future_target_pred_norm = future_trend_pred + future_resid_hat_norm
            
            std_floor = 0.05
            if use_patch_revin:
                mu_ref  = future_mu_t_mix.detach()
                std_ref = future_std_t_mix.detach()      
                std_ref = torch.clamp(std_ref, min=std_floor)
                future_target_pred = revin_patch_denorm_target(future_target_pred_norm, mu_ref, std_ref)
                loss_future_pred = F.mse_loss(future_target_pred, batch_future_target)
                target_norm_ref = (batch_future_target - mu_ref) / std_ref
            else:
                # no revin
                future_target_pred = future_target_pred_norm
                loss_future_pred = F.mse_loss(future_target_pred_norm, batch_future_target)
                target_norm_ref = batch_future_target
            
            future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
            loss_stats_pred =  F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5*F.mse_loss(future_logstd_p_hat, future_logstd_p_true)
            
            # print('evl: std pred vs true', future_std_p_hat[0], future_std_p_true[0])

            # weak latent stats alignment
            # loss_latent_stats = self._latent_stats_loss(past_latents, future_latents_true)
            loss_latent_stats_trend = self._latent_stats_loss(past_latents, future_latents_true)
            loss_latent_stats_resid  = self._latent_stats_loss(past_proxy, future_proxy_true)
            loss_latent_stats = 0.5 * loss_latent_stats_trend + 0.5 * loss_latent_stats_resid

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
    
    def pretrain_past_reconstruction_only(
        self,
        past_target: np.ndarray,        # (N,1,L) scaled
        past_covariates: np.ndarray,    # (N,Cx,L)
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # debug
        debug: bool = True,
        debug_dir: str = "./debug_no_revin_residual",
        debug_plot: bool = True,
        debug_num_seq: int = 2,
        verbose: bool = True,
    ):
        """
        latents = encoder(y_past, x_past)      # (B,1,L)
        proxy   = y_past - latents             # (B,1,L)  residual, 后续送 LTM 的代理变量
        y_recon = decoder(proxy, x_past)       # (B,1,L)
        loss = MSE(y_recon, y_past)
        """

        os.makedirs(debug_dir, exist_ok=True)

        def _to_np(x: torch.Tensor):
            return x.detach().float().cpu().numpy()

        def _print_stats(name: str, x: torch.Tensor):
            x_ = x.detach()
            print(f"[dbg] {name:28s} shape={tuple(x_.shape)} "
                f"min={x_.min().item():.4g} mean={x_.mean().item():.4g} max={x_.max().item():.4g} std={x_.std().item():.4g}")

        def _patch_stats(x: torch.Tensor, patch: int):
            # x: (B,1,T)
            B, C, T = x.shape
            assert C == 1
            if patch <= 1 or (T % patch != 0):
                mu = x.mean(dim=-1, keepdim=True)  # (B,1,1)
                sd = x.std(dim=-1, keepdim=True)
                return mu, sd
            Np = T // patch
            x4 = x.view(B, 1, Np, patch)
            mu = x4.mean(dim=-1)  # (B,1,Np)
            sd = x4.std(dim=-1)   # (B,1,Np)
            return mu, sd

        def _print_patch(name: str, x: torch.Tensor, patch: int, b: int = 0, k: int = 6):
            mu, sd = _patch_stats(x, patch)
            mu_np = mu[b, 0, :min(k, mu.shape[-1])].detach().cpu().numpy()
            sd_np = sd[b, 0, :min(k, sd.shape[-1])].detach().cpu().numpy()
            print(f"[dbg] {name:28s} patch_mean[:{k}]={mu_np}")
            print(f"[dbg] {name:28s} patch_std [: {k}]={sd_np}")
            print(f"[dbg] {name:28s} patch_std summary min/mean/max="
                f"{sd.min().item():.4g}/{sd.mean().item():.4g}/{sd.max().item():.4g}")

        def _plot_1d(save_path: str, curves: dict, title: str):
            plt.figure(figsize=(14, 4))
            for k, v in curves.items():
                plt.plot(v, label=k)
            plt.legend()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

        dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.adapter.train()
        optim = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        P_dbg = getattr(self.adapter, "revin_patch_size_past", 24)

        for epoch in range(n_epochs):
            total = 0.0
            count = 0
            dumped = False

            for (y_p, x_p) in loader:
                y_p = y_p.to(self.device)  # (B,1,L)
                x_p = x_p.to(self.device)  # (B,Cx,L)

                latents = self.adapter.encode(y_p, x_p)  # (B,1,L) if latent_dim=1
                if latents.size(1) != 1:
                    raise ValueError(f"latent_dim must be 1, got {latents.size(1)}")

                proxy = y_p - latents
                y_recon = self.adapter.decode(proxy, x_p)

                loss = F.mse_loss(y_recon, y_p)

                optim.zero_grad()
                loss.backward()
                optim.step()

                total += loss.item() * y_p.size(0)
                count += y_p.size(0)

                # sanity + debug（每个 epoch 第一个 batch）
                if debug and (not dumped):
                    dumped = True
                    print("\n=========== [Pretrain past recon ONLY] DEBUG ===========")
                    print(f"[dbg] epoch={epoch}/{n_epochs-1} batch={y_p.size(0)}")

                    _print_stats("y_past", y_p)
                    _print_stats("latents(trend)", latents)
                    _print_stats("proxy(residual)", proxy)
                    _print_stats("y_recon", y_recon)

                    _print_patch("y_past", y_p, P_dbg, b=0)
                    _print_patch("latents(trend)", latents, P_dbg, b=0)
                    _print_patch("proxy(residual)", proxy, P_dbg, b=0)
                    _print_patch("y_recon", y_recon, P_dbg, b=0)

                    with torch.no_grad():
                        perm = torch.randperm(y_p.size(0), device=self.device)
                        y_recon_shuf_cov = self.adapter.decode(proxy, x_p[perm])
                        cov_sens = (y_recon - y_recon_shuf_cov).abs().mean().item()

                        y_recon_shuf_proxy = self.adapter.decode(proxy[perm], x_p)
                        proxy_sens = (y_recon - y_recon_shuf_proxy).abs().mean().item()

                        print(f"[probe] decoder sensitivity cov   : {cov_sens:.4f}")
                        print(f"[probe] decoder sensitivity proxy : {proxy_sens:.4f}")

                    if debug_plot:
                        out_dir = os.path.join(debug_dir, "pretrain_stage1")
                        os.makedirs(out_dir, exist_ok=True)
                        i = 0
                        _plot_1d(
                            os.path.join(out_dir, f"ep{epoch:03d}_past_recon_b{i}.png"),
                            {"y_past": _to_np(y_p[i, 0]), "y_recon": _to_np(y_recon[i, 0])},
                            title=f"Stage1 past recon | ep={epoch} b={i}",
                        )
                        _plot_1d(
                            os.path.join(out_dir, f"ep{epoch:03d}_trend_proxy_b{i}.png"),
                            {"latents(trend)": _to_np(latents[i, 0]), "proxy(residual)": _to_np(proxy[i, 0])},
                            title=f"Stage1 trend & proxy | ep={epoch} b={i}",
                        )
                    print("=========================================================\n")

            if verbose:
                print(f"[stage1-pretrain] epoch={epoch:03d} loss_past_recon={total/max(1,count):.6f}")
    
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
        use_patch_revin: bool = False,
        use_swanlab: bool = True,
        swanlab_run = None,
        debug: bool = True,
        debug_dir: str = "./debug_train_adapter",
        debug_every: int = 2,      # 每隔多少个 epoch dump 一次
        debug_num_seq: int = 2,    # 每次画多少条样本
        debug_latent_ch: int = 1,  # 每次画几个 latent channel
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
        
        os.makedirs(debug_dir, exist_ok=True)

        def _to_np(x: torch.Tensor):
            return x.detach().float().cpu().numpy()

        def _patch_stats_1d(x: torch.Tensor, patch: int):
            """
            x: (B,1,T) or (B,C,T)
            return mean/std over last dim within patch:
            mean: (B,C,Np), std: (B,C,Np)
            """
            B, C, T = x.shape
            assert T % patch == 0
            Np = T // patch
            x4 = x.view(B, C, Np, patch)
            mean = x4.mean(dim=-1)
            std  = x4.std(dim=-1)
            return mean, std

        def _print_tensor_stats(name: str, x: torch.Tensor):
            x_ = x.detach()
            print(f"[dbg] {name:24s} shape={tuple(x_.shape)} "
                f"min={x_.min().item():.4g} mean={x_.mean().item():.4g} max={x_.max().item():.4g} "
                f"std(all)={x_.std().item():.4g}")

        def _print_patch_stats(name: str, x: torch.Tensor, patch: int, b: int = 0, c: int = 0, k: int = 6):
            """
            打印第 b 个样本、第 c 个通道前 k 个 patch 的 mean/std
            """
            mean, std = _patch_stats_1d(x, patch)
            m = mean[b, c, :k].detach().cpu().numpy()
            s = std[b, c, :k].detach().cpu().numpy()
            print(f"[dbg] {name:24s} patch_mean[{b},{c},:]{m}")
            print(f"[dbg] {name:24s} patch_std [{b},{c},:]{s}")
            print(f"[dbg] {name:24s} patch_std summary min/mean/max="
                f"{std.min().item():.4g}/{std.mean().item():.4g}/{std.max().item():.4g}")

        def _collapse_index(x_norm: torch.Tensor, patch: int):
            """
            一个简单的“坍塌指数”：
            - 取 patch 内 std 的均值 / (全局 std + eps)
            如果输出几乎常数，patch 内 std 会很小，指数趋近 0。
            """
            eps = 1e-6
            _, std_p = _patch_stats_1d(x_norm, patch)
            return (std_p.mean() / (x_norm.std() + eps)).item()

        def _plot_series_grid(save_path: str, series_dict: dict, title: str, max_len: int = None):
            """
            series_dict: {name: 1D np.array}
            """
            plt.figure(figsize=(14, 4))
            for k, v in series_dict.items():
                if max_len is not None:
                    v = v[:max_len]
                plt.plot(v, label=k)
            plt.legend()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

        def _plot_patch_curves(save_path: str, patch_mean: np.ndarray, patch_std: np.ndarray, title: str):
            """
            patch_mean/std: (Np,) for one sample/channel
            """
            x = np.arange(len(patch_mean))
            plt.figure(figsize=(14, 4))
            plt.plot(x, patch_mean, label="patch_mean")
            plt.plot(x, patch_std, label="patch_std")
            plt.legend()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

        # LTM frozen
        self.iclearner.eval()
        for p in self.iclearner.backbone.parameters():
            p.requires_grad_(False)
        
        best_monitor_value = float("inf")
        best_epoch = -1
        best_state_dict = None
        bad_count = 0

        for epoch in range(n_epochs):
            did_dump_this_epoch = False
            
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
                
                if use_patch_revin:
                    batch_past_target_norm, past_mu_t, past_std_t, _, _ = revin_patch_norm_target(
                        batch_past_target, patch_size=P_past
                    )
                else:
                    batch_past_target_norm, past_mu_t, past_std_t = self._identity_norm(batch_past_target)
                # batch_past_covariates_aug = torch.cat([batch_past_covariates, past_mu_t.detach(), past_std_t.detach()], dim=1)  # (B,Cx+2,L)
                batch_past_covariates_aug = batch_past_covariates
                if use_patch_revin:
                    future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = revin_patch_stats_target(
                        batch_future_target, patch_size=P_fut
                    )
                else:
                    future_mu_t_true, future_std_t_true = self._identity_stats(batch_future_target)
                    future_mu_p_true, future_std_p_true = _patch_stats_1d(batch_future_target, P_fut)  # (B,1,Np)
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
                if past_latents.size(1) != 1:
                    raise ValueError(f"[train_adapter] latent_dim must be 1 for residual proxy, got {past_latents.size(1)}")
                past_proxy = batch_past_target_norm - past_latents  # (B,1,L)  residual proxy

                batch_future_target_norm_true = (batch_future_target - future_mu_t_true) / (future_std_t_true + 1e-5)
                # batch_future_covariates_aug_true = torch.cat([batch_future_covariates, future_mu_t_true.detach(), future_std_t_true.detach()], dim=1)
                batch_future_covariates_aug_true = batch_future_covariates
                future_latents_true = self.adapter.encode(batch_future_target_norm_true, batch_future_covariates_aug_true)
                if future_latents_true.size(1) != 1:
                    raise ValueError(f"[train_adapter] latent_dim must be 1 for residual proxy, got {future_latents_true.size(1)}")
                future_proxy_true = batch_future_target_norm_true - future_latents_true  # (B,1,H)

                # past reconstruction
                # past_target_recon_norm = self.adapter.decode(past_latents, batch_past_covariates_aug)
                past_resid_recon_norm = self.adapter.decode(past_proxy, batch_past_covariates_aug)  # residual_hat
                past_target_recon_norm = past_latents + past_resid_recon_norm                        # y_hat_norm

                if use_patch_revin:
                    past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
                    loss_past_recon = F.mse_loss(past_target_recon, batch_past_target)
                else:
                    loss_past_recon = F.mse_loss(past_target_recon_norm, batch_past_target)

                # LTM consistency
                future_proxy_pred = self._predict_future_latents_with_ltm(
                    # past_latents=past_latents,
                    past_latents=past_proxy,
                    future_horizon=batch_future_target.shape[-1],
                    ltm_batch_size=ltm_batch_size,
                )
                future_trend_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=batch_future_target.shape[-1],
                    ltm_batch_size=ltm_batch_size,
                )
                future_latents_pred = future_proxy_pred

                
                future_resid_hat_norm = self.adapter.decode(future_proxy_pred, batch_future_covariates_aug)
                future_target_pred_norm = future_trend_pred + future_resid_hat_norm
                
                std_floor = 0.05
                if use_patch_revin:
                    mu_ref  = future_mu_t_mix.detach()
                    std_ref = future_std_t_mix.detach()      
                    std_ref = torch.clamp(std_ref, min=std_floor)
                    future_target_pred = revin_patch_denorm_target(future_target_pred_norm, mu_ref, std_ref)
                    loss_future_pred = F.mse_loss(future_target_pred, batch_future_target)
                    target_norm_ref = (batch_future_target - mu_ref) / std_ref
                else:
                    # no revin
                    loss_future_pred = F.mse_loss(future_target_pred_norm, batch_future_target)
                    future_target_pred = future_target_pred_norm
                    target_norm_ref = batch_future_target
                    mu_ref = torch.zeros_like(batch_future_target)
                    std_ref = torch.ones_like(batch_future_target)
                
                future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
                loss_stats_pred = F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5* F.mse_loss(future_logstd_p_hat, future_logstd_p_true)
                
                # loss_latent_stats = self._latent_stats_loss(past_latents.detach(), future_latents_true)
                loss_latent_stats_trend = self._latent_stats_loss(past_latents.detach(), future_latents_true)
                loss_latent_stats_resid  = self._latent_stats_loss(past_proxy.detach(), future_proxy_true)
                loss_latent_stats = 0.5 * loss_latent_stats_trend + 0.5 * loss_latent_stats_resid
                
                # 计算预测 y 的 patch 级统计
                P = P_fut
                B, _, H = future_target_pred.shape
                assert H % P == 0
                Np = H // P
                pred4_norm = future_target_pred_norm.view(B, 1, Np, P)
                true4_norm = target_norm_ref.view(B, 1, Np, P)

                m_pred = pred4_norm.mean(dim=-1)          # (B,1,Np)
                m_true = true4_norm.mean(dim=-1)
                ms_pred = (pred4_norm ** 2).mean(dim=-1)  # (B,1,Np)
                ms_true = (true4_norm ** 2).mean(dim=-1)
                loss_y_patch_moment = F.mse_loss(ms_pred, ms_true) + 0.5 * F.mse_loss(m_pred, m_true)

                total_loss = (
                    lambda_past_recon * loss_past_recon
                    + lambda_future_pred * loss_future_pred
                    + lambda_latent_stats * loss_latent_stats
                    + lambda_stats_pred * loss_stats_pred
                    # + lambda_y_patch_std * loss_y_patch_std
                    + lambda_y_patch_std * loss_y_patch_moment
                )

                optimizer.zero_grad()
                total_loss.backward()
                
                if debug and (not did_dump_this_epoch) and (epoch % debug_every == 0):
                    gnorm = 0.0
                    for n, p in self.adapter.named_parameters():
                        if p.grad is None:
                            continue
                        if "target_decoder" in n:   # 或者你的 decoder 名称
                            gnorm += p.grad.detach().data.norm(2).item() ** 2
                    gnorm = gnorm ** 0.5
                    print(f"[dbg] grad_norm(target_decoder)={gnorm:.4g}")
                
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
                
            # ====== debug dump (only once per epoch) ======
            if debug and (not did_dump_this_epoch) and (epoch % debug_every == 0):
                did_dump_this_epoch = True
                with torch.no_grad():
                    B0 = batch_past_target.size(0)
                    nseq = min(debug_num_seq, B0)
                    nch  = min(debug_latent_ch, past_latents.size(1))

                    print("\n================ DEBUG DUMP ================")
                    print(f"[dbg] epoch={epoch} alpha_stats={alpha_stats:.3f} B={B0} "
                        f"P_past={P_past} P_fut={P_fut}")

                    _print_tensor_stats("past_latents(trend)", past_latents)
                    _print_tensor_stats("past_proxy(residual)", past_proxy)
                    _print_tensor_stats("future_trend_pred", future_trend_pred)
                    _print_tensor_stats("future_proxy_pred", future_proxy_pred)

                    _print_tensor_stats("past_target_norm", batch_past_target_norm)
                    _print_tensor_stats("past_resid_hat_norm", past_resid_recon_norm)
                    _print_tensor_stats("past_recon_norm(y)", past_target_recon_norm)

                    _print_tensor_stats("future_resid_hat_norm", future_resid_hat_norm)
                    _print_tensor_stats("future_pred_norm(y)", future_target_pred_norm)
                    _print_tensor_stats("target_norm_ref", target_norm_ref)

                    _print_patch_stats("past_target_norm", batch_past_target_norm, P_past, b=0, c=0)
                    _print_patch_stats("past_recon_norm(y)", past_target_recon_norm, P_past, b=0, c=0)
                    _print_patch_stats("future_pred_norm(y)", future_target_pred_norm, P_fut, b=0, c=0)
                    _print_patch_stats("target_norm_ref", target_norm_ref, P_fut, b=0, c=0)

                    ci_past_recon = _collapse_index(past_target_recon_norm, P_past)
                    ci_future_pred = _collapse_index(future_target_pred_norm, P_fut)
                    print(f"[dbg] collapse_index past_recon={ci_past_recon:.4f} future_pred={ci_future_pred:.4f} "
                        f"(smaller => flatter)")

                    out_dir = os.path.join(debug_dir, f"epoch_{epoch:03d}")
                    os.makedirs(out_dir, exist_ok=True)

                    for i in range(nseq):
                        y_p = _to_np(batch_past_target_norm[i, 0])
                        y_pr = _to_np(past_target_recon_norm[i, 0])
                        _plot_series_grid(
                            save_path=os.path.join(out_dir, f"past_recon_norm_b{i}.png"),
                            series_dict={"past_target_norm": y_p, "past_recon_norm": y_pr},
                            title=f"past recon (trend+resid) | epoch={epoch} b={i}"
                        )

                        y_f_pred = _to_np(future_target_pred_norm[i, 0])
                        y_f_true = _to_np(target_norm_ref[i, 0])
                        _plot_series_grid(
                            save_path=os.path.join(out_dir, f"future_pred_norm_b{i}.png"),
                            series_dict={"future_pred_norm": y_f_pred, "target_norm_ref": y_f_true},
                            title=f"future pred (trend+resid) | epoch={epoch} b={i}"
                        )

                        pm, ps = _patch_stats_1d(future_target_pred_norm[i:i+1], P_fut)
                        tm, ts = _patch_stats_1d(target_norm_ref[i:i+1], P_fut)

                        _plot_patch_curves(
                            save_path=os.path.join(out_dir, f"future_pred_patchstats_b{i}.png"),
                            patch_mean=_to_np(pm[0, 0]),
                            patch_std=_to_np(ps[0, 0]),
                            title=f"future_pred patch mean/std | epoch={epoch} b={i}"
                        )
                        _plot_patch_curves(
                            save_path=os.path.join(out_dir, f"target_ref_patchstats_b{i}.png"),
                            patch_mean=_to_np(tm[0, 0]),
                            patch_std=_to_np(ts[0, 0]),
                            title=f"target_ref patch mean/std | epoch={epoch} b={i}"
                        )

                    # latent 折线图：trend 和 residual proxy 都画（尽量不破坏你的原逻辑）
                    for i in range(nseq):
                        for c in range(nch):
                            z_tr = _to_np(past_latents[i, c])
                            z_tr_f = _to_np(future_trend_pred[i, c])
                            _plot_series_grid(
                                save_path=os.path.join(out_dir, f"trend_ch{c}_b{i}.png"),
                                series_dict={"past_trend": z_tr, "future_trend_pred": z_tr_f},
                                title=f"trend | epoch={epoch} b={i} ch={c}"
                            )

                            z_res = _to_np(past_proxy[i, c])
                            z_res_f = _to_np(future_proxy_pred[i, c])
                            _plot_series_grid(
                                save_path=os.path.join(out_dir, f"residual_ch{c}_b{i}.png"),
                                series_dict={"past_residual": z_res, "future_residual_pred": z_res_f},
                                title=f"residual proxy | epoch={epoch} b={i} ch={c}"
                            )

                    print(f"[dbg] saved debug plots to: {out_dir}")
                    print("================================================\n")

            with torch.no_grad():
                P = P_fut
                B, _, H = future_target_pred_norm.shape
                Np = H // P

                pred4_norm = future_target_pred_norm.view(B, 1, Np, P)
                std_pred_norm_patch = pred4_norm.std(dim=-1).mean().item()

                target_norm_ref2 = (batch_future_target - mu_ref) / std_ref
                true4_norm = target_norm_ref2.view(B, 1, Np, P)
                std_true_norm_patch = true4_norm.std(dim=-1).mean().item()

                std_ref_patch = std_ref.view(B, 1, Np, P).mean(dim=-1).mean().item()

                print("[debug] patch std (norm) pred:", std_pred_norm_patch,
                    "| true:", std_true_norm_patch,
                    "| std_ref_patch_mean:", std_ref_patch)

                pred4_raw = future_target_pred.view(B, 1, Np, P)
                true4_raw = batch_future_target.view(B, 1, Np, P)
                print("[debug] patch std (raw) pred:", pred4_raw.std(dim=-1).mean().item(),
                    "| true:", true4_raw.std(dim=-1).mean().item())

                # probe：decoder 对 cov 的敏感度（现在 decoder 输出 residual_hat）
                z = future_proxy_true
                y1 = self.adapter.decode(z, batch_future_covariates_aug_true)
                y2 = self.adapter.decode(z, batch_future_covariates_aug_true[torch.randperm(z.size(0))])
                diff = (y1 - y2).abs().mean().item()
                print("[probe] decoder sensitivity to cov:", diff)

                z1 = future_proxy_true
                z2 = future_proxy_true[torch.randperm(z1.size(0))]
                y1 = self.adapter.decode(z1, batch_future_covariates_aug_true)
                y2 = self.adapter.decode(z2, batch_future_covariates_aug_true)
                diffz = (y1 - y2).abs().mean().item()
                print("[probe] decoder sensitivity to latents(proxy):", diffz)

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
                    lambda_stats_pred=lambda_stats_pred,
                    use_patch_revin=use_patch_revin,
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
        use_patch_revin: bool = False,
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
                
                if use_patch_revin:
                    past_target_norm, past_mu_t, past_std_t, _, _ = revin_patch_norm_target(
                        past_target_tensor, patch_size=P_past
                    )
                else:
                    past_target_norm, past_mu_t, past_std_t = self._identity_norm(past_target_tensor)
                # past_covariates_aug = torch.cat([past_covariates_tensor, past_mu_t, past_std_t], dim=1)
                past_covariates_aug = past_covariates_tensor

                future_mu_t_hat, future_std_t_hat, _, _, _ = self.adapter.predict_future_stats(future_covariates_tensor)
                # future_covariates_aug = torch.cat([future_covariates_tensor, future_mu_t_hat, future_std_t_hat], dim=1)
                future_covariates_aug = future_covariates_tensor

                past_latents = self.adapter.encode(past_target_norm, past_covariates_aug)
                if past_latents.size(1) != 1:
                    raise ValueError(f"[predict] latent_dim must be 1 for residual trend, got {past_latents.size(1)}")
                past_proxy = past_target_norm - past_latents

                future_trend_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )
                future_proxy_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_proxy,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )

                future_resid_hat_norm = self.adapter.decode(future_proxy_pred, future_covariates_aug)
                future_target_pred_norm = future_trend_pred + future_resid_hat_norm
                # future_target_pred = self._revin_denorm_target(future_target_pred_norm, past_mu, past_std)
                if use_patch_revin:
                    future_target_pred = revin_patch_denorm_target(future_target_pred_norm, future_mu_t_hat, torch.clamp(future_std_t_hat,min=0.05))
                else:
                    future_target_pred = future_target_pred_norm

                
                all_future_target_samples.append(future_target_pred.detach().cpu().numpy())
                
                if sample_idx == 0:
                    print("------------------------predict----------------------")
                    print("\n==== latent stats ====")
                    print("past_latents(trend)   min/mean/max:", past_latents.min().item(), past_latents.mean().item(), past_latents.max().item())
                    print("past_latents(trend)   std:", past_latents.std().item())
                    print("past_proxy(residual)  std:", past_proxy.std().item())

                    print("future_cov     min/mean/max:", future_covariates_tensor.min().item(), future_covariates_tensor.mean().item(), future_covariates_tensor.max().item())

                    print("future_trend_pred std:", future_trend_pred.std().item())
                    print("future_proxy_pred std:", future_proxy_pred.std().item())

                    print("future_resid_hat_norm std:", future_resid_hat_norm.std().item())
                    print("future_target_pred_norm std:", future_target_pred_norm.std().item())

                    mu_patch_hat, std_patch_hat, _ = self.adapter.future_stats_predictor(future_covariates_tensor)
                    print("mu_patch_hat min/mean/max:", mu_patch_hat.min().item(), mu_patch_hat.mean().item(), mu_patch_hat.max().item())
                    print("std_patch_hat min/mean/max:", std_patch_hat.min().item(), std_patch_hat.mean().item(), std_patch_hat.max().item())

        all_future_target_samples = np.stack(all_future_target_samples, axis=0)  # (S,B,1,H)
        mean = all_future_target_samples.mean(axis=0)
        std = all_future_target_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
