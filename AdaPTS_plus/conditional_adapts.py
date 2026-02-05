# conditional_adapts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

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

logger = None

@dataclass
class PredPack:
    mean: np.ndarray   # (B,1,H)
    lb:   np.ndarray
    ub:   np.ndarray
    
@dataclass
class EpochMeters:
    sums: Dict[str, float]
    count: int = 0

    @staticmethod
    def new(keys: List[str]) -> "EpochMeters":
        return EpochMeters(sums={k: 0.0 for k in keys}, count=0)

    def update(self, values: Dict[str, torch.Tensor], bsz: int):
        for k, v in values.items():
            self.sums[k] += float(v.detach().item()) * bsz
        self.count += bsz

    def mean(self) -> Dict[str, float]:
        den = max(1, self.count)
        return {k: v / den for k, v in self.sums.items()}
    
    
class Callback:
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        pass
    
class ConsoleLogger(Callback):
    """
    每个 epoch 打印一次
    """
    def __init__(self, print_earlystop: bool = True):
        self.print_earlystop = print_earlystop

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        mode = state["mode"]           # "train" or "val"
        epoch = state["epoch"]
        metrics = state["metrics"]
        alpha_stats = state.get("alpha_stats", None)

        if mode == "train":
            msg = (
                f"[adapter-train] epoch={epoch:03d} "
                f"alpha_stats={alpha_stats:.3f} "
                f"loss={metrics.get('loss', 0.0):.6f} "
                f"past_recon={metrics.get('past_recon', 0.0):.4f} "
                f"future_pred={metrics.get('future_pred', 0.0):.4f} "
                f"latent_stats={metrics.get('latent_stats', 0.0):.4f} "
                f"stats_pred={metrics.get('stats_pred', 0.0):.4f} "
                f"y_patch_moment={metrics.get('y_patch_moment', 0.0):.4f}"
            )
            logger.info(msg)

        elif mode == "val":
            msg = (
                f"[adapter-val  ] epoch={epoch:03d} "
                f"loss={metrics.get('loss', 0.0):.6f} "
                f"past_recon={metrics.get('past_recon', 0.0):.4f} "
                f"future_pred={metrics.get('future_pred', 0.0):.4f} "
                f"latent_stats={metrics.get('latent_stats', 0.0):.4f} "
                f"stats_pred={metrics.get('stats_pred', 0.0):.4f} "
                f"y_patch_moment={metrics.get('y_patch_moment', 0.0):.4f}"
            )
            logger.info(msg)

class PlotDumper(Callback):
    """
    每 N epoch 保存 debug 图：
    - past recon 曲线
    - future pred vs target_ref 曲线
    - future pred patch mean/std & target_ref patch mean/std
    - trend latent 曲线（past & future_pred）
    """
    def __init__(
        self,
        out_dir: str = "./debug_train_adapter",
        every: int = 2,
        num_seq: int = 2,
        latent_ch: int = 1,
        enabled: bool = True,
    ):
        self.out_dir = out_dir
        self.every = every
        self.num_seq = num_seq
        self.latent_ch = latent_ch
        self.enabled = enabled
        os.makedirs(self.out_dir, exist_ok=True)

        # 只取“该 epoch 第一次遇到 batch_end 的 debug_state”来画图/打印
        self._epoch_debug_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    @staticmethod
    def _to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    @staticmethod
    def _plot_series(save_path: str, series_dict: Dict[str, np.ndarray], title: str, max_len: Optional[int] = None):
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

    @staticmethod
    def _patch_stats_1d(x: torch.Tensor, patch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape
        assert T % patch == 0
        Np = T // patch
        x4 = x.view(B, C, Np, patch)
        return x4.mean(dim=-1), x4.std(dim=-1)

    @staticmethod
    def _plot_patch_curves(save_path: str, patch_mean: np.ndarray, patch_std: np.ndarray, title: str):
        x = np.arange(len(patch_mean))
        plt.figure(figsize=(14, 4))
        plt.plot(x, patch_mean, label="patch_mean")
        plt.plot(x, patch_std, label="patch_std")
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def _collapse_index(x_norm: torch.Tensor, patch: int) -> float:
        eps = 1e-6
        _, std_p = PlotDumper._patch_stats_1d(x_norm, patch)
        return float((std_p.mean() / (x_norm.std() + eps)).detach().cpu().item())

    @staticmethod
    def _print_tensor_stats(name: str, x: torch.Tensor):
        x_ = x.detach()
        logger.info(f"[dbg] {name:24s} shape={tuple(x_.shape)} "
              f"min={x_.min().item():.4g} mean={x_.mean().item():.4g} max={x_.max().item():.4g} "
              f"std(all)={x_.std().item():.4g}")

    @staticmethod
    def _print_patch_stats(name: str, x: torch.Tensor, patch: int, b: int = 0, c: int = 0, k: int = 6):
        mean, std = PlotDumper._patch_stats_1d(x, patch)
        m = mean[b, c, :k].detach().cpu().numpy()
        s = std[b, c, :k].detach().cpu().numpy()
        logger.info(f"[dbg] {name:24s} patch_mean[{b},{c},:]{m}")
        logger.info(f"[dbg] {name:24s} patch_std [{b},{c},:]{s}")
        logger.info(f"[dbg] {name:24s} patch_std summary min/mean/max="
              f"{std.min().item():.4g}/{std.mean().item():.4g}/{std.max().item():.4g}")

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        mode = state["mode"]      # train / val
        epoch = state["epoch"]

        if mode != "train":
            return
        if (epoch % self.every) != 0:
            return

        key = (mode, epoch)
        if key in self._epoch_debug_cache:
            return

        # 只缓存一次：该 epoch 第一批
        debug = state.get("debug", None)
        if debug is None:
            return

        self._epoch_debug_cache[key] = debug

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        mode = state["mode"]
        epoch = state["epoch"]
        if mode != "train":
            return
        if (epoch % self.every) != 0:
            return

        key = (mode, epoch)
        if key not in self._epoch_debug_cache:
            return
        dbg = self._epoch_debug_cache.pop(key)

        # unpack
        P_past = dbg["P_past"]
        P_fut = dbg["P_fut"]
        alpha_stats = dbg["alpha_stats"]

        past_latents = dbg["past_latents"]
        future_latents_pred = dbg["future_latents_pred"]

        past_target_norm = dbg["past_target_norm"]
        past_recon_norm = dbg["past_target_recon_norm"]

        future_pred_norm = dbg["future_target_pred_norm"]
        target_norm_ref = dbg["target_norm_ref"]

        future_target_pred = dbg["future_target_pred"]
        future_target_true = dbg["future_target_true"]

        future_latents_true = dbg["future_latents_true"]
        future_cov_aug_true = dbg["future_covariates_aug_true"]

        mu_ref = dbg["mu_ref"]
        std_ref = dbg["std_ref"]

        out_dir = os.path.join(self.out_dir, f"epoch_{epoch:03d}")
        os.makedirs(out_dir, exist_ok=True)

        B0 = past_target_norm.size(0)
        nseq = min(self.num_seq, B0)
        nch = min(self.latent_ch, past_latents.size(1))

        logger.info("\n================ DEBUG DUMP ================")
        logger.info(f"[dbg] epoch={epoch} alpha_stats={alpha_stats:.3f} B={B0} "
              f"P_past={P_past} P_fut={P_fut}")

        self._print_tensor_stats("past_latents(trend)", past_latents)
        self._print_tensor_stats("future_latents_pred", future_latents_pred)

        self._print_tensor_stats("past_target_norm", past_target_norm)
        self._print_tensor_stats("past_recon_norm(y)", past_recon_norm)

        self._print_tensor_stats("future_pred_norm(y)", future_pred_norm)
        self._print_tensor_stats("target_norm_ref", target_norm_ref)

        self._print_patch_stats("past_target_norm", past_target_norm, P_past, b=0, c=0)
        self._print_patch_stats("past_recon_norm(y)", past_recon_norm, P_past, b=0, c=0)
        self._print_patch_stats("future_pred_norm(y)", future_pred_norm, P_fut, b=0, c=0)
        self._print_patch_stats("target_norm_ref", target_norm_ref, P_fut, b=0, c=0)

        ci_past_recon = self._collapse_index(past_recon_norm, P_past)
        ci_future_pred = self._collapse_index(future_pred_norm, P_fut)
        logger.info(f"[dbg] collapse_index past_recon={ci_past_recon:.4f} future_pred={ci_future_pred:.4f} "
              f"(smaller => flatter)")

        # save plots
        for i in range(nseq):
            y_p = self._to_np(past_target_norm[i, 0])
            y_pr = self._to_np(past_recon_norm[i, 0])
            self._plot_series(
                save_path=os.path.join(out_dir, f"past_recon_norm_b{i}.png"),
                series_dict={"past_target_norm": y_p, "past_recon_norm": y_pr},
                title=f"past recon | epoch={epoch} b={i}",
            )

            y_f_pred = self._to_np(future_pred_norm[i, 0])
            y_f_true = self._to_np(target_norm_ref[i, 0])
            self._plot_series(
                save_path=os.path.join(out_dir, f"future_pred_norm_b{i}.png"),
                series_dict={"future_pred_norm": y_f_pred, "target_norm_ref": y_f_true},
                title=f"future pred norm | epoch={epoch} b={i}",
            )

            pm, ps = self._patch_stats_1d(future_pred_norm[i:i+1], P_fut)
            tm, ts = self._patch_stats_1d(target_norm_ref[i:i+1], P_fut)

            self._plot_patch_curves(
                save_path=os.path.join(out_dir, f"future_pred_patchstats_b{i}.png"),
                patch_mean=self._to_np(pm[0, 0]),
                patch_std=self._to_np(ps[0, 0]),
                title=f"future_pred patch mean/std | epoch={epoch} b={i}",
            )
            self._plot_patch_curves(
                save_path=os.path.join(out_dir, f"target_ref_patchstats_b{i}.png"),
                patch_mean=self._to_np(tm[0, 0]),
                patch_std=self._to_np(ts[0, 0]),
                title=f"target_ref patch mean/std | epoch={epoch} b={i}",
            )

        for i in range(nseq):
            for c in range(nch):
                z_tr = self._to_np(past_latents[i, c])
                z_tr_f = self._to_np(future_latents_pred[i, c])
                self._plot_series(
                    save_path=os.path.join(out_dir, f"trend_ch{c}_b{i}.png"),
                    series_dict={"past_trend": z_tr, "future_trend_pred": z_tr_f},
                    title=f"trend | epoch={epoch} b={i} ch={c}",
                )

        logger.info(f"[dbg] saved debug plots to: {out_dir}")

        with torch.no_grad():
            P = P_fut
            B, _, H = future_pred_norm.shape
            Np = H // P

            pred4_norm = future_pred_norm.view(B, 1, Np, P)
            std_pred_norm_patch = pred4_norm.std(dim=-1).mean().item()

            target_norm_ref2 = (future_target_true - mu_ref) / std_ref
            true4_norm = target_norm_ref2.view(B, 1, Np, P)
            std_true_norm_patch = true4_norm.std(dim=-1).mean().item()

            std_ref_patch = std_ref.view(B, 1, Np, P).mean(dim=-1).mean().item()

            logger.info("[debug] patch std (norm) pred:", std_pred_norm_patch,
                  "| true:", std_true_norm_patch,
                  "| std_ref_patch_mean:", std_ref_patch)

            pred4_raw = future_target_pred.view(B, 1, Np, P)
            true4_raw = future_target_true.view(B, 1, Np, P)
            logger.info("[debug] patch std (raw) pred:", pred4_raw.std(dim=-1).mean().item(),
                  "| true:", true4_raw.std(dim=-1).mean().item())

            z = future_latents_true
            y1 = dbg["adapter_decode"](z, future_cov_aug_true)
            y2 = dbg["adapter_decode"](z, future_cov_aug_true[torch.randperm(z.size(0))])
            diff = (y1 - y2).abs().mean().item()
            logger.info("[probe] decoder sensitivity to cov:", diff)

            z1 = future_latents_true
            z2 = future_latents_true[torch.randperm(z1.size(0))]
            y1 = dbg["adapter_decode"](z1, future_cov_aug_true)
            y2 = dbg["adapter_decode"](z2, future_cov_aug_true)
            diffz = (y1 - y2).abs().mean().item()
            logger.info("[probe] decoder sensitivity to latents(proxy):", diffz)

        logger.info("================================================\n")

class EarlyStopper(Callback):
    """
    根据 val loss 决定是否 stop，并保存 best_state_dict
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_value = float("inf")
        self.best_epoch = -1
        self.bad_count = 0
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        self.should_stop = False

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        mode = state["mode"]
        if mode != "val":
            return

        epoch = state["epoch"]
        metrics = state["metrics"]
        model = state["adapter"]  # nn.Module

        current = float(metrics.get("loss", float("inf")))
        improved = (self.best_value - current) > self.min_delta

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.bad_count = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_count += 1

        if self.verbose:
            logger.info(f"[earlystop] best={self.best_value:.6f} (epoch={self.best_epoch}) "
                  f"bad_count={self.bad_count}/{self.patience}")

        if self.bad_count >= self.patience:
            self.should_stop = True
            if self.verbose:
                logger.info(f"[earlystop] stop at epoch={epoch}, best_epoch={self.best_epoch}, best_loss={self.best_value:.6f}")


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
    
    def _patch_stats_1d(self, x: torch.Tensor, patch: int):
        B, C, T = x.shape
        assert T % patch == 0
        Np = T // patch
        x4 = x.view(B, C, Np, patch)
        mean = x4.mean(dim=-1)
        std = x4.std(dim=-1) + 1e-5
        return mean, std

    def _latent_stats_loss(self, past_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
        past_mu = past_latents.mean(dim=-1)
        future_mu = future_latents.mean(dim=-1)
        past_std = past_latents.std(dim=-1)
        future_std = future_latents.std(dim=-1)
        return F.mse_loss(past_mu, future_mu) + F.mse_loss(past_std, future_std)

    def _compute_alpha_stats(self, epoch: int, n_epochs: int) -> float:
        progress = epoch / max(1, n_epochs - 1)
        return float(np.clip((progress - 0.1) / 0.4, 0.0, 1.0))

    def _norm_past(self, y_past: torch.Tensor, patch_size: int, use_patch_revin: bool):
        if use_patch_revin:
            y_norm, mu_t, std_t, _, _ = revin_patch_norm_target(y_past, patch_size=patch_size)
        else:
            y_norm, mu_t, std_t = self._identity_norm(y_past)
        return y_norm, mu_t, std_t

    def _future_true_stats(self, y_future: torch.Tensor, patch_size: int, use_patch_revin: bool):
        if use_patch_revin:
            mu_t_true, std_t_true, mu_p_true, std_p_true = revin_patch_stats_target(y_future, patch_size=patch_size)
        else:
            mu_t_true, std_t_true = self._identity_stats(y_future)
            mu_p_true, std_p_true = self._patch_stats_1d(y_future, patch_size)
        return mu_t_true, std_t_true, mu_p_true, std_p_true

    def _mix_future_time_stats(
        self,
        mu_t_true: torch.Tensor,
        std_t_true: torch.Tensor,
        mu_t_hat: torch.Tensor,
        std_t_hat: torch.Tensor,
        alpha_stats: float,
        std_floor: float = 0.05,
    ):
        mu_mix = (1.0 - alpha_stats) * mu_t_true + alpha_stats * mu_t_hat
        logstd_true = torch.log(std_t_true + 1e-5)
        logstd_hat = torch.log(std_t_hat + 1e-5)
        logstd_mix = (1.0 - alpha_stats) * logstd_true + alpha_stats * logstd_hat
        std_mix = torch.exp(logstd_mix)
        std_mix = torch.clamp(std_mix, min=std_floor)
        return mu_mix, std_mix

    # ---------- LTM prediction ----------
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
            channel_pred = per_channel_outputs[channel_idx].predictions  # (B,1,H)
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
    
    
    # ---------- unified forward (train/eval shared) ----------
    def _forward_once(
        self,
        y_past: torch.Tensor,
        x_past: torch.Tensor,
        y_future: torch.Tensor,
        x_future: torch.Tensor,
        *,
        alpha_stats: float,
        use_patch_revin: bool,
        ltm_batch_size: int,
        loss_weights: Dict[str, float],
        mode: str,  # "train" or "val"
        debug_collect: bool = False,
    ) -> Dict[str, Any]:
        """
        返回：
          out["losses"]：tensor losses（包含总 loss）
          out["debug"]：用于画图/打印的张量（只在 debug_collect=True 时提供）
        """
        P_past = self.adapter.revin_patch_size_past
        P_fut = self.adapter.revin_patch_size_future

        # -------- stats & norms --------
        y_past_norm, past_mu_t, past_std_t = self._norm_past(y_past, P_past, use_patch_revin)
        future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = self._future_true_stats(y_future, P_fut, use_patch_revin)

        # adapter: predict stats from future covariates
        future_mu_t_hat, future_std_t_hat, future_mu_p_hat, future_std_p_hat, future_logstd_p_hat = self.adapter.predict_future_stats(x_future)

        # mix time-level stats
        mu_hat_for_mix = future_mu_t_hat.detach()
        mu_ref, std_ref = self._mix_future_time_stats(future_mu_t_true, future_std_t_true, mu_hat_for_mix, future_std_t_hat, alpha_stats)

        # -------- encode / decode --------
        past_latents = self.adapter.encode(y_past_norm, x_past)
        if past_latents.size(1) != 1:
            raise ValueError(f"[{mode}] latent_dim must be 1 for proxy/residual setting, got {past_latents.size(1)}")

        y_future_norm_true = (y_future - future_mu_t_true) / (future_std_t_true + 1e-5)
        future_latents_true = self.adapter.encode(y_future_norm_true, x_future)
        if future_latents_true.size(1) != 1:
            raise ValueError(f"[{mode}] latent_dim must be 1, got {future_latents_true.size(1)}")

        # past recon
        past_target_recon_norm = self.adapter.decode(past_latents, x_past)
        if use_patch_revin:
            past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
            loss_past_recon = F.mse_loss(past_target_recon, y_past)
        else:
            loss_past_recon = F.mse_loss(past_target_recon_norm, y_past_norm)
            

        # future pred via LTM + decoder
        future_latents_pred = self._predict_future_latents_with_ltm(
            past_latents=past_latents,
            future_horizon=y_future.shape[-1],
            ltm_batch_size=ltm_batch_size,
        )
        
        future_target_pred_norm = self.adapter.decode(future_latents_pred, x_future)
        if use_patch_revin:
            # denorm with mixed stats
            future_target_pred = revin_patch_denorm_target(future_target_pred_norm, mu_ref.detach(), std_ref.detach())
            loss_future_pred = F.mse_loss(future_target_pred, y_future)
            target_norm_ref = (y_future - mu_ref) / std_ref
        else:
            # no revin
            future_target_pred = future_target_pred_norm
            loss_future_pred = F.mse_loss(future_target_pred_norm, y_future)
            target_norm_ref = y_future
            mu_ref = torch.zeros_like(y_future)
            std_ref = torch.ones_like(y_future)

        # -------- stats predictor loss (patch-level) --------
        future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
        loss_stats_pred = F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5.0 * F.mse_loss(future_logstd_p_hat, future_logstd_p_true)

        # -------- latent stats alignment --------
        loss_latent_stats = self._latent_stats_loss(past_latents.detach(), future_latents_true)

        # -------- y patch moment loss --------
        loss_y_patch_moment = torch.tensor(0.0, device=self.device)
        if loss_weights.get("y_patch_moment", 0.0) > 0.0:
            P = P_fut
            B, _, H = future_target_pred_norm.shape
            assert H % P == 0
            Np = H // P
            pred4_norm = future_target_pred_norm.view(B, 1, Np, P)
            true4_norm = target_norm_ref.view(B, 1, Np, P)
            m_pred = pred4_norm.mean(dim=-1)
            m_true = true4_norm.mean(dim=-1)
            ms_pred = (pred4_norm ** 2).mean(dim=-1)
            ms_true = (true4_norm ** 2).mean(dim=-1)
            loss_y_patch_moment = F.mse_loss(ms_pred, ms_true) + 0.5 * F.mse_loss(m_pred, m_true)

        # -------- total --------
        total_loss = (
            loss_weights.get("past_recon", 1.0) * loss_past_recon
            + loss_weights.get("future_pred", 1.0) * loss_future_pred
            + loss_weights.get("latent_stats", 0.1) * loss_latent_stats
            + loss_weights.get("stats_pred", 0.1) * loss_stats_pred
            + loss_weights.get("y_patch_moment", 0.0) * loss_y_patch_moment
        )

        out: Dict[str, Any] = {
            "losses": {
                "loss": total_loss,
                "past_recon": loss_past_recon,
                "future_pred": loss_future_pred,
                "latent_stats": loss_latent_stats,
                "stats_pred": loss_stats_pred,
                "y_patch_moment": loss_y_patch_moment,
            }
        }

        if debug_collect:
            # 用于 PlotDumper
            out["debug"] = {
                "alpha_stats": alpha_stats,
                "P_past": P_past,
                "P_fut": P_fut,

                "past_latents": past_latents.detach(),
                "future_latents_pred": future_latents_pred.detach(),
                "future_latents_true": future_latents_true.detach(),

                "past_target_norm": y_past_norm.detach(),
                "past_target_recon_norm": past_target_recon_norm.detach(),

                "future_target_pred_norm": future_target_pred_norm.detach(),
                "target_norm_ref": target_norm_ref.detach(),

                "future_target_pred": future_target_pred.detach(),
                "future_target_true": y_future.detach(),

                "mu_ref": mu_ref.detach(),
                "std_ref": std_ref.detach(),

                "future_covariates_aug_true": x_future.detach(),
                "adapter_decode": self.adapter.decode,            # probe 用
            }

        return out
    
    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch: int,
        n_epochs: int,
        mode: str,  # "train" or "val"
        optimizer: Optional[torch.optim.Optimizer],
        use_patch_revin: bool,
        ltm_batch_size: int,
        loss_weights: Dict[str, float],
        callbacks: Optional[List[Callback]] = None,
        debug_every: int = 3,
    ) -> Dict[str, float]:
        is_train = (mode == "train")
        if is_train:
            self.adapter.train()
        else:
            self.adapter.eval()

        self.iclearner.eval()
        alpha_stats = self._compute_alpha_stats(epoch, n_epochs)

        meter = EpochMeters.new(keys=["loss", "past_recon", "future_pred", "latent_stats", "stats_pred", "y_patch_moment"])

        # debug 只在 train 且 epoch%debug_every==0 时收集一次
        want_debug = (mode == "train") and (epoch % debug_every == 0)

        for (y_p, x_p, y_f, x_f) in dataloader:
            y_p = y_p.to(self.device)
            x_p = x_p.to(self.device)
            y_f = y_f.to(self.device)
            x_f = x_f.to(self.device)

            debug_collect = want_debug  # 由 PlotDumper 控制只缓存一个 batch；这里每 batch 都给 state，但回调会只取第一次

            with torch.set_grad_enabled(is_train):
                out = self._forward_once(
                    y_past=y_p,
                    x_past=x_p,
                    y_future=y_f,
                    x_future=x_f,
                    alpha_stats=alpha_stats,
                    use_patch_revin=use_patch_revin,
                    ltm_batch_size=ltm_batch_size,
                    loss_weights=loss_weights,
                    mode=mode,
                    debug_collect=debug_collect,
                )

                if is_train:
                    optimizer.zero_grad()
                    out["losses"]["loss"].backward()

                    if want_debug:
                        gnorm = 0.0
                        for n, p in self.adapter.named_parameters():
                            if p.grad is None:
                                continue
                            if "target_decoder" in n:
                                gnorm += p.grad.detach().data.norm(2).item() ** 2
                        gnorm = gnorm ** 0.5
                        logger.info(f"[dbg] grad_norm(target_decoder)={gnorm:.4g}")

                        # 防止一个 epoch 多次打印：打印一次后关闭 want_debug 的打印部分
                        want_debug = False

                    optimizer.step()

            bsz = y_p.size(0)
            meter.update(out["losses"], bsz)

            if callbacks:
                state = {
                    "mode": mode,
                    "epoch": epoch,
                    "alpha_stats": alpha_stats,
                    "out": out,
                    "debug": out.get("debug", None),
                }
                for cb in callbacks:
                    cb.on_batch_end(state)

        metrics = meter.mean()

        if callbacks:
            state = {
                "mode": mode,
                "epoch": epoch,
                "alpha_stats": alpha_stats,
                "metrics": metrics,
                "adapter": self.adapter,
            }
            for cb in callbacks:
                cb.on_epoch_end(state)

        return metrics
    
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
        lambda_latent_stats: float = 0.1,
        lambda_stats_pred: float = 0.1,
        lambda_y_patch_std: float = 0.2,
        ltm_batch_size: int = 1,
        verbose: bool = True,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        use_patch_revin: bool = False,
        use_swanlab: bool = True,
        swanlab_run=None,
        debug: bool = True,
        debug_dir: str = "./debug_train_adapter",
        debug_every: int = 2,
        debug_num_seq: int = 2,
        debug_latent_ch: int = 1,
        earlystop_patience: int = 10,
    ):
        dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data is not None:
            val_dataset = TensorDataset(
                torch.tensor(val_data["past_target"], dtype=torch.float32),
                torch.tensor(val_data["past_covariates"], dtype=torch.float32),
                torch.tensor(val_data["future_target"], dtype=torch.float32),
                torch.tensor(val_data["future_covariates"], dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        self.iclearner.eval()
        for p in self.iclearner.backbone.parameters():
            p.requires_grad_(False)

        loss_weights = {
            "past_recon": lambda_past_recon,
            "future_pred": lambda_future_pred,
            "latent_stats": lambda_latent_stats,
            "stats_pred": lambda_stats_pred,
            "y_patch_moment": lambda_y_patch_std,
        }

        callbacks: List[Callback] = []
        if verbose:
            callbacks.append(ConsoleLogger())

        if debug:
            callbacks.append(
                PlotDumper(
                    out_dir=debug_dir,
                    every=debug_every,
                    num_seq=debug_num_seq,
                    latent_ch=debug_latent_ch,
                    enabled=True,
                )
            )

        earlystop = EarlyStopper(patience=earlystop_patience, verbose=verbose)
        callbacks.append(earlystop)

        for epoch in range(n_epochs):
            # train
            train_metrics = self._run_epoch(
                dataloader=train_loader,
                epoch=epoch,
                n_epochs=n_epochs,
                mode="train",
                optimizer=optimizer,
                use_patch_revin=use_patch_revin,
                ltm_batch_size=ltm_batch_size,
                loss_weights=loss_weights,
                callbacks=callbacks,
                debug_every=debug_every,
            )

            # swanlab
            if use_swanlab and (swanlab_run is not None):
                log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
                log_dict["epoch"] = epoch
                swanlab_run.log(log_dict)

            # val
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    dataloader=val_loader,
                    epoch=epoch,
                    n_epochs=n_epochs,
                    mode="val",
                    optimizer=None,
                    use_patch_revin=use_patch_revin,
                    ltm_batch_size=ltm_batch_size,
                    loss_weights=loss_weights,
                    callbacks=callbacks,
                    debug_every=debug_every,
                )

                if use_swanlab and (swanlab_run is not None):
                    log_dict = {f"val/{k}": v for k, v in val_metrics.items()}
                    log_dict["epoch"] = epoch
                    swanlab_run.log(log_dict)

            else:
                # 无 val：用 train loss 做 early stop
                fake_val_state = {
                    "mode": "val",
                    "epoch": epoch,
                    "metrics": train_metrics,
                    "adapter": self.adapter,
                }
                earlystop.on_epoch_end(fake_val_state)

            if earlystop.should_stop:
                break

        # restore best
        if earlystop.best_state_dict is not None:
            self.adapter.load_state_dict(earlystop.best_state_dict)
            if verbose:
                logger.info(f"[earlystop] restored best adapter weights from epoch={earlystop.best_epoch} "
                      f"(best_loss={earlystop.best_value:.6f})")
        else:
            if verbose:
                logger.info("[earlystop] WARNING: no best_state_dict saved (maybe no epochs ran?)")
    
    
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
                
                # logger.info('std:true vs pred', logstd_true, logstd_hat)
                
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
                # 如果没有 val，就用 train_loss 监控 early stop
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
                logger.info(msg)
            
            if bad_count >= patience:
                if verbose:
                    logger.info(f"[stats-pretrain] early stop at epoch={epoch}, "
                          f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                break
        
        if best_state_dict is not None:
            self.adapter.future_stats_predictor.load_state_dict(best_state_dict)
            if verbose:
                logger.info(f"[stats-pretrain] restored best future_stats_predictor "
                      f"from epoch={best_epoch} (val_loss={best_val_loss:.6f})")
        else:
            if verbose:
                logger.info("[stats-pretrain] WARNING: no best_state_dict saved (maybe no epochs ran?)")

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
        y_recon = decoder(latents, x_past)       # (B,1,L)
        loss = MSE(y_recon, y_past)
        """

        os.makedirs(debug_dir, exist_ok=True)

        def _to_np(x: torch.Tensor):
            return x.detach().float().cpu().numpy()

        def _print_stats(name: str, x: torch.Tensor):
            x_ = x.detach()
            logger.info(f"[dbg] {name:28s} shape={tuple(x_.shape)} "
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
            logger.info(f"[dbg] {name:28s} patch_mean[:{k}]={mu_np}")
            logger.info(f"[dbg] {name:28s} patch_std [: {k}]={sd_np}")
            logger.info(f"[dbg] {name:28s} patch_std summary min/mean/max="
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

                y_recon = self.adapter.decode(latents, x_p)

                loss = F.mse_loss(y_recon, y_p)

                optim.zero_grad()
                loss.backward()
                optim.step()

                total += loss.item() * y_p.size(0)
                count += y_p.size(0)

                # sanity + debug 每个 epoch 第一个 batch
                if debug and (not dumped):
                    dumped = True
                    logger.info("\n=========== [Pretrain past recon ONLY] DEBUG ===========")
                    logger.info(f"[dbg] epoch={epoch}/{n_epochs-1} batch={y_p.size(0)}")

                    _print_stats("y_past", y_p)
                    _print_stats("latents(trend)", latents)
                    _print_stats("y_recon", y_recon)

                    _print_patch("y_past", y_p, P_dbg, b=0)
                    _print_patch("latents(trend)", latents, P_dbg, b=0)
                    _print_patch("y_recon", y_recon, P_dbg, b=0)

                    with torch.no_grad():
                        perm = torch.randperm(y_p.size(0), device=self.device)
                        y_recon_shuf_cov = self.adapter.decode(latents, x_p[perm])
                        cov_sens = (y_recon - y_recon_shuf_cov).abs().mean().item()

                        y_recon_shuf_proxy = self.adapter.decode(latents[perm], x_p)
                        proxy_sens = (y_recon - y_recon_shuf_proxy).abs().mean().item()

                        logger.info(f"[probe] decoder sensitivity cov   : {cov_sens:.4f}")
                        logger.info(f"[probe] decoder sensitivity proxy : {proxy_sens:.4f}")

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
                            {"latents(trend)": _to_np(latents[i, 0]), "y_past": _to_np(y_p[i, 0])},
                            title=f"Stage1 latent & y | ep={epoch} b={i}",
                        )
                    logger.info("=========================================================\n")

            if verbose:
                logger.info(f"[stage1-pretrain] epoch={epoch:03d} loss_past_recon={total/max(1,count):.6f}")     
    

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

        self.adapter.train()
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

                future_latents_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )

                future_target_pred_norm = self.adapter.decode(future_latents_pred, future_covariates_aug)
                if use_patch_revin:
                    future_target_pred = revin_patch_denorm_target(future_target_pred_norm, future_mu_t_hat, torch.clamp(future_std_t_hat,min=0.05))
                else:
                    future_target_pred = future_target_pred_norm

                
                all_future_target_samples.append(future_target_pred.detach().cpu().numpy())
                
                if sample_idx == 0:
                    logger.info("------------------------predict----------------------")
                    logger.info("\n==== latent stats ====")
                    logger.info("past_latents(trend)   min/mean/max:", past_latents.min().item(), past_latents.mean().item(), past_latents.max().item())
                    logger.info("past_latents(trend)   std:", past_latents.std().item())

                    logger.info("future_cov     min/mean/max:", future_covariates_tensor.min().item(), future_covariates_tensor.mean().item(), future_covariates_tensor.max().item()) 
                    logger.info("future_target_pred_norm std:", future_target_pred_norm.std().item())

                    mu_patch_hat, std_patch_hat, _ = self.adapter.future_stats_predictor(future_covariates_tensor)
                    logger.info("mu_patch_hat min/mean/max:", mu_patch_hat.min().item(), mu_patch_hat.mean().item(), mu_patch_hat.max().item())
                    logger.info("std_patch_hat min/mean/max:", std_patch_hat.min().item(), std_patch_hat.mean().item(), std_patch_hat.max().item())

        all_future_target_samples = np.stack(all_future_target_samples, axis=0)  # (S,B,1,H)
        mean = all_future_target_samples.mean(axis=0)
        std = all_future_target_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
