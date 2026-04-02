# conditional_adapts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import copy
import numpy as np
import pandas as pd
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
        future_cov_aug_true = dbg.get("future_covariates_aug_true", dbg.get("future_covariates_true", None))
        adapter_decode = dbg.get("adapter_decode", state.get("adapter", None).decode if state.get("adapter", None) is not None else None)

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

            if (future_cov_aug_true is not None) and (adapter_decode is not None):
                z = future_latents_true
                y1 = adapter_decode(z, future_cov_aug_true)
                y2 = adapter_decode(z, future_cov_aug_true[torch.randperm(z.size(0), device=z.device)])
                diff = (y1 - y2).abs().mean().item()
                logger.info("[probe] decoder sensitivity to cov:", diff)

                z1 = future_latents_true
                z2 = future_latents_true[torch.randperm(z1.size(0), device=z1.device)]
                y1 = adapter_decode(z1, future_cov_aug_true)
                y2 = adapter_decode(z2, future_cov_aug_true)
                diffz = (y1 - y2).abs().mean().item()
                logger.info("[probe] decoder sensitivity to latents(proxy):", diffz)
            else:
                logger.info("[probe] skip decoder sensitivity probes: missing future covariates or adapter_decode")

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
    """
    adapter: PipelineAdapter
        encoder / decoder / normalizer / stats_predictor(optional)
    """
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
    
    # def _identity_norm(self, y: torch.Tensor):
    #     # y: (B,1,T)
    #     mu = torch.zeros_like(y)
    #     std = torch.ones_like(y)
    #     return y, mu, std

    # def _identity_stats(self, y: torch.Tensor):
    #     mu = torch.zeros_like(y)
    #     std = torch.ones_like(y)
    #     return mu, std

    # def _identity_denorm(self, y_norm: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
    #     return y_norm
    
    # def _patch_stats_1d(self, x: torch.Tensor, patch: int):
    #     B, C, T = x.shape
    #     assert T % patch == 0
    #     Np = T // patch
    #     x4 = x.view(B, C, Np, patch)
    #     mean = x4.mean(dim=-1)
    #     std = x4.std(dim=-1) + 1e-5
    #     return mean, std

    def _latent_stats_loss(self, past_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
        past_mu = past_latents.mean(dim=-1)
        future_mu = future_latents.mean(dim=-1)
        past_std = past_latents.std(dim=-1)
        future_std = future_latents.std(dim=-1)
        return F.mse_loss(past_mu, future_mu) + F.mse_loss(past_std, future_std)

    def _compute_alpha_stats(self, epoch: int, n_epochs: int) -> float:
        progress = epoch / max(1, n_epochs - 1)
        return float(np.clip((progress - 0.1) / 0.4, 0.0, 1.0))

    # def _norm_past(self, y_past: torch.Tensor, patch_size: int, use_patch_revin: bool):
    #     if use_patch_revin:
    #         y_norm, mu_t, std_t, _, _ = revin_patch_norm_target(y_past, patch_size=patch_size)
    #     else:
    #         y_norm, mu_t, std_t = self._identity_norm(y_past)
    #     return y_norm, mu_t, std_t

    # def _future_true_stats(self, y_future: torch.Tensor, patch_size: int, use_patch_revin: bool):
    #     if use_patch_revin:
    #         mu_t_true, std_t_true, mu_p_true, std_p_true = revin_patch_stats_target(y_future, patch_size=patch_size)
    #     else:
    #         mu_t_true, std_t_true = self._identity_stats(y_future)
    #         mu_p_true, std_p_true = self._patch_stats_1d(y_future, patch_size)
    #     return mu_t_true, std_t_true, mu_p_true, std_p_true

    # def _mix_future_time_stats(
    #     self,
    #     mu_t_true: torch.Tensor,
    #     std_t_true: torch.Tensor,
    #     mu_t_hat: torch.Tensor,
    #     std_t_hat: torch.Tensor,
    #     alpha_stats: float,
    #     std_floor: float = 0.05,
    # ):
    #     mu_mix = (1.0 - alpha_stats) * mu_t_true + alpha_stats * mu_t_hat
    #     logstd_true = torch.log(std_t_true + 1e-5)
    #     logstd_hat = torch.log(std_t_hat + 1e-5)
    #     logstd_mix = (1.0 - alpha_stats) * logstd_true + alpha_stats * logstd_hat
    #     std_mix = torch.exp(logstd_mix)
    #     std_mix = torch.clamp(std_mix, min=std_floor)
    #     return mu_mix, std_mix

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
        # use_patch_revin: bool,
        ltm_batch_size: int,
        loss_weights: Dict[str, float],
        mode: str,  # "train" or "val"
        debug_collect: bool = False,
        block_future_pred_to_decoder: bool = False,
    ) -> Dict[str, Any]:
        """
        返回：
          out["losses"]：tensor losses（包含总 loss）
          out["debug"]：用于画图/打印的张量（只在 debug_collect=True 时提供）
        """
        P_past = self.adapter.revin_patch_size_past
        P_fut = self.adapter.revin_patch_size_future

        # -------- stats & norms --------
        y_past_norm, past_mu_t, past_std_t = self.adapter.normalizer.norm_past(y_past, patch_size=P_past)

        future_mu_t_true, future_std_t_true, future_mu_p_true, future_std_p_true = \
            self.adapter.normalizer.future_true_stats(y_future, patch_size=P_fut)

        # adapter: predict stats from future covariates
        future_mu_t_hat, future_std_t_hat, future_mu_p_hat, future_std_p_hat, future_logstd_p_hat = \
            self.adapter.predict_future_stats(x_future)

        # mix time-level stats
        # mu_hat_for_mix = future_mu_t_hat.detach()
        # mu_ref, std_ref = self._mix_future_time_stats(future_mu_t_true, future_std_t_true, mu_hat_for_mix, future_std_t_hat, alpha_stats)
        mu_ref, std_ref = self.adapter.normalizer.mix_future_time_stats(
            mu_t_true=future_mu_t_true,
            std_t_true=future_std_t_true,
            mu_t_hat=future_mu_t_hat,
            std_t_hat=future_std_t_hat,
            alpha_stats=alpha_stats,
        )

        # -------- encode / decode --------
        past_latents = self.adapter.encode(y_past_norm, x_past)
        if past_latents.size(1) != 1:
            raise ValueError(f"[{mode}] latent_dim must be 1 for proxy/residual setting, got {past_latents.size(1)}")

        y_future_norm_true = self.adapter.normalizer.norm_future_with_true_stats(
            y_future, mu_t_true=future_mu_t_true, std_t_true=future_std_t_true
        )
        future_latents_true = self.adapter.encode(y_future_norm_true, x_future)
        if future_latents_true.size(1) != 1:
            raise ValueError(f"[{mode}] latent_dim must be 1, got {future_latents_true.size(1)}")

        #* past recon loss
        past_target_recon_norm = self.adapter.decode(past_latents, x_past)
        loss_past_recon = F.mse_loss(past_target_recon_norm, y_past_norm)
        # if use_patch_revin:
        #     past_target_recon = revin_patch_denorm_target(past_target_recon_norm, past_mu_t, past_std_t)
        #     loss_past_recon = F.mse_loss(past_target_recon, y_past)
        # else:
        #     loss_past_recon = F.mse_loss(past_target_recon_norm, y_past_norm)
            

        # future pred via LTM + decoder
        future_latents_pred = self._predict_future_latents_with_ltm(
            past_latents=past_latents,
            future_horizon=y_future.shape[-1],
            ltm_batch_size=ltm_batch_size,
        )
        
        future_target_pred_norm = self.adapter.decode(future_latents_pred, x_future)

        future_target_pred = self.adapter.normalizer.denorm_future_pred(
            y_future_pred_norm=future_target_pred_norm,
            mu_ref=mu_ref.detach(),
            std_ref=std_ref.detach(),
        )
        
        # if use_patch_revin:
        #     # denorm with mixed stats
        #     future_target_pred = revin_patch_denorm_target(future_target_pred_norm, mu_ref.detach(), std_ref.detach())
        #     loss_future_pred = F.mse_loss(future_target_pred, y_future)
        #     target_norm_ref = (y_future - mu_ref) / std_ref
        # else:
        #     # no revin
        #     future_target_pred = future_target_pred_norm
        #     loss_future_pred = F.mse_loss(future_target_pred_norm, y_future)
        #     target_norm_ref = y_future
        #     mu_ref = torch.zeros_like(y_future)
        #     std_ref = torch.ones_like(y_future)
        target_norm_ref = self.adapter.normalizer.target_norm_ref(
            y_future=y_future,
            mu_ref=mu_ref.detach(),
            std_ref=std_ref.detach(),
        )
        
        #* future pred loss
        loss_future_pred = self.adapter.normalizer.future_pred_loss(
            y_pred=future_target_pred,
            y_true=y_future,
            y_pred_norm=future_target_pred_norm,
            y_true_norm=y_future,  # identity uses this; revin_patch ignores
        )

        #* stats predictor loss
        loss_stats_pred = torch.tensor(0.0, device=self.device)
        if loss_weights.get("stats_pred", 0.0) > 0.0 and self.adapter.has_stats_predictor:
            # define "true logstd patch" from true patch std
            future_logstd_p_true = torch.log(future_std_p_true + 1e-5)
            loss_stats_pred = F.mse_loss(future_mu_p_hat, future_mu_p_true) + 5.0 * F.mse_loss(future_logstd_p_hat, future_logstd_p_true)

        #* latent stats alignment loss
        loss_latent_stats = torch.tensor(0.0, device=self.device)
        if loss_weights.get("latent_stats", 0.0) > 0.0:
            loss_latent_stats = self._latent_stats_loss(past_latents.detach(), future_latents_true)

        #* y patch moment loss
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
            out["debug"] = {
                "alpha_stats": alpha_stats,
                "P_past": P_past,
                "P_fut": P_fut,
                "past_latents": past_latents.detach(),
                "future_latents_pred": future_latents_pred.detach(),
                "future_latents_true": future_latents_true.detach(),
                "future_covariates_true": x_future.detach(),
                "adapter_decode": self.adapter.decode,
                "past_target_norm": y_past_norm.detach(),
                "past_target_recon_norm": past_target_recon_norm.detach(),
                "future_target_pred_norm": future_target_pred_norm.detach(),
                "target_norm_ref": target_norm_ref.detach(),
                "future_target_pred": future_target_pred.detach(),
                "future_target_true": y_future.detach(),
                "mu_ref": mu_ref.detach(),
                "std_ref": std_ref.detach(),
            }
        return out

    def _run_epoch(
        self,
        dataloader: DataLoader,
        *,
        epoch: int,
        n_epochs: int,
        mode: str,  # train or eval
        optimizer: Optional[Any],
        ltm_batch_size: int,
        loss_weights: Dict[str, float],
        callbacks: Optional[List[Callback]] = None,
        debug_every: int = 3,
        block_future_pred_to_decoder: bool = False,
    ) -> Dict[str, float]:
        is_train = (mode == "train")
        if is_train:
            self.adapter.train()
        else:
            self.adapter.eval()

        self.iclearner.eval()
        alpha_stats = self._compute_alpha_stats(epoch, n_epochs)

        meter = EpochMeters.new(keys=["loss", "past_recon", "future_pred", "latent_stats", "stats_pred", "y_patch_moment"])
        want_debug = (mode == "train") and (epoch % debug_every == 0)

        dual_optimizer_mode = (
            is_train
            and block_future_pred_to_decoder
            and isinstance(optimizer, dict)
            and ("encoder" in optimizer)
            and ("decoder" in optimizer)
        )

        for (y_p, x_p, y_f, x_f) in dataloader:
            y_p = y_p.to(self.device)
            x_p = x_p.to(self.device)
            y_f = y_f.to(self.device)
            x_f = x_f.to(self.device)

            debug_collect = want_debug

            with torch.set_grad_enabled(is_train):
                if dual_optimizer_mode:
                    optimizer_decoder = optimizer["decoder"]
                    optimizer_encoder = optimizer["encoder"]

                    # -------------------------
                    # decoder step: past reconstruction only
                    # -------------------------
                    optimizer_decoder.zero_grad()
                    optimizer_encoder.zero_grad()

                    out_dec = self._forward_once(
                        y_past=y_p,
                        x_past=x_p,
                        y_future=y_f,
                        x_future=x_f,
                        alpha_stats=alpha_stats,
                        ltm_batch_size=ltm_batch_size,
                        loss_weights=loss_weights,
                        mode=mode,
                        debug_collect=False,
                        block_future_pred_to_decoder=False,
                    )
                    out_dec["losses"]["past_recon"].backward()
                    optimizer_decoder.step()

                    # 清掉第一次 backward 在 encoder / decoder 上留下的梯度
                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()

                    # -------------------------
                    # encoder step: total loss = past + future + others
                    # -------------------------
                    out = self._forward_once(
                        y_past=y_p,
                        x_past=x_p,
                        y_future=y_f,
                        x_future=x_f,
                        alpha_stats=alpha_stats,
                        ltm_batch_size=ltm_batch_size,
                        loss_weights=loss_weights,
                        mode=mode,
                        debug_collect=debug_collect,
                        block_future_pred_to_decoder=False,
                    )
                    out["losses"]["loss"].backward()
                    optimizer_encoder.step()

                    # 清掉第二次 backward 在 decoder 上产生但不用于更新的梯度
                    optimizer_decoder.zero_grad()

                    if want_debug:
                        want_debug = False
                else:
                    out = self._forward_once(
                        y_past=y_p,
                        x_past=x_p,
                        y_future=y_f,
                        x_future=x_f,
                        alpha_stats=alpha_stats,
                        ltm_batch_size=ltm_batch_size,
                        loss_weights=loss_weights,
                        mode=mode,
                        debug_collect=debug_collect,
                        block_future_pred_to_decoder=block_future_pred_to_decoder,
                    )

                    if is_train:
                        optimizer.zero_grad()
                        out["losses"]["loss"].backward()
                        optimizer.step()

                        if want_debug:
                            want_debug = False

            bsz = y_p.size(0)
            meter.update(out["losses"], bsz)

            if callbacks:
                state = {
                    "mode": mode,
                    "epoch": epoch,
                    "alpha_stats": alpha_stats,
                    "out": out,
                    "debug": out.get("debug", None),
                    "adapter": self.adapter,
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
        use_swanlab: bool = True,
        swanlab_run=None,
        debug: bool = True,
        debug_dir: str = "./debug_train_adapter",
        debug_every: int = 2,
        debug_num_seq: int = 2,
        debug_latent_ch: int = 1,
        earlystop_patience: int = 10,
        # ===== new args =====
        freeze_decoder: bool = False,
        block_future_pred_to_decoder: bool = False,
        encoder_lr: Optional[float] = None,
        decoder_lr: Optional[float] = None,
        stats_lr: Optional[float] = None,
        
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ):
        
        self.adapter.train()
        if not getattr(self.adapter, "is_trainable", True):
            logger.info("[train_adapter] adapter is not trainable -> skip.")
            return

        train_dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        
        dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = persistent_workers
            dl_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **dl_kwargs,
        )

        val_loader = None
        if val_data is not None:
            val_dataset = TensorDataset(
                torch.tensor(val_data["past_target"], dtype=torch.float32),
                torch.tensor(val_data["past_covariates"], dtype=torch.float32),
                torch.tensor(val_data["future_target"], dtype=torch.float32),
                torch.tensor(val_data["future_covariates"], dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **dl_kwargs,
            )

        # -------------------------
        # freeze / unfreeze decoder
        # -------------------------
        if hasattr(self.adapter, "decoder"):
            if freeze_decoder:
                for p in self.adapter.decoder.parameters():
                    p.requires_grad_(False)
                logger.info("[train_adapter] decoder frozen.")
            else:
                for p in self.adapter.decoder.parameters():
                    p.requires_grad_(True)

        self.iclearner.eval()
        for p in self.iclearner.backbone.parameters():
            p.requires_grad_(False)

        # -------------------------
        # optimizer with param groups
        # -------------------------
        enc_lr = lr if encoder_lr is None else encoder_lr
        dec_lr = lr if decoder_lr is None else decoder_lr
        st_lr = lr if stats_lr is None else stats_lr

        known_param_ids = set()

        def _unique_trainable_params(module):
            params = []
            if module is None:
                return params
            for p in module.parameters():
                if (not p.requires_grad) or (id(p) in known_param_ids):
                    continue
                params.append(p)
                known_param_ids.add(id(p))
            return params

        enc_params = []
        dec_params = []
        stats_params = []

        if hasattr(self.adapter, "encoder") and self.adapter.encoder is not None:
            enc_params = _unique_trainable_params(self.adapter.encoder)

        if hasattr(self.adapter, "decoder") and self.adapter.decoder is not None:
            dec_params = _unique_trainable_params(self.adapter.decoder)

        if hasattr(self.adapter, "future_stats_predictor") and self.adapter.future_stats_predictor is not None:
            stats_params = _unique_trainable_params(self.adapter.future_stats_predictor)

        # 其余还可训练的参数（如 normalizer 等），默认跟 encoder_lr
        other_params = []
        for p in self.adapter.parameters():
            if (not p.requires_grad) or (id(p) in known_param_ids):
                continue
            other_params.append(p)
            known_param_ids.add(id(p))
            
            
        def _count_trainable(module):
            if module is None:
                return 0, 0
            total = 0
            trainable = 0
            for p in module.parameters():
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n
            return total, trainable

        enc_total, enc_train = _count_trainable(getattr(self.adapter, "encoder", None))
        dec_total, dec_train = _count_trainable(getattr(self.adapter, "decoder", None))
        fsp_total, fsp_train = _count_trainable(getattr(self.adapter, "future_stats_predictor", None))
        ada_total, ada_train = _count_trainable(self.adapter)

        logger.info(f"[train_adapter] encoder total/trainable: {enc_total}/{enc_train}")
        logger.info(f"[train_adapter] decoder total/trainable: {dec_total}/{dec_train}")
        logger.info(f"[train_adapter] future_stats_predictor total/trainable: {fsp_total}/{fsp_train}")
        logger.info(f"[train_adapter] adapter total/trainable: {ada_total}/{ada_train}")

        optimizer = None
        dual_optimizer_mode = block_future_pred_to_decoder and (len(dec_params) > 0)

        if dual_optimizer_mode:
            encoder_param_groups = []
            if len(enc_params) > 0:
                encoder_param_groups.append({
                    "params": enc_params,
                    "lr": enc_lr,
                })
            if len(stats_params) > 0:
                encoder_param_groups.append({
                    "params": stats_params,
                    "lr": st_lr,
                })
            if len(other_params) > 0:
                encoder_param_groups.append({
                    "params": other_params,
                    "lr": enc_lr,
                })

            if len(encoder_param_groups) == 0:
                logger.info("[train_adapter] block_future_pred_to_decoder=True but encoder-side trainable params are empty -> fallback to single optimizer.")
                dual_optimizer_mode = False
            else:
                optimizer = {
                    "encoder": torch.optim.Adam(encoder_param_groups, weight_decay=weight_decay),
                    "decoder": torch.optim.Adam([
                        {
                            "params": dec_params,
                            "lr": dec_lr,
                        }
                    ], weight_decay=weight_decay),
                }

        if not dual_optimizer_mode:
            param_groups = []
            if len(enc_params) > 0:
                param_groups.append({
                    "params": enc_params,
                    "lr": enc_lr,
                })
            if len(dec_params) > 0:
                param_groups.append({
                    "params": dec_params,
                    "lr": dec_lr,
                })
            if len(stats_params) > 0:
                param_groups.append({
                    "params": stats_params,
                    "lr": st_lr,
                })
            if len(other_params) > 0:
                param_groups.append({
                    "params": other_params,
                    "lr": enc_lr,
                })

            if len(param_groups) == 0:
                logger.info("[train_adapter] no trainable params -> skip.")
                return

            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)

        logger.info(
            f"[train_adapter] optimizer groups | "
            f"freeze_decoder={freeze_decoder} "
            f"block_future_pred_to_decoder={block_future_pred_to_decoder} "
            f"dual_optimizer_mode={dual_optimizer_mode} "
            f"encoder_lr={enc_lr:.2e} "
            f"decoder_lr={dec_lr:.2e} "
            f"stats_lr={st_lr:.2e}"
        )

        loss_weights = {
            "past_recon": lambda_past_recon,
            "future_pred": lambda_future_pred,
            "latent_stats": lambda_latent_stats,
            "stats_pred": lambda_stats_pred,
            "y_patch_moment": lambda_y_patch_std,
        }

        callbacks: List[Callback] = []
        if verbose:
            callbacks.append(ConsoleLogger(print_earlystop=True))
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

        early_stopper = None
        if val_loader is not None:
            early_stopper = EarlyStopper(
                patience=earlystop_patience,
                min_delta=1e-4,
                verbose=verbose,
            )
            callbacks.append(early_stopper)

        for epoch in range(1, n_epochs + 1):
            train_metrics = self._run_epoch(
                train_loader,
                epoch=epoch,
                n_epochs=n_epochs,
                mode="train",
                optimizer=optimizer,
                ltm_batch_size=ltm_batch_size,
                loss_weights=loss_weights,
                callbacks=callbacks,
                debug_every=debug_every,
                block_future_pred_to_decoder=block_future_pred_to_decoder,
            )

            if val_loader is not None:
                with torch.no_grad():
                    val_metrics = self._run_epoch(
                        val_loader,
                        epoch=epoch,
                        n_epochs=n_epochs,
                        mode="val",
                        optimizer=None,
                        ltm_batch_size=ltm_batch_size,
                        loss_weights=loss_weights,
                        callbacks=callbacks,
                        debug_every=debug_every,
                        block_future_pred_to_decoder=block_future_pred_to_decoder,
                    )

                if early_stopper is not None and early_stopper.should_stop:
                    logger.info(
                        f"[train_adapter] early stop at epoch={epoch}, "
                        f"best_epoch={early_stopper.best_epoch}, "
                        f"best_loss={early_stopper.best_value:.6f}"
                    )
                    break

        if early_stopper is not None and early_stopper.best_state_dict is not None:
            self.adapter.load_state_dict(early_stopper.best_state_dict)
            logger.info(
                f"[train_adapter] restored best adapter from epoch="
                f"{early_stopper.best_epoch} "
                f"(val_loss={early_stopper.best_value:.6f})"
            )
    
    
    def pretrain_stats_predictor(
        self,
        future_target: np.ndarray,
        future_covariates: np.ndarray,
        n_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_data: Optional[Dict[str, np.ndarray]] = None,
        patience: int = 10,
        verbose: bool = True,
        use_swanlab: bool = False,
        swanlab_run=None,
    ):
        """
        只训练 adapter.future_stats_predictor:
          x_future -> (mu_patch_hat, std_patch_hat)
        目标: 贴近 revin_patch_stats_target 得到的真实 patch 均值/方差
        """
        if not getattr(self.adapter, "has_stats_predictor", False):
            logger.info("[pretrain_stats_predictor] no stats_predictor -> skip.")
            return
        
        self.adapter.train()
        optimizer = torch.optim.Adam(self.adapter.future_stats_predictor.parameters(), lr=lr, weight_decay=weight_decay)
        
        P_fut = self.adapter.revin_patch_size_future
        
        train_dataset = TensorDataset(
            torch.tensor(future_target, dtype=torch.float32),
            torch.tensor(future_covariates, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = TensorDataset(
                torch.tensor(val_data["future_target"], dtype=torch.float32),
                torch.tensor(val_data["future_covariates"], dtype=torch.float32),
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
                _, _, mu_p_true, std_p_true = self.adapter.normalizer.future_true_stats(
                    batch_future_target, patch_size=P_fut
                )
                logstd_true = torch.log(std_p_true + 1e-5)
                
                # 预测 patch 统计
                mu_p_hat, std_p_hat, logstd_hat = self.adapter.future_stats_predictor(batch_future_cov)
                
                # 和 train_adapter 中保持一致的损失形式
                loss_mu = F.mse_loss(mu_p_hat, mu_p_true)
                loss_std = F.mse_loss(logstd_hat, logstd_true)
                loss = loss_mu + 2.0 * loss_std
                
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
                logger.info(f"[stats-pretrain] restored best future_stats_predictor from epoch={best_epoch} (val_loss={best_val_loss:.6f})")

    def pretrain_past_reconstruction_only(
        self,
        past_target: np.ndarray,        # (N,1,L) scaled
        past_covariates: np.ndarray,    # (N,Cx,L)
        n_epochs: int = 30,
        min_n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        # early stop
        val_data: Optional[Dict[str, np.ndarray]] = None,
        patience: int = 10,
        min_delta: float = 1e-4,
        # anti-shortcut regularization
        lambda_proxy_floor: float = 0.2,
        proxy_sens_floor: float = 0.10,
        lambda_ratio_floor: float = 0.2,
        proxy_cov_ratio_floor: float = 0.20,
        # early-stop monitor on proxy usage
        lambda_monitor: float = 1.0,
        # lambda_latent_l2: float = 0.5,
        # latent scale regularization (replace raw latent L2)
        lambda_latent_scale: float = 1e-4,
        latent_rms_target: float = 3.0,
        latent_std_floor=0.3,
        lambda_latent_std_floor = 1.0,
        # debug
        debug: bool = True,
        debug_dir: str = "./debug_no_revin_residual",
        debug_plot: bool = True,
        debug_num_seq: int = 2,
        verbose: bool = True,
        
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
    ):
        """
        Stage1: pretrain past reconstruction only

        latents = encoder(y_past_norm, x_past)    # (B,z_dim,L)
        y_recon = decoder(latents, x_past)        # (B,1,L)

        Base loss:
            recon_loss = MSE(y_recon_norm, y_past_norm)

        Anti-shortcut regularization:
            1) proxy floor: force decoder to use latent/proxy
            2) proxy/cov ratio floor: prevent severe cov shortcut
        """
        if not getattr(self.adapter, "can_reconstruct_past", True):
            logger.info("[pretrain_past_reconstruction_only] pipeline can't reconstruct -> skip.")
            return

        os.makedirs(debug_dir, exist_ok=True)

        def _to_np(x: torch.Tensor):
            return x.detach().float().cpu().numpy()

        def _print_stats(name: str, x: torch.Tensor):
            x_ = x.detach()
            logger.info(
                f"[dbg] {name:28s} shape={tuple(x_.shape)} "
                f"min={x_.min().item():.4g} mean={x_.mean().item():.4g} "
                f"max={x_.max().item():.4g} std={x_.std().item():.4g}"
            )

        def _patch_stats(x: torch.Tensor, patch: int):
            B, C, T = x.shape
            if patch <= 1 or (T % patch != 0):
                mu = x.mean(dim=-1, keepdim=True)   # (B,C,1)
                sd = x.std(dim=-1, keepdim=True)
                return mu, sd
            Np = T // patch
            x4 = x.view(B, C, Np, patch)
            mu = x4.mean(dim=-1)  # (B,C,Np)
            sd = x4.std(dim=-1)   # (B,C,Np)
            return mu, sd

        def _print_patch(name: str, x: torch.Tensor, patch: int, b: int = 0, c: int = 0, k: int = 6):
            mu, sd = _patch_stats(x, patch)
            mu_np = mu[b, c, :min(k, mu.shape[-1])].detach().cpu().numpy()
            sd_np = sd[b, c, :min(k, sd.shape[-1])].detach().cpu().numpy()
            logger.info(f"[dbg] {name:28s} patch_mean[:{k}]={mu_np}")
            logger.info(f"[dbg] {name:28s} patch_std [: {k}]={sd_np}")
            logger.info(
                f"[dbg] {name:28s} patch_std summary min/mean/max="
                f"{sd.min().item():.4g}/{sd.mean().item():.4g}/{sd.max().item():.4g}"
            )

        def _plot_1d(save_path: str, curves: dict, title: str):
            plt.figure(figsize=(14, 4))
            for k, v in curves.items():
                plt.plot(v, label=k)
            plt.legend()
            plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

        def _batch_probe_metrics(latents: torch.Tensor, x_p: torch.Tensor, y_recon_norm: torch.Tensor):
            """
            Compute decoder sensitivity to:
            - covariates shuffle
            - latent/proxy shuffle
            """
            with torch.no_grad():
                B = y_recon_norm.size(0)
                if B <= 1:
                    # 无法 shuffle 时返回 0，避免 nan
                    zero = torch.tensor(0.0, device=y_recon_norm.device)
                    return zero, zero

                perm = torch.randperm(B, device=y_recon_norm.device)

                y_recon_shuf_cov = self.adapter.decode(latents, x_p[perm])
                cov_sens = (y_recon_norm - y_recon_shuf_cov).abs().mean()

                y_recon_shuf_proxy = self.adapter.decode(latents[perm], x_p)
                proxy_sens = (y_recon_norm - y_recon_shuf_proxy).abs().mean()

            return cov_sens, proxy_sens

        train_dataset = TensorDataset(
            torch.tensor(past_target, dtype=torch.float32),
            torch.tensor(past_covariates, dtype=torch.float32),
        )
        
        dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = persistent_workers
            dl_kwargs["prefetch_factor"] = prefetch_factor

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **dl_kwargs,
        )

        val_loader = None
        if val_data is not None:
            assert "past_target" in val_data and "past_covariates" in val_data, \
                "val_data for pretrain_past_reconstruction_only must contain 'past_target' and 'past_covariates'"
            val_dataset = TensorDataset(
                torch.tensor(val_data["past_target"], dtype=torch.float32),
                torch.tensor(val_data["past_covariates"], dtype=torch.float32),
            )
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                **dl_kwargs,
            )

        self.adapter.train()
        optim = torch.optim.Adam(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        P_dbg = getattr(self.adapter, "revin_patch_size_past", 24)
        eps = 1e-6

        best_monitor = float("inf")
        best_epoch = -1
        best_state_dict = None
        bad_count = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": [],
            "train_proxy_sens": [],
            "train_cov_sens": [],
            "val_proxy_sens": [],
            "val_cov_sens": [],
            "train_proxy_cov_ratio": [],
            "val_proxy_cov_ratio": [],
            "train_monitor": [],
            "val_monitor": [],
            "train_loss_proxy_floor": [],
            "train_loss_ratio_floor": [],
            "train_loss_latent_scale": [],
            "val_loss_latent_scale": [],
            "train_loss_latent_std": [],
            "val_loss_latent_std": [],
            "train_latent_rms": [],
            "val_latent_rms": [],
        }

        def _run_one_epoch(loader, is_train: bool, epoch: int):
            if is_train:
                self.adapter.train()
            else:
                self.adapter.eval()

            total_loss = 0.0
            total_recon = 0.0
            total_proxy_floor = 0.0
            total_ratio_floor = 0.0
            total_latent_scale = 0.0
            total_latent_std = 0.0
            total_latent_rms = 0.0
            total_cov_sens = 0.0
            total_proxy_sens = 0.0
            total_ratio = 0.0
            count = 0
            dumped = False

            for (y_p, x_p) in loader:
                y_p = y_p.to(self.device)  # (B,1,L)
                x_p = x_p.to(self.device)  # (B,Cx,L)

                with torch.set_grad_enabled(is_train):
                    y_norm, _, _ = self.adapter.normalizer.norm_past(y_p, patch_size=P_dbg)
                    latents = self.adapter.encode(y_norm, x_p)
                    y_recon_norm = self.adapter.decode(latents, x_p)

                    recon_loss = F.mse_loss(y_recon_norm, y_norm)

                    # ----- latent scale penalty: only penalize "too large", not "too small" -----
                    latent_rms = torch.sqrt(torch.mean(latents ** 2) + 1e-8)
                    loss_latent_scale = F.relu(
                        latent_rms - torch.tensor(latent_rms_target, device=self.device, dtype=latents.dtype)
                    ) ** 2

                    cov_sens, proxy_sens = _batch_probe_metrics(latents, x_p, y_recon_norm)
                    proxy_cov_ratio = proxy_sens / (cov_sens + eps)

                    # anti-shortcut regularization
                    loss_proxy_floor = F.relu(
                        torch.tensor(proxy_sens_floor, device=self.device, dtype=y_recon_norm.dtype) - proxy_sens
                    )
                    loss_ratio_floor = F.relu(
                        torch.tensor(proxy_cov_ratio_floor, device=self.device, dtype=y_recon_norm.dtype) - proxy_cov_ratio
                    )
                    
                    # 防止latent坍缩为常数
                    latent_std = latents.std(dim=-1).mean()
                    loss_latent_std_floor = F.relu(
                        torch.tensor(latent_std_floor, device=self.device, dtype=latents.dtype) - latent_std
                    ) ** 2

                    if is_train:
                        loss = (
                            recon_loss
                            + lambda_proxy_floor * loss_proxy_floor
                            + lambda_ratio_floor * loss_ratio_floor
                            + lambda_latent_scale * loss_latent_scale
                            + lambda_latent_std_floor * loss_latent_std_floor
                        )
                    else:
                        # validation loss本体只看recon；scale/probe只做记录和monitor
                        loss = recon_loss

                    if is_train:
                        optim.zero_grad()
                        loss.backward()
                        optim.step()

                bsz = y_p.size(0)
                total_loss += float(loss.detach().item()) * bsz
                total_recon += float(recon_loss.detach().item()) * bsz
                total_proxy_floor += float(loss_proxy_floor.detach().item()) * bsz
                total_ratio_floor += float(loss_ratio_floor.detach().item()) * bsz
                total_latent_scale += float(loss_latent_scale.detach().item()) * bsz
                total_latent_rms += float(latent_rms.detach().item()) * bsz
                total_cov_sens += float(cov_sens.detach().item()) * bsz
                total_proxy_sens += float(proxy_sens.detach().item()) * bsz
                total_latent_std += float(loss_latent_std_floor.detach().item()) * bsz
                total_ratio += float(proxy_cov_ratio.detach().item()) * bsz
                count += bsz

                if is_train and debug and (not dumped) and (epoch % 10 == 0):
                    dumped = True
                    logger.info("\n=========== [Pretrain past recon ONLY] DEBUG ===========")
                    logger.info(f"[dbg] epoch={epoch}/{n_epochs-1} batch={y_p.size(0)}")

                    if debug_plot and epoch % 20 == 0:
                        out_dir = os.path.join(debug_dir, "pretrain_stage1")
                        os.makedirs(out_dir, exist_ok=True)
                        n_show = min(debug_num_seq, y_p.size(0))
                        for i in range(n_show):
                            _plot_1d(
                                os.path.join(out_dir, f"ep{epoch:03d}_past_recon_b{i}.png"),
                                {
                                    "y_past_norm": _to_np(y_norm[i, 0]),
                                    "y_recon_norm": _to_np(y_recon_norm[i, 0]),
                                },
                                title=f"Stage1 past recon(norm) | ep={epoch} b={i}",
                            )

                    _print_stats("y_past", y_p)
                    _print_stats("y_past_norm", y_norm)
                    _print_stats("latents(trend)", latents)
                    _print_stats("y_recon_norm", y_recon_norm)

                    _print_patch("y_past", y_p, P_dbg, b=0, c=0)
                    _print_patch("y_past_norm", y_norm, P_dbg, b=0, c=0)
                    _print_patch("latents(trend)", latents, P_dbg, b=0, c=0)
                    _print_patch("y_recon_norm", y_recon_norm, P_dbg, b=0, c=0)

                    logger.info(f"[probe] decoder sensitivity cov   : {float(cov_sens.detach().item()):.4f}")
                    logger.info(f"[probe] decoder sensitivity proxy : {float(proxy_sens.detach().item()):.4f}")
                    logger.info(f"[probe] proxy/cov ratio          : {float(proxy_cov_ratio.detach().item()):.4f}")
                    logger.info(f"[loss ] recon                    : {float(recon_loss.detach().item()):.6f}")
                    logger.info(f"[loss ] proxy_floor              : {float(loss_proxy_floor.detach().item()):.6f}")
                    logger.info(f"[loss ] ratio_floor              : {float(loss_ratio_floor.detach().item()):.6f}")
                    logger.info(f"[loss ] latent_scale             : {float(loss_latent_scale.detach().item()):.6f}")
                    logger.info(f"[loss ] latent_std               : {float(loss_latent_std_floor.detach().item()):.6f}")
                    logger.info(f"[stat ] latent_rms               : {float(latent_rms.detach().item()):.6f}")
                    logger.info("=========================================================\n")

            den = max(1, count)
            mean_loss = total_loss / den
            mean_recon = total_recon / den
            mean_proxy_floor = total_proxy_floor / den
            mean_ratio_floor = total_ratio_floor / den
            mean_latent_scale = total_latent_scale / den
            mean_latent_std = total_latent_std / den
            mean_latent_rms = total_latent_rms / den
            mean_cov_sens = total_cov_sens / den
            mean_proxy_sens = total_proxy_sens / den
            mean_ratio = total_ratio / den

            # early-stop / selection monitor:
            # reconstruction + penalty if proxy_sens is too small
            monitor = mean_recon + lambda_monitor * max(0.0, proxy_sens_floor - mean_proxy_sens)

            return dict(
                loss=mean_loss,
                recon_loss=mean_recon,
                loss_proxy_floor=mean_proxy_floor,
                loss_ratio_floor=mean_ratio_floor,
                loss_latent_scale=mean_latent_scale,
                loss_latent_std=mean_latent_std,
                latent_rms=mean_latent_rms,
                cov_sens=mean_cov_sens,
                proxy_sens=mean_proxy_sens,
                proxy_cov_ratio=mean_ratio,
                monitor=monitor,
            )

        for epoch in range(n_epochs):
            train_stats = _run_one_epoch(train_loader, is_train=True, epoch=epoch)

            if val_loader is not None:
                with torch.no_grad():
                    val_stats = _run_one_epoch(val_loader, is_train=False, epoch=epoch)
            else:
                val_stats = dict(train_stats)

            current_monitor = val_stats["monitor"] if val_loader is not None else train_stats["monitor"]

            history["epoch"].append(epoch)
            history["train_loss"].append(train_stats["loss"])
            history["val_loss"].append(val_stats["loss"])
            history["train_recon_loss"].append(train_stats["recon_loss"])
            history["val_recon_loss"].append(val_stats["recon_loss"])
            history["train_proxy_sens"].append(train_stats["proxy_sens"])
            history["train_cov_sens"].append(train_stats["cov_sens"])
            history["val_proxy_sens"].append(val_stats["proxy_sens"])
            history["val_cov_sens"].append(val_stats["cov_sens"])
            history["train_proxy_cov_ratio"].append(train_stats["proxy_cov_ratio"])
            history["val_proxy_cov_ratio"].append(val_stats["proxy_cov_ratio"])
            history["train_monitor"].append(train_stats["monitor"])
            history["val_monitor"].append(val_stats["monitor"])
            history["train_loss_proxy_floor"].append(train_stats["loss_proxy_floor"])
            history["train_loss_ratio_floor"].append(train_stats["loss_ratio_floor"])
            history["train_loss_latent_scale"].append(train_stats["loss_latent_scale"])
            history["val_loss_latent_scale"].append(val_stats["loss_latent_scale"])
            history["train_loss_latent_std"].append(train_stats["loss_latent_std"])
            history["val_loss_latent_std"].append(val_stats["loss_latent_std"])
            history["train_latent_rms"].append(train_stats["latent_rms"])
            history["val_latent_rms"].append(val_stats["latent_rms"])

            improved = (best_monitor - current_monitor) > min_delta
            if improved:
                best_monitor = current_monitor
                best_epoch = epoch
                bad_count = 0
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.adapter.state_dict().items()
                }
            else:
                if epoch >= min_n_epochs:
                    bad_count += 1

            if verbose and (epoch % 10 == 0):
                logger.info(
                    f"[stage1-pretrain] epoch={epoch:03d} "
                    f"train_loss={train_stats['loss']:.6f} "
                    f"train_recon={train_stats['recon_loss']:.6f} "
                    f"train_proxy={train_stats['proxy_sens']:.4f} "
                    f"train_cov={train_stats['cov_sens']:.4f} "
                    f"train_ratio={train_stats['proxy_cov_ratio']:.4f} "
                    f"train_latent_rms={train_stats['latent_rms']:.4f} | "
                    f"train_latent_std={train_stats['loss_latent_std']:.4f} | "
                    f"val_loss={val_stats['loss']:.6f} "
                    f"val_recon={val_stats['recon_loss']:.6f} "
                    f"val_proxy={val_stats['proxy_sens']:.4f} "
                    f"val_cov={val_stats['cov_sens']:.4f} "
                    f"val_ratio={val_stats['proxy_cov_ratio']:.4f} "
                    f"val_latent_rms={val_stats['latent_rms']:.4f} | "
                    f"val_latent_std={val_stats['loss_latent_std']:.4f} | "
                    f"monitor={current_monitor:.6f} "
                    f"best_monitor={best_monitor:.6f} "
                    f"(best_epoch={best_epoch}) "
                    f"bad_count={bad_count}/{patience}"
                )

            if epoch >= min_n_epochs and bad_count >= patience:
                if verbose:
                    logger.info(
                        f"[stage1-pretrain] early stop at epoch={epoch}, "
                        f"best_epoch={best_epoch}, best_monitor={best_monitor:.6f}"
                    )
                break

        if best_state_dict is not None:
            self.adapter.load_state_dict(best_state_dict)
            if verbose:
                logger.info(
                    f"[stage1-pretrain] restored best adapter weights "
                    f"from epoch={best_epoch} (best_monitor={best_monitor:.6f})"
                )
        else:
            if verbose:
                logger.info("[stage1-pretrain] WARNING: no best_state_dict saved.")

        try:
            hist_df = pd.DataFrame(history)
            hist_path = os.path.join(debug_dir, "pretrain_stage1_history.csv")
            hist_df.to_csv(hist_path, index=False)
            if verbose:
                logger.info(f"[stage1-pretrain] saved history to: {hist_path}")

            if len(hist_df) > 0:
                plt.figure(figsize=(12, 4))
                plt.plot(hist_df["epoch"], hist_df["train_recon_loss"], label="train_recon_loss")
                plt.plot(hist_df["epoch"], hist_df["val_recon_loss"], label="val_recon_loss")
                plt.plot(hist_df["epoch"], hist_df["train_monitor"], label="train_monitor")
                plt.plot(hist_df["epoch"], hist_df["val_monitor"], label="val_monitor")
                plt.legend()
                plt.title("Stage1 loss / monitor")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, "pretrain_stage1_loss_monitor.png"), dpi=150)
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.plot(hist_df["epoch"], hist_df["train_proxy_sens"], label="train_proxy_sens")
                plt.plot(hist_df["epoch"], hist_df["train_cov_sens"], label="train_cov_sens")
                plt.plot(hist_df["epoch"], hist_df["val_proxy_sens"], label="val_proxy_sens")
                plt.plot(hist_df["epoch"], hist_df["val_cov_sens"], label="val_cov_sens")
                plt.legend()
                plt.title("Stage1 sensitivity")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, "pretrain_stage1_sensitivity.png"), dpi=150)
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.plot(hist_df["epoch"], hist_df["train_proxy_cov_ratio"], label="train_proxy_cov_ratio")
                plt.plot(hist_df["epoch"], hist_df["val_proxy_cov_ratio"], label="val_proxy_cov_ratio")
                plt.axhline(proxy_cov_ratio_floor, linestyle="--", label="ratio_floor")
                plt.legend()
                plt.title("Stage1 proxy/cov ratio")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, "pretrain_stage1_ratio.png"), dpi=150)
                plt.close()

                plt.figure(figsize=(12, 4))
                plt.plot(hist_df["epoch"], hist_df["train_latent_rms"], label="train_latent_rms")
                plt.plot(hist_df["epoch"], hist_df["val_latent_rms"], label="val_latent_rms")
                plt.axhline(latent_rms_target, linestyle="--", label="latent_rms_target")
                plt.legend()
                plt.title("Stage1 latent RMS")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, "pretrain_stage1_latent_rms.png"), dpi=150)
                plt.close()
        except Exception as e:
            logger.info(f"[stage1-pretrain] save history/plots failed: {e}")

    def predict(
        self,
        past_target: np.ndarray,
        past_covariates: np.ndarray,
        future_covariates: np.ndarray,
        pred_horizon: int,
        ltm_batch_size: int = 128,
        n_samples: int = 20,
        **ltm_kwargs,
    ) -> PredPack:
        self.adapter.train()
        self.iclearner.eval()

        past_target_tensor = torch.tensor(past_target, dtype=torch.float32, device=self.device)
        past_covariates_tensor = torch.tensor(past_covariates, dtype=torch.float32, device=self.device)
        future_covariates_tensor = torch.tensor(future_covariates, dtype=torch.float32, device=self.device)

        all_future_target_samples = []

        P_past = self.adapter.revin_patch_size_past
        P_fut = self.adapter.revin_patch_size_future

        for sample_idx in range(n_samples):
            with torch.no_grad():
                past_target_norm, _, _ = self.adapter.normalizer.norm_past(past_target_tensor, patch_size=P_past)

                past_latents = self.adapter.encode(past_target_norm, past_covariates_tensor)

                future_latents_pred = self._predict_future_latents_with_ltm(
                    past_latents=past_latents,
                    future_horizon=pred_horizon,
                    ltm_batch_size=ltm_batch_size,
                    **ltm_kwargs,
                )

                future_target_pred_norm = self.adapter.decode(future_latents_pred, future_covariates_tensor)

                mu_t_hat, std_t_hat, _, _, _ = self.adapter.predict_future_stats(future_covariates_tensor)
                # IdentityNormalizer will ignore these effectively
                mu_ref, std_ref = self.adapter.normalizer.mix_future_time_stats(
                    mu_t_true=None, std_t_true=None,
                    mu_t_hat=mu_t_hat, std_t_hat=std_t_hat,
                    alpha_stats=1.0,
                )
                future_target_pred = self.adapter.normalizer.denorm_future_pred(
                    y_future_pred_norm=future_target_pred_norm,
                    mu_ref=mu_ref.detach(),
                    std_ref=std_ref.detach(),
                )

                all_future_target_samples.append(future_target_pred.detach().cpu().numpy())

                if sample_idx == 0:
                    logger.info("------------------------predict----------------------")
                    logger.info("past_latents min/mean/max:", past_latents.min().item(), past_latents.mean().item(), past_latents.max().item())
                    logger.info("past_latents std:", past_latents.std().item())
                    logger.info("future_cov min/mean/max:", future_covariates_tensor.min().item(), future_covariates_tensor.mean().item(), future_covariates_tensor.max().item())
                    logger.info("future_target_pred_norm std:", future_target_pred_norm.std().item())

                    if self.adapter.has_stats_predictor:
                        mu_patch_hat, std_patch_hat, logstd_patch_hat = self.adapter.future_stats_predictor(future_covariates_tensor)
                        logger.info("mu_patch_hat min/mean/max:", mu_patch_hat.min().item(), mu_patch_hat.mean().item(), mu_patch_hat.max().item())
                        logger.info("std_patch_hat min/mean/max:", std_patch_hat.min().item(), std_patch_hat.mean().item(), std_patch_hat.max().item())

        all_future_target_samples = np.stack(all_future_target_samples, axis=0)  # (S,B,1,H)
        mean = all_future_target_samples.mean(axis=0)
        std = all_future_target_samples.std(axis=0)
        lb = mean - std
        ub = mean + std
        return PredPack(mean=mean, lb=lb, ub=ub)
