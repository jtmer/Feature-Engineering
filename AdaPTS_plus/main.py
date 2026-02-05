# main.py
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ====== repo path ======
adapts_repo_root = "/data/mahaoke/AdaPTS"
out_root = Path("./results_monthly_backtest")
import sys
sys.path.insert(0, adapts_repo_root)

from sundial.iclearner_sundial import SundialICLTrainer
from conditional_adapters import NeuralTimeAdapter
from conditional_adapts import ConditionalAdaPTS
import conditional_adapts

# =========================
# Utilities
# =========================

def _get_logger(name: str = "run", log_path: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    _logger.propagate = False  # 防止重复打印

    # 避免重复添加 handler
    if _logger.handlers:
        return _logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    _logger.addHandler(ch)

    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)

    return _logger

class SimpleLogger:
    def __init__(self, name: str = "run", log_path: Optional[str] = None, level=logging.INFO):
        self._logger = _get_logger(name, log_path, level)

    def info(self, *args):
        msg = " ".join(str(x) for x in args)
        self._logger.info(msg)

    def debug(self, *args):
        msg = " ".join(str(x) for x in args)
        self._logger.debug(msg)

    def warning(self, *args):
        msg = " ".join(str(x) for x in args)
        self._logger.warning(msg)

    def error(self, *args):
        msg = " ".join(str(x) for x in args)
        self._logger.error(msg)

    @property
    def raw(self) -> logging.Logger:
        return self._logger

logger = SimpleLogger(
    "backtest",
    log_path=str(out_root / "train.log")
)

conditional_adapts.logger = logger

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_basic_stats(name: str, x: np.ndarray):
    x_ = np.asarray(x)
    logger.info(f"{name:24s} shape={x_.shape} min/mean/max={x_.min():.4g}/{x_.mean():.4g}/{x_.max():.4g}")


def scale_xy(
    x_cov: np.ndarray,
    y: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    clip: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    x_cov: (T, Cx)
    y:     (T,)
    """
    x = scaler_X.transform(x_cov)
    x = np.clip(x, -clip, clip)
    yy = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)
    return x.astype(np.float32), yy.astype(np.float32)

def inverse_y(scaler_y: StandardScaler, y_scaled: np.ndarray) -> np.ndarray:
    flat = y_scaled.reshape(-1, 1)
    inv = scaler_y.inverse_transform(flat).reshape(y_scaled.shape)
    return inv


def make_windows(
    x_cov: np.ndarray, y: np.ndarray,
    L: int, H: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    x_cov: (T, Cx)
    y:     (T,)
    return:
      y_past:   (N,1,L)
      x_past:   (N,Cx,L)
      x_future: (N,Cx,H)
      y_future: (N,1,H)
    """
    T, Cx = x_cov.shape
    max_i = T - L - H
    if max_i < 0:
        # 数据长度不足以切一个窗口
        return (
            np.zeros((0, 1, L), dtype=np.float32),
            np.zeros((0, Cx, L), dtype=np.float32),
            np.zeros((0, Cx, H), dtype=np.float32),
            np.zeros((0, 1, H), dtype=np.float32),
        )

    ys_p, xs_p, xs_f, ys_f = [], [], [], []
    for i in range(0, max_i + 1, stride):
        x_p = x_cov[i:i + L].T              # (Cx,L)
        y_p = y[i:i + L][None, :]           # (1,L)
        x_f = x_cov[i + L:i + L + H].T      # (Cx,H)
        y_f = y[i + L:i + L + H][None, :]   # (1,H)
        xs_p.append(x_p); ys_p.append(y_p); xs_f.append(x_f); ys_f.append(y_f)

    return (
        np.stack(ys_p).astype(np.float32),
        np.stack(xs_p).astype(np.float32),
        np.stack(xs_f).astype(np.float32),
        np.stack(ys_f).astype(np.float32),
    )


def rmse_per_window(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def mae_per_window(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=1)

def imse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_true/y_pred: (B,H) inverse scale
    return float(np.mean((y_true - y_pred) ** 2))

def imae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def imape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.abs(y_true) + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def imape_ad(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    # adjusted (sMAPE-style)
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


# =========================
# Config
# =========================

@dataclass
class WindowConfig:
    hist_window: int = 2880
    pred_window: int = 96
    stride_train: int = 96
    stride_eval: int = 96


@dataclass
class TrainConfig:
    seed: int = 13
    model_name: str = "thuml/sundial-base-128m"
    z_dim: int = 1
    patch_size: int = 16

    # stats pretrain
    stats_pretrain_epochs: int = 80
    stats_pretrain_bs: int = 64
    stats_pretrain_lr: float = 1e-4
    stats_pretrain_wd: float = 1e-4
    stats_pretrain_patience: int = 8

    # past recon pretrain
    past_pretrain_epochs: int = 20
    past_pretrain_bs: int = 32
    past_pretrain_lr: float = 1e-4
    past_pretrain_wd: float = 1e-4

    # adapter train
    train_epochs: int = 60
    train_bs: int = 32
    train_lr: float = 1e-4
    train_wd: float = 1e-4
    lambda_past_recon: float = 0.4
    lambda_future_pred: float = 2.0
    lambda_latent_stats: float = 0.0
    lambda_stats_pred: float = 0.0
    lambda_y_patch_std: float = 2.0

    ltm_batch_size_train: int = 1
    ltm_batch_size_pred: int = 32

    # predict sampling
    n_samples: int = 30

    # vis
    max_vis_windows: int = 6  # 每个月最多画几条


# =========================
# Date-based split
# =========================

def month_starts(start_ym: str, end_ym: str) -> List[pd.Timestamp]:
    start = pd.to_datetime(start_ym + "-01")
    end   = pd.to_datetime(end_ym + "-01")
    return list(pd.date_range(start=start, end=end, freq="MS"))

def get_test_with_lookahead_points(
    df: pd.DataFrame,
    time_col: str,
    test_start: pd.Timestamp,
    test_months: int,
    need_points: int,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    返回：
      d_test_ext: 从 test_start 开始，至少 need_points 个点（不足则向后补到够）
      test_end_raw: 当月结束（不含补齐部分）
    """
    test_end_raw = test_start + pd.DateOffset(months=test_months)

    d_month = df[(df[time_col] >= test_start) & (df[time_col] < test_end_raw)].copy()

    if len(d_month) >= need_points:
        return d_month, test_end_raw

    # 需要补齐
    need_more = need_points - len(d_month)
    d_more = df[df[time_col] >= test_end_raw].copy().head(need_more)

    d_test_ext = pd.concat([d_month, d_more], axis=0)
    return d_test_ext, test_end_raw

def split_train_test_by_month(
    df: pd.DataFrame,
    time_col: str,
    test_month_start: pd.Timestamp,
    train_years: int = 2,
    test_months: int = 1,
    train_ratio: float = 0.7,   # train/val 比例
) -> Dict[str, pd.DataFrame]:
    """
    Train_all: [test_start - train_years, test_start)
    Test     : [test_start, test_start + test_months)

    在 Train_all 内部按时间顺序切：
      Train = 前 train_ratio
      Val   = 后 (1-train_ratio)
    """
    assert 0.0 < train_ratio < 1.0, "train_ratio must be in (0,1)"

    test_start = test_month_start
    test_end = test_start + pd.DateOffset(months=test_months)
    train_start = test_start - pd.DateOffset(years=train_years)
    train_end = test_start

    d_train_all = df[(df[time_col] >= train_start) & (df[time_col] < train_end)].copy()
    d_test = df[(df[time_col] >= test_start) & (df[time_col] < test_end)].copy()

    # 过去两年不足时，直接返回空
    if len(d_train_all) < 2:
        return dict(
            train=d_train_all.iloc[:0].copy(),
            val=d_train_all.iloc[:0].copy(),
            train_all=d_train_all,
            test=d_test,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            val_start=None,
            val_end=None,
        )

    # 按时间顺序切分
    d_train_all = d_train_all.sort_values(time_col).reset_index(drop=True)
    n_all = len(d_train_all)
    n_train = int(np.floor(n_all * train_ratio))
    n_train = max(1, min(n_train, n_all - 1))  # 至少给 val 留 1 条

    d_train = d_train_all.iloc[:n_train].copy()
    d_val   = d_train_all.iloc[n_train:].copy()

    val_start = d_val[time_col].iloc[0]
    val_end   = d_val[time_col].iloc[-1]

    return dict(
        train=d_train,
        val=d_val,
        train_all=d_train_all,
        test=d_test,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        val_start=val_start,
        val_end=val_end,
    )


# =========================
# Data load & preprocess
# =========================

def load_dataset(path: str, time_col: str = "time") -> pd.DataFrame:
    df = pd.read_csv(path)
    # drop all-nan columns
    df = df.dropna(axis=1, how="any")

    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df


def build_xy_from_df(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    返回:
      X_all: (T, Cx)
      y_all: (T,)
      cov_cols: covariate columns used
    """
    drop_cols = drop_cols or []
    use_cols = [c for c in df.columns if c not in drop_cols]

    if target_col not in use_cols:
        raise ValueError(f"target_col={target_col} not found after drop_cols")

    cov_cols = [c for c in use_cols if c != target_col]
    X_all = df[cov_cols].values.astype(np.float32)
    y_all = df[target_col].values.astype(np.float32)
    return X_all, y_all, cov_cols


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    return scaler_X, scaler_y


# =========================
# Model build
# =========================

def build_iclearner(model_name: str, z_dim: int, pred_window: int, device: str):
    return SundialICLTrainer(
        sundial_name=model_name,
        n_features=z_dim,
        forecast_horizon=pred_window,
        device=device,
        trust_remote_code=True,
    )


def build_adapter(cov_dim: int, z_dim: int, patch_size: int):
    return NeuralTimeAdapter(
        covariates_dim=cov_dim,
        latent_dim=z_dim,
        revin_patch_size_past=patch_size,
        revin_patch_size_future=patch_size,
        hidden_dim=256,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.0,
        stats_hidden_dim=256,
        normalize_latents=True,
    )


# =========================
# Evaluation / Visualization
# =========================

def predict_pack_to_arrays(pack) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = pack.mean[:, 0, :]
    lb = pack.lb[:, 0, :]
    ub = pack.ub[:, 0, :]
    return mean, lb, ub


def predict_and_metrics_one_month(
    model: ConditionalAdaPTS,
    scaler_y: StandardScaler,
    y_p_te: np.ndarray,
    x_p_te: np.ndarray,
    x_f_te: np.ndarray,
    y_f_te: np.ndarray,
    hist_window: int,
    pred_window: int,
    ltm_batch_size: int,
    n_samples: int,
    vis_dir: Path,
    target_name: str,
    patch_size: int,
    z_dim: int,
    use_patch_revin: bool = False,
    max_vis_windows: int = 6,
) -> Dict[str, float]:
    """
    在 test month windows 上预测，输出 RMSE/MAE/coverage，并保存少量可视化图。
    """
    B = y_p_te.shape[0]
    if B == 0:
        return dict(rmse=np.nan, mae=np.nan, coverage=np.nan, n_windows=0)
    
    pack = model.predict(
        past_target=y_p_te,
        past_covariates=x_p_te,
        future_covariates=x_f_te,
        pred_horizon=pred_window,
        ltm_batch_size=ltm_batch_size,
        n_samples=n_samples,
        use_patch_revin=use_patch_revin,
    )
    mean_y, lb_y, ub_y = predict_pack_to_arrays(pack)
    true_y = y_f_te[:, 0, :]

    # inverse scale
    mean_y_inv = inverse_y(scaler_y, mean_y)
    true_y_inv = inverse_y(scaler_y, true_y)
    lb_y_inv = inverse_y(scaler_y, lb_y)
    ub_y_inv = inverse_y(scaler_y, ub_y)

    # history inverse for plotting
    past_y_inv = inverse_y(scaler_y, y_p_te[:, 0, :])

    # metrics
    # rmses = rmse_per_window(true_y_inv, mean_y_inv)
    # maes = mae_per_window(true_y_inv, mean_y_inv)
    # covered = (true_y_inv >= lb_y_inv) & (true_y_inv <= ub_y_inv)
    # coverage = float(covered.mean())

    # rmse_mean = float(np.mean(rmses))
    # mae_mean = float(np.mean(maes))
    iMSE = imse(true_y_inv, mean_y_inv)
    iMAE = imae(true_y_inv, mean_y_inv)
    iMAPE = imape(true_y_inv, mean_y_inv)
    iMAPE_ad = imape_ad(true_y_inv, mean_y_inv)
    
    eps = 1e-6
    iMSEs = np.mean((true_y_inv - mean_y_inv) ** 2, axis=1)  # (B,)
    iMAEs = np.mean(np.abs(true_y_inv - mean_y_inv), axis=1)  # (B,)
    iMAPEs = np.mean(np.abs(true_y_inv - mean_y_inv) / (np.abs(true_y_inv) + eps), axis=1) * 100.0
    iMAPE_ads = np.mean(
        2.0 * np.abs(true_y_inv - mean_y_inv) / (np.abs(true_y_inv) + np.abs(mean_y_inv) + eps),
        axis=1
    ) * 100.0

    covered = (true_y_inv >= lb_y_inv) & (true_y_inv <= ub_y_inv)
    coverage = float(covered.mean())

    # plots
    vis_dir.mkdir(parents=True, exist_ok=True)
    L = hist_window
    H = pred_window
    t_past = np.arange(L)
    t_future = np.arange(L, L + H)

    n_show = min(max_vis_windows, B)
    for idx in range(n_show):
        plt.figure(figsize=(14, 4))
        plt.plot(t_past, past_y_inv[idx], label="History (true)")
        plt.plot(t_future, true_y_inv[idx], label="Future (true)")
        plt.plot(t_future, mean_y_inv[idx], label="Forecast (mean)")
        plt.fill_between(t_future, lb_y_inv[idx], ub_y_inv[idx], alpha=0.2, label="Uncertainty (lb~ub)")
        plt.axvline(L - 1, linestyle="--")
        plt.title(f"ConditionalAdaPTS+Sundial (idx={idx}) | patch={patch_size} | z_dim={z_dim}")
        plt.xlabel("Time index within window")
        plt.ylabel(target_name)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(vis_dir / f"forecast_window_{idx:03d}.png", dpi=150)
        plt.close()

    # error plots
    plt.figure(figsize=(12, 3))
    plt.plot(iMSEs, label="iMSE per window")
    plt.plot(iMAEs, label="iMAE per window")
    plt.plot(iMAPEs, label="iMAPE per window (%)")
    plt.plot(iMAPE_ads, label="iMAPE-ad per window (%)")
    plt.title(f"Error across test windows | patch={patch_size} | z_dim={z_dim}")
    plt.xlabel("Test window index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "error_curve.png", dpi=150)
    plt.close()

    def safe_hist(ax, x, bins=20, label=None, alpha=0.5):
        x = np.asarray(x).reshape(-1)
        x = x[np.isfinite(x)]
        # 至少要 2 个点，且有非零范围，才画直方图
        if x.size < 2:
            ax.text(0.5, 0.8, f"{label}: <2 finite points", ha="center", transform=ax.transAxes)
            return
        xmin, xmax = float(x.min()), float(x.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            ax.text(0.5, 0.8, f"{label}: zero/invalid range", ha="center", transform=ax.transAxes)
            return

        # bins 不超过样本数-1，避免 weird edge case
        b = int(min(bins, max(1, x.size - 1)))
        ax.hist(x, bins=b, alpha=alpha, label=label)

    fig, ax = plt.subplots(figsize=(12, 3))
    safe_hist(ax, iMSEs, bins=20, label="iMSE")
    safe_hist(ax, iMAEs, bins=20, label="iMAE")
    safe_hist(ax, iMAPEs, bins=20, label="iMAPE (%)")
    safe_hist(ax, iMAPE_ads, bins=20, label="iMAPE-ad (%)")
    ax.set_title(f"Error distribution | patch={patch_size} | z_dim={z_dim}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vis_dir / "error_hist.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(6, 3))
    plt.bar(["coverage"], [coverage])
    plt.ylim(0, 1)
    plt.title("Interval coverage (lb~ub)")
    plt.tight_layout()
    plt.savefig(vis_dir / "interval_coverage.png", dpi=150)
    plt.close()

    # return dict(
    #     rmse=rmse_mean,
    #     mae=mae_mean,
    #     coverage=coverage,
    #     n_windows=int(B),
    # )
    return dict(
        iMSE=iMSE,
        iMAE=iMAE,
        iMAPE=iMAPE,
        iMAPE_ad=iMAPE_ad,
        coverage=coverage,
        n_windows=int(B),
    )


# =========================
# Main rolling backtest
# =========================

def main():
    # ---- configs ----
    wcfg = WindowConfig(hist_window=2880, pred_window=96, stride_train=96, stride_eval=96)
    cfg = TrainConfig()
    cfg.patch_size = 16
    cfg.z_dim = 1
    
    # ---- seed & device ----
    device = get_device()
    set_seed(cfg.seed)
    logger.info("device:", device)

    # ---- load data ----
    path = "/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv"
    time_col = "time"
    target_col = "day_ahead_clearing_price"

    df = load_dataset(path, time_col=time_col)

    drop_cols = [time_col, "realtime_clearing_price"]
    X_all_raw, y_all_raw, cov_cols = build_xy_from_df(df, target_col=target_col, drop_cols=drop_cols)
    cov_dim = X_all_raw.shape[1]

    assert wcfg.hist_window % cfg.patch_size == 0 and wcfg.pred_window % cfg.patch_size == 0, \
        "L/H must be divisible by patch_size"

    test_months = month_starts("2024-08", "2025-07")
    logger.info("test_months:", [d.strftime("%Y-%m-%d") for d in test_months], "len=", len(test_months))

    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    for m0 in test_months:
        split = split_train_test_by_month(
            df=df,
            time_col=time_col,
            test_month_start=m0,
            train_years=2,
            test_months=1,
        )

        d_train = split["train"]
        d_val = split["val"]
        # d_test = split["test"]
        L, H = wcfg.hist_window, wcfg.pred_window
        need_points = L + H
        # split 里仍然负责 train/val 的时间范围；test 我们自己按点数补齐
        d_test_ext, test_end_raw = get_test_with_lookahead_points(
            df=df,
            time_col=time_col,
            test_start=m0,
            test_months=1,
            need_points=need_points,
        )
        d_test = d_test_ext

        month_tag = m0.strftime("%Y-%m")
        logger.info("\n====================================================")
        logger.info(f"[Backtest month] {month_tag}")
        logger.info(f"train: {split['train_start'].date()} -> {split['val_start'].date()}  (len={len(d_train)})")
        logger.info(f"val  : {split['val_start'].date()} -> {split['val_end'].date()}    (len={len(d_val)})")
        logger.info(f"test : {split['test_start'].date()} -> {split['test_end'].date()}  (len={len(d_test)})")
        logger.info(f"test(raw month) {m0.date()} -> {test_end_raw.date()}  raw_len={len(df[(df[time_col]>=m0)&(df[time_col]<test_end_raw)])}  ext_len={len(d_test)} need={need_points}")

        # 基本检查：训练/测试数据够不够切窗口
        if len(d_train) < (wcfg.hist_window + wcfg.pred_window) or len(d_test) < (wcfg.hist_window + wcfg.pred_window):
            logger.info(f"[skip] Not enough samples for windows in month {month_tag}")
            results.append(dict(month=month_tag, rmse=np.nan, mae=np.nan, coverage=np.nan, n_windows=0))
            continue

        # ---- build raw arrays for each split ----
        X_tr_raw, y_tr_raw, _ = build_xy_from_df(d_train, target_col=target_col, drop_cols=drop_cols)
        X_va_raw, y_va_raw, _ = build_xy_from_df(d_val, target_col=target_col, drop_cols=drop_cols)
        X_te_raw, y_te_raw, _ = build_xy_from_df(d_test, target_col=target_col, drop_cols=drop_cols)

        # ---- fit scalers ONLY on train ----
        scaler_X, scaler_y = fit_scalers(X_tr_raw, y_tr_raw)

        # ---- scale ----
        X_tr_s, y_tr_s = scale_xy(X_tr_raw, y_tr_raw, scaler_X, scaler_y)
        X_va_s, y_va_s = scale_xy(X_va_raw, y_va_raw, scaler_X, scaler_y)
        X_te_s, y_te_s = scale_xy(X_te_raw, y_te_raw, scaler_X, scaler_y)

        # ---- window ----
        y_p_tr, x_p_tr, x_f_tr, y_f_tr = make_windows(X_tr_s, y_tr_s, wcfg.hist_window, wcfg.pred_window, wcfg.stride_train)
        y_p_va, x_p_va, x_f_va, y_f_va = make_windows(X_va_s, y_va_s, wcfg.hist_window, wcfg.pred_window, wcfg.stride_eval)
        y_p_te, x_p_te, x_f_te, y_f_te = make_windows(X_te_s, y_te_s, wcfg.hist_window, wcfg.pred_window, wcfg.stride_eval)

        logger.info(f"[windows] train={y_p_tr.shape[0]}  val={y_p_va.shape[0]}  test={y_p_te.shape[0]}")
        if y_p_tr.shape[0] == 0 or y_p_te.shape[0] == 0:
            logger.info(f"[skip] Windows empty in month {month_tag}")
            results.append(dict(month=month_tag, rmse=np.nan, mae=np.nan, coverage=np.nan, n_windows=0))
            continue

        # ---- build model for this month ----
        iclearner = build_iclearner(cfg.model_name, cfg.z_dim, wcfg.pred_window, device=device)
        adapter = build_adapter(cov_dim=cov_dim, z_dim=cfg.z_dim, patch_size=cfg.patch_size)
        model2 = ConditionalAdaPTS(adapter=adapter, iclearner=iclearner, device=device)

        val_data = dict(
            past_target=y_p_va,
            past_covariates=x_p_va,
            future_target=y_f_va,
            future_covariates=x_f_va,
        )
        stats_val_data = dict(
            future_target=y_f_va,
            future_covariates=x_f_va,
        )

        # ---- pretrain stats predictor ----
        logger.info(">>> Pretrain future_stats_predictor ...")
        model2.pretrain_stats_predictor(
            future_target=y_f_tr,
            future_covariates=x_f_tr,
            n_epochs=cfg.stats_pretrain_epochs,
            batch_size=cfg.stats_pretrain_bs,
            lr=cfg.stats_pretrain_lr,
            weight_decay=cfg.stats_pretrain_wd,
            patience=cfg.stats_pretrain_patience,
            val_data=stats_val_data,
            verbose=True,
            use_swanlab=False,
        )
        logger.info(">>> Done pretraining stats predictor.")

        # ---- pretrain past reconstruction ----
        logger.info(">>> Pretrain past reconstruction only ...")
        model2.pretrain_past_reconstruction_only(
            past_target=y_p_tr,
            past_covariates=x_p_tr,
            n_epochs=cfg.past_pretrain_epochs,
            batch_size=cfg.past_pretrain_bs,
            lr=cfg.past_pretrain_lr,
            weight_decay=cfg.past_pretrain_wd,
            debug=True,
            debug_dir=str(out_root / f"debug_no_revin_residual_{month_tag}"),
            debug_plot=True,
            verbose=True,
        )
        logger.info(">>> Done pretraining past reconstruction.")

        # ---- train adapter ----
        logger.info(">>> Train adapter ...")
        model2.train_adapter(
            past_target=y_p_tr,
            past_covariates=x_p_tr,
            future_target=y_f_tr,
            future_covariates=x_f_tr,
            n_epochs=cfg.train_epochs,
            batch_size=cfg.train_bs,
            lr=cfg.train_lr,
            weight_decay=cfg.train_wd,
            lambda_past_recon=cfg.lambda_past_recon,
            lambda_future_pred=cfg.lambda_future_pred,
            lambda_latent_stats=cfg.lambda_latent_stats,
            lambda_stats_pred=cfg.lambda_stats_pred,
            lambda_y_patch_std=cfg.lambda_y_patch_std,
            ltm_batch_size=cfg.ltm_batch_size_train,
            val_data=val_data,
            verbose=True,
            use_swanlab=False,
            debug=True,
            debug_dir=str(out_root / f"debug_train_adapter_{month_tag}"),
            debug_every=2,
            debug_num_seq=2,
            debug_latent_ch=1,
            earlystop_patience=10,
        )
        logger.info(">>> Done training adapter.")

        # ---- test metrics + plots ----
        vis_dir = out_root / f"vis_{month_tag}"
        metrics = predict_and_metrics_one_month(
            model=model2,
            scaler_y=scaler_y,
            y_p_te=y_p_te,
            x_p_te=x_p_te,
            x_f_te=x_f_te,
            y_f_te=y_f_te,
            hist_window=wcfg.hist_window,
            pred_window=wcfg.pred_window,
            ltm_batch_size=cfg.ltm_batch_size_pred,
            n_samples=cfg.n_samples,
            vis_dir=vis_dir,
            target_name=target_col,
            patch_size=cfg.patch_size,
            z_dim=cfg.z_dim,
            max_vis_windows=cfg.max_vis_windows,
        )

        logger.info(f"[Test {month_tag}] iMSE={metrics['iMSE']:.4f}  iMAE={metrics['iMAE']:.4f} iMAPE={metrics['iMAPE']:.2f}% iMAPE-adj={metrics['iMAPE_ad']:.2f}% coverage={metrics['coverage']:.3f}  n_windows={metrics['n_windows']}")
        results.append(dict(month=month_tag, **metrics))

    # ---- save summary ----
    res_df = pd.DataFrame(results)
    res_path = out_root / "results_summary.csv"
    res_df.to_csv(res_path, index=False)
    logger.info("\n==================== SUMMARY ====================")
    logger.info(res_df)
    logger.info(f"[Saved] {res_path}")
    logger.info("=================================================\n")


if __name__ == "__main__":
    main()
