# main.py
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
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
from conditional_adapters import NeuralTimeAdapter, build_pipeline_adapter
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

def make_time_windows(
    t: np.ndarray,
    L: int, H: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    t: (T,) datetime64 / pandas Timestamp array
    return:
      t_past:   (N,L)
      t_future: (N,H)
    """
    t = np.asarray(t)
    T = t.shape[0]
    max_i = T - L - H
    if max_i < 0:
        return (
            np.zeros((0, L), dtype="datetime64[ns]"),
            np.zeros((0, H), dtype="datetime64[ns]"),
        )

    ts_p, ts_f = [], []
    for i in range(0, max_i + 1, stride):
        t_p = t[i:i + L]
        t_f = t[i + L:i + L + H]
        ts_p.append(t_p)
        ts_f.append(t_f)

    return np.stack(ts_p), np.stack(ts_f)

def make_daily_anchored_test_windows(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    cov_cols: List[str],
    drop_cols: List[str],
    month_start: pd.Timestamp,     # e.g. 2024-08-01
    month_end: pd.Timestamp,       # e.g. 2024-09-01 (exclusive)
    hist_window: int,              # L
    pred_window: int,              # H (96)
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    freq: int = 15,             # 15分钟一个点
    target_hour: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对测试月内每一天做 0 点起报的日预测窗口：
      输入：该日 0 点之前 hist_window 点 (past)
      预测：该日 0 点起 pred_window 点 (future)

    返回:
      y_p_te: (B,1,L)
      x_p_te: (B,Cx,L)
      x_f_te: (B,Cx,H)
      y_f_te: (B,1,H)
      t_f_te: (B,H)  用于 workday/anchor 筛选与指标
    """
    dff = df[[time_col, target_col] + cov_cols].copy()
    dff = dff.sort_values(time_col).reset_index(drop=True)

    # 用时间做索引，便于按 timestamp 切片
    dff = dff.set_index(time_col)

    # 测试月内每天的起报锚点（自然日 00:00）
    anchors = pd.date_range(start=month_start, end=month_end, freq="D", inclusive="left")
    anchors = [a for a in anchors if a.hour == target_hour]

    ys_p, xs_p, xs_f, ys_f, ts_f = [], [], [], [], []

    for a in anchors:
        # future: [a, a+H)
        fut = dff.loc[a: a + pd.Timedelta(minutes=freq * (pred_window - 1))]

        if len(fut) != pred_window:
            continue

        # past: 取锚点之前 hist_window 个点
        past = dff.loc[:a].iloc[:-1]  # strictly before a
        if len(past) < hist_window:
            continue
        past = past.iloc[-hist_window:]

        # --- 转成 raw arrays ---
        X_p_raw = past[cov_cols].to_numpy(np.float32)          # (L,Cx)
        y_p_raw = past[target_col].to_numpy(np.float32)        # (L,)
        X_f_raw = fut[cov_cols].to_numpy(np.float32)           # (H,Cx)
        y_f_raw = fut[target_col].to_numpy(np.float32)         # (H,)
        t_f_raw = fut.index.to_numpy()                         # (H,)

        # --- scale ---
        X_p_s, y_p_s = scale_xy(X_p_raw, y_p_raw, scaler_X, scaler_y)
        X_f_s, y_f_s = scale_xy(X_f_raw, y_f_raw, scaler_X, scaler_y)

        # --- reshape to model format ---
        xs_p.append(X_p_s.T)                 # (Cx,L)
        ys_p.append(y_p_s[None, :])          # (1,L)
        xs_f.append(X_f_s.T)                 # (Cx,H)
        ys_f.append(y_f_s[None, :])          # (1,H)
        ts_f.append(t_f_raw)                 # (H,)

    if len(ys_p) == 0:
        cov_dim = len(cov_cols)
        return (
            np.zeros((0, 1, hist_window), dtype=np.float32),
            np.zeros((0, cov_dim, hist_window), dtype=np.float32),
            np.zeros((0, cov_dim, pred_window), dtype=np.float32),
            np.zeros((0, 1, pred_window), dtype=np.float32),
            np.zeros((0, pred_window), dtype="datetime64[ns]"),
        )

    return (
        np.stack(ys_p).astype(np.float32),
        np.stack(xs_p).astype(np.float32),
        np.stack(xs_f).astype(np.float32),
        np.stack(ys_f).astype(np.float32),
        np.stack(ts_f),
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

'''
def _imse_imae_workday(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_true: str,
    y_pred: str,
    target_hour: int,
) -> tuple[float, float]:
    if y_pred not in df.columns:
        return float("nan"), float("nan")

    dff = df[[time_col, y_true, y_pred]].copy()
    dff["_d"] = dff[time_col].dt.date

    n_days_total = 0
    n_days_full96 = 0
    n_days_anchor_ok = 0
    n_days_workday = 0
    n_days_used = 0

    mses, maes = [], []
    for _, g in dff.groupby("_d"):
        n_days_total += 1
        g = g.sort_values(time_col)

        if len(g) != POINTS_PER_DAY:
            continue
        n_days_full96 += 1

        if g[time_col].iloc[0].hour != target_hour:
            continue
        n_days_anchor_ok += 1

        if not calendar.is_workday(g[time_col].iloc[NOON_OFFSET].date()):
            continue
        n_days_workday += 1

        yt = pd.to_numeric(g[y_true], errors="coerce").to_numpy(np.float32)
        yp = pd.to_numeric(g[y_pred], errors="coerce").to_numpy(np.float32)
        if (not np.isfinite(yt).all()) or (not np.isfinite(yp).all()):
            continue

        err = yp - yt
        mses.append(float(np.mean(err * err)))
        maes.append(float(np.mean(np.abs(err))))
        n_days_used += 1

    print(
        f"[tsfm_zero_shot][{y_pred}] days_total={n_days_total} "
        f"full96={n_days_full96} anchor_ok={n_days_anchor_ok} workday={n_days_workday} used={n_days_used}"
    )

    if not mses:
        return float("nan"), float("nan")

    return float(np.mean(mses)), float(np.mean(maes))
'''
def _imse_imae_workday_from_windows(
    t_future: np.ndarray,          # (B,H) datetime-like
    y_true_inv: np.ndarray,         # (B,H) inverse-scale
    y_pred_inv: np.ndarray,         # (B,H) inverse-scale
    *,
    target_hour: int = 0,
    points_per_day: int = 96,
    noon_offset: int = 48,
    calendar_obj=None,
) -> Tuple[float, float, Dict[str, int], np.ndarray]:
    """
    严格复刻 _imse_imae_workday 逻辑，但输入来自 window 预测结果：
    - 每个 window 对应一天 (H=96)
    - 按 date 分组（这里其实一条 window 就是一组 day，但仍按 date 做严格检查）
    返回:
      iMSE, iMAE, counters, used_mask_per_window
    """
    tf = pd.to_datetime(t_future.reshape(-1), errors="coerce").values.reshape(t_future.shape)

    B, H = y_true_inv.shape
    assert y_pred_inv.shape == (B, H), "y_pred_inv must match y_true_inv shape"
    assert tf.shape == (B, H), "t_future must match (B,H)"

    n_days_total = 0
    n_days_full96 = 0
    n_days_anchor_ok = 0
    n_days_workday = 0
    n_days_used = 0

    mses, maes = [], []
    used_mask = np.zeros((B,), dtype=bool)

    def _is_workday_fallback(d):
        # fallback: Mon-Fri as workday
        # d is datetime.date
        return d.weekday() < 5

    is_workday_fn = None
    if calendar_obj is not None and hasattr(calendar_obj, "is_workday"):
        is_workday_fn = calendar_obj.is_workday
    else:
        is_workday_fn = _is_workday_fallback

    for i in range(B):
        n_days_total += 1

        # 取该 window 的一天
        t_day = pd.to_datetime(tf[i], errors="coerce")
        # t_day 可能是 DatetimeIndex，也可能是 ndarray[datetime64] 转出来的 DatetimeIndex
        if pd.isna(t_day).any():
            continue

        # 对齐原逻辑：len(g) == POINTS_PER_DAY
        if len(t_day) != points_per_day or H != points_per_day:
            continue
        n_days_full96 += 1

        # DatetimeIndex 用位置索引；同时兼容 Series/ndarray
        t0 = t_day[0]
        if int(getattr(t0, "hour")) != int(target_hour):
            continue
        n_days_anchor_ok += 1

        t_noon = t_day[noon_offset]
        noon_date = t_noon.date()
        if not is_workday_fn(noon_date):
            continue
        n_days_workday += 1

        yt = np.asarray(y_true_inv[i], dtype=np.float32)
        yp = np.asarray(y_pred_inv[i], dtype=np.float32)
        if (not np.isfinite(yt).all()) or (not np.isfinite(yp).all()):
            continue

        err = yp - yt
        mses.append(float(np.mean(err * err)))
        maes.append(float(np.mean(np.abs(err))))
        n_days_used += 1
        used_mask[i] = True

    logger.info(
        f"[metrics_workday] days_total={n_days_total} "
        f"full96={n_days_full96} anchor_ok={n_days_anchor_ok} "
        f"workday={n_days_workday} used={n_days_used}"
    )

    if not mses:
        return float("nan"), float("nan"), dict(
            days_total=n_days_total,
            full96=n_days_full96,
            anchor_ok=n_days_anchor_ok,
            workday=n_days_workday,
            used=n_days_used,
        ), used_mask

    return float(np.mean(mses)), float(np.mean(maes)), dict(
        days_total=n_days_total,
        full96=n_days_full96,
        anchor_ok=n_days_anchor_ok,
        workday=n_days_workday,
        used=n_days_used,
    ), used_mask



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
    
    encoder_type: str = "neural"   # ["none", "neural"]
    decoder_type: str = "neural"   # ["none", "neural"]
    normalizer: str = "identity"   # ["identity", "revin_patch"]

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
    lambda_past_recon: float = 0.5
    lambda_future_pred: float = 2.0
    lambda_latent_stats: float = 0.5
    lambda_stats_pred: float = 0.0
    lambda_y_patch_std: float = 0.5

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


# def build_adapter(cov_dim: int, z_dim: int, patch_size: int):
#     return NeuralTimeAdapter(
#         covariates_dim=cov_dim,
#         latent_dim=z_dim,
#         revin_patch_size_past=patch_size,
#         revin_patch_size_future=patch_size,
#         hidden_dim=256,
#         encoder_layers=2,
#         decoder_layers=2,
#         dropout=0.0,
#         stats_hidden_dim=256,
#         normalize_latents=True,
#     )


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
    t_f_te: np.ndarray,  # (B,H) datetime-like
    target_hour: int = 0,
    calendar_obj=None,
    use_patch_revin: bool = False,
    max_vis_windows: int = 6,
) -> Dict[str, float]:
    """
    在 test month windows 上预测，输出指标并保存少量可视化图。
    test windows 现在是：测试月内“每天 0 点起报”的日窗口集合。
    """
    B = y_p_te.shape[0]
    if B == 0:
        return dict(
            iMSE=np.nan, iMAE=np.nan, iMAPE=np.nan, iMAPE_ad=np.nan,
            coverage=np.nan, n_windows=0, n_days_used=0
        )

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

    # ---- metrics (STRICT: same logic as your _imse_imae_workday) ----
    iMSE_wd, iMAE_wd, wd_cnt, used_mask = _imse_imae_workday_from_windows(
        t_future=t_f_te,
        y_true_inv=true_y_inv,
        y_pred_inv=mean_y_inv,
        target_hour=target_hour,
        points_per_day=pred_window,    # 96
        noon_offset=pred_window // 2,  # 48
        calendar_obj=calendar_obj,
    )

    # 其他指标：同样只在 used days 上统计（口径一致）
    if np.any(used_mask):
        true_used = true_y_inv[used_mask]
        pred_used = mean_y_inv[used_mask]
        iMAPE = imape(true_used, pred_used)
        iMAPE_ad = imape_ad(true_used, pred_used)
        covered = (true_y_inv[used_mask] >= lb_y_inv[used_mask]) & (true_y_inv[used_mask] <= ub_y_inv[used_mask])
        coverage = float(covered.mean())
    else:
        iMAPE = float("nan")
        iMAPE_ad = float("nan")
        coverage = float("nan")

    # ---- per-window errors for plots (only used days) ----
    eps = 1e-6
    if np.any(used_mask):
        true_plot = true_y_inv[used_mask]
        mean_plot = mean_y_inv[used_mask]
        lb_plot = lb_y_inv[used_mask]
        ub_plot = ub_y_inv[used_mask]
        past_plot = past_y_inv[used_mask]
    else:
        true_plot = true_y_inv[:0]
        mean_plot = mean_y_inv[:0]
        lb_plot = lb_y_inv[:0]
        ub_plot = ub_y_inv[:0]
        past_plot = past_y_inv[:0]

    iMSEs = np.mean((true_plot - mean_plot) ** 2, axis=1)  # (B_used,)
    iMAEs = np.mean(np.abs(true_plot - mean_plot), axis=1)  # (B_used,)
    iMAPEs = np.mean(np.abs(true_plot - mean_plot) / (np.abs(true_plot) + eps), axis=1) * 100.0
    iMAPE_ads = np.mean(
        2.0 * np.abs(true_plot - mean_plot) / (np.abs(true_plot) + np.abs(mean_plot) + eps),
        axis=1
    ) * 100.0

    # plots
    vis_dir.mkdir(parents=True, exist_ok=True)
    L = hist_window
    H = pred_window
    t_past = np.arange(L)
    t_future = np.arange(L, L + H)

    B_used = past_plot.shape[0]
    n_show = min(max_vis_windows, B_used)

    for idx in range(n_show):
        plt.figure(figsize=(14, 4))
        plt.plot(t_past, past_plot[idx], label="History (true)")
        plt.plot(t_future, true_plot[idx], label="Future (true)")
        plt.plot(t_future, mean_plot[idx], label="Forecast (mean)")
        plt.fill_between(t_future, lb_plot[idx], ub_plot[idx], alpha=0.2, label="Uncertainty (lb~ub)")
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
    plt.title(f"Error across used test windows | patch={patch_size} | z_dim={z_dim}")
    plt.xlabel("Used test window index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "error_curve.png", dpi=150)
    plt.close()

    def safe_hist(ax, x, bins=20, label=None, alpha=0.5):
        x = np.asarray(x).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size < 2:
            ax.text(0.5, 0.8, f"{label}: <2 finite points", ha="center", transform=ax.transAxes)
            return
        xmin, xmax = float(x.min()), float(x.max())
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            ax.text(0.5, 0.8, f"{label}: zero/invalid range", ha="center", transform=ax.transAxes)
            return
        b = int(min(bins, max(1, x.size - 1)))
        ax.hist(x, bins=b, alpha=alpha, label=label)

    fig, ax = plt.subplots(figsize=(12, 3))
    safe_hist(ax, iMSEs, bins=20, label="iMSE")
    safe_hist(ax, iMAEs, bins=20, label="iMAE")
    safe_hist(ax, iMAPEs, bins=20, label="iMAPE (%)")
    safe_hist(ax, iMAPE_ads, bins=20, label="iMAPE-ad (%)")
    ax.set_title(f"Error distribution (used days) | patch={patch_size} | z_dim={z_dim}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vis_dir / "error_hist.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(6, 3))
    plt.bar(["coverage"], [coverage if np.isfinite(coverage) else 0.0])
    plt.ylim(0, 1)
    plt.title("Interval coverage (used days)")
    plt.tight_layout()
    plt.savefig(vis_dir / "interval_coverage.png", dpi=150)
    plt.close()

    logger.info(
        f"[workday-filter] {target_name} "
        f"days_total={wd_cnt['days_total']} full96={wd_cnt['full96']} "
        f"anchor_ok={wd_cnt['anchor_ok']} workday={wd_cnt['workday']} used={wd_cnt['used']}"
    )

    return dict(
        iMSE=iMSE_wd,
        iMAE=iMAE_wd,
        iMAPE=iMAPE,
        iMAPE_ad=iMAPE_ad,
        coverage=coverage,
        n_windows=int(B),
        n_days_used=int(wd_cnt.get("used", 0)),
    )


def build_parser():
    import argparse
    p = argparse.ArgumentParser("monthly backtest pipeline")

    # io
    p.add_argument("--data_path", type=str, default="/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv")
    p.add_argument("--out_root", type=str, default="./results_monthly_backtest")
    p.add_argument("--time_col", type=str, default="time")
    p.add_argument("--target_col", type=str, default="day_ahead_clearing_price")
    p.add_argument("--drop_cols", type=str, default="time,realtime_clearing_price")

    # backtest range
    p.add_argument("--start_ym", type=str, default="2024-08")
    p.add_argument("--end_ym", type=str, default="2025-07")
    p.add_argument("--train_years", type=int, default=2)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--target_hour", type=int, default=0)
    p.add_argument("--freq_min", type=int, default=15)

    # window
    p.add_argument("--hist_window", type=int, default=2880)
    p.add_argument("--pred_window", type=int, default=96)
    p.add_argument("--stride_train", type=int, default=96)
    p.add_argument("--stride_eval", type=int, default=96)

    # model / pipeline
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--model_name", type=str, default="thuml/sundial-base-128m")
    p.add_argument("--z_dim", type=int, default=1)
    p.add_argument("--patch_size", type=int, default=16)

    p.add_argument("--encoder_type", type=str, default="neural", choices=["none", "neural"])
    p.add_argument("--decoder_type", type=str, default="neural", choices=["none", "neural"])
    p.add_argument("--normalizer", type=str, default="identity", choices=["identity", "revin_patch"])

    # train
    p.add_argument("--stats_pretrain_epochs", type=int, default=80)
    p.add_argument("--stats_pretrain_bs", type=int, default=64)
    p.add_argument("--stats_pretrain_lr", type=float, default=1e-4)
    p.add_argument("--stats_pretrain_wd", type=float, default=1e-4)
    p.add_argument("--stats_pretrain_patience", type=int, default=8)

    p.add_argument("--past_pretrain_epochs", type=int, default=20)
    p.add_argument("--past_pretrain_bs", type=int, default=32)
    p.add_argument("--past_pretrain_lr", type=float, default=1e-4)
    p.add_argument("--past_pretrain_wd", type=float, default=1e-4)

    p.add_argument("--train_epochs", type=int, default=60)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--train_lr", type=float, default=1e-4)
    p.add_argument("--train_wd", type=float, default=1e-4)

    p.add_argument("--lambda_past_recon", type=float, default=0.5)
    p.add_argument("--lambda_future_pred", type=float, default=2.0)
    p.add_argument("--lambda_latent_stats", type=float, default=0.5)
    p.add_argument("--lambda_stats_pred", type=float, default=0.0)
    p.add_argument("--lambda_y_patch_std", type=float, default=0.5)

    p.add_argument("--ltm_batch_size_train", type=int, default=1)
    p.add_argument("--ltm_batch_size_pred", type=int, default=32)

    # predict / vis
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--max_vis_windows", type=int, default=6)

    # debug
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_every", type=int, default=2)
    p.add_argument("--debug_num_seq", type=int, default=2)
    p.add_argument("--debug_latent_ch", type=int, default=1)
    p.add_argument("--earlystop_patience", type=int, default=10)

    return p


# =========================
# Main rolling backtest
# =========================

def main():
    args = build_parser().parse_args()

    # ---- seed & device ----
    device = get_device()
    set_seed(args.seed)
    logger.info("device:", device)
    
    # ---- configs ----
    wcfg = WindowConfig(
        hist_window=args.hist_window,
        pred_window=args.pred_window,
        stride_train=args.stride_train,
        stride_eval=args.stride_eval,
    )
    cfg = TrainConfig(
        seed=args.seed,
        model_name=args.model_name,
        z_dim=args.z_dim,
        patch_size=args.patch_size,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        normalizer=args.normalizer,
        stats_pretrain_epochs=args.stats_pretrain_epochs,
        stats_pretrain_bs=args.stats_pretrain_bs,
        stats_pretrain_lr=args.stats_pretrain_lr,
        stats_pretrain_wd=args.stats_pretrain_wd,
        stats_pretrain_patience=args.stats_pretrain_patience,
        past_pretrain_epochs=args.past_pretrain_epochs,
        past_pretrain_bs=args.past_pretrain_bs,
        past_pretrain_lr=args.past_pretrain_lr,
        past_pretrain_wd=args.past_pretrain_wd,
        train_epochs=args.train_epochs,
        train_bs=args.train_bs,
        train_lr=args.train_lr,
        train_wd=args.train_wd,
        lambda_past_recon=args.lambda_past_recon,
        lambda_future_pred=args.lambda_future_pred,
        lambda_latent_stats=args.lambda_latent_stats,
        lambda_stats_pred=args.lambda_stats_pred,
        lambda_y_patch_std=args.lambda_y_patch_std,
        ltm_batch_size_train=args.ltm_batch_size_train,
        ltm_batch_size_pred=args.ltm_batch_size_pred,
        n_samples=args.n_samples,
        max_vis_windows=args.max_vis_windows,
    )

    logger.info("[config] window:", asdict(wcfg))
    logger.info("[config] train :", asdict(cfg))
    logger.info("[config] misc  :", dict(
        start_ym=args.start_ym, end_ym=args.end_ym,
        train_years=args.train_years, train_ratio=args.train_ratio,
        target_hour=args.target_hour, freq_min=args.freq_min,
        debug=args.debug, debug_every=args.debug_every,
        earlystop_patience=args.earlystop_patience,
    ))

    # ---- load data ----
    df = load_dataset(args.data_path, time_col=args.time_col)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    X_all_raw, y_all_raw, cov_cols = build_xy_from_df(df, target_col=args.target_col, drop_cols=drop_cols)
    cov_dim = X_all_raw.shape[1]

    assert wcfg.hist_window % cfg.patch_size == 0 and wcfg.pred_window % cfg.patch_size == 0, \
        "L/H must be divisible by patch_size"

    test_months = month_starts(args.start_ym, args.end_ym)
    logger.info("test_months:", [d.strftime("%Y-%m-%d") for d in test_months], "len=", len(test_months))

    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for m0 in test_months:
        split = split_train_test_by_month(
            df=df,
            time_col=args.time_col,
            test_month_start=m0,
            train_years=args.train_years,
            test_months=1,
            train_ratio=args.train_ratio,
        )

        d_train = split["train"]
        d_val = split["val"]

        month_tag = m0.strftime("%Y-%m")
        test_start = m0
        test_end = m0 + pd.DateOffset(months=1)

        logger.info("\n====================================================")
        logger.info(f"[Backtest month] {month_tag}")
        logger.info(f"train: {split['train_start'].date()} -> {split['val_start'].date()}  (len={len(d_train)})")
        logger.info(f"val  : {split['val_start'].date()} -> {split['val_end'].date()}    (len={len(d_val)})")
        logger.info(f"test(daily anchors): {test_start.date()} -> {test_end.date()}")

        if len(d_train) < (wcfg.hist_window + wcfg.pred_window):
            logger.info(f"[skip] Not enough train samples for windows in month {month_tag}")
            results.append(dict(
                month=month_tag,
                iMSE=np.nan, iMAE=np.nan, iMAPE=np.nan, iMAPE_ad=np.nan,
                coverage=np.nan, n_windows=0, n_days_used=0
            ))
            continue

        X_tr_raw, y_tr_raw, _ = build_xy_from_df(d_train, target_col=args.target_col, drop_cols=drop_cols)
        X_va_raw, y_va_raw, _ = build_xy_from_df(d_val, target_col=args.target_col, drop_cols=drop_cols)

        scaler_X, scaler_y = fit_scalers(X_tr_raw, y_tr_raw)

        X_tr_s, y_tr_s = scale_xy(X_tr_raw, y_tr_raw, scaler_X, scaler_y)
        X_va_s, y_va_s = scale_xy(X_va_raw, y_va_raw, scaler_X, scaler_y)

        y_p_tr, x_p_tr, x_f_tr, y_f_tr = make_windows(X_tr_s, y_tr_s, wcfg.hist_window, wcfg.pred_window, wcfg.stride_train)
        y_p_va, x_p_va, x_f_va, y_f_va = make_windows(X_va_s, y_va_s, wcfg.hist_window, wcfg.pred_window, wcfg.stride_eval)

        y_p_te, x_p_te, x_f_te, y_f_te, t_f_te = make_daily_anchored_test_windows(
            df=df,
            time_col=args.time_col,
            target_col=args.target_col,
            cov_cols=cov_cols,
            drop_cols=drop_cols,
            month_start=test_start,
            month_end=test_end,
            hist_window=wcfg.hist_window,
            pred_window=wcfg.pred_window,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            freq=args.freq_min,
            target_hour=args.target_hour,
        )

        logger.info(f"[windows] train={y_p_tr.shape[0]}  val={y_p_va.shape[0]}  test(daily)={y_p_te.shape[0]}")

        if y_p_tr.shape[0] == 0:
            logger.info(f"[skip] Train windows empty in month {month_tag}")
            results.append(dict(month=month_tag, iMSE=np.nan, iMAE=np.nan, iMAPE=np.nan, iMAPE_ad=np.nan, coverage=np.nan, n_windows=0, n_days_used=0))
            continue

        if y_p_te.shape[0] == 0:
            logger.info(f"[skip] Test daily windows empty in month {month_tag}")
            results.append(dict(month=month_tag, iMSE=np.nan, iMAE=np.nan, iMAPE=np.nan, iMAPE_ad=np.nan, coverage=np.nan, n_windows=0, n_days_used=0))
            continue

        # ---- build model for this month ----
        iclearner = build_iclearner(cfg.model_name, cfg.z_dim, wcfg.pred_window, device=device)

        # pipeline adapter (encoder/decoder/normalizer/stats predictor)
        adapter = build_pipeline_adapter(
            cov_dim=cov_dim,
            z_dim=cfg.z_dim,
            patch_size=cfg.patch_size,
            pred_window=wcfg.pred_window,
            encoder_type=cfg.encoder_type,
            decoder_type=cfg.decoder_type,
            normalizer=cfg.normalizer,
        )

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

        # ---- pretrain stats predictor ONLY if enabled by pipeline ----
        if model2.adapter.has_stats_predictor:
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
        else:
            logger.info(">>> Skip stats_pretrain (pipeline has no stats_predictor).")

        # ---- pretrain past reconstruction ----
        if model2.adapter.can_reconstruct_past:
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
        else:
            logger.info(">>> Skip past_reconstruction_pretrain (pipeline encoder/decoder is no-op).")

        # ---- train adapter ----
        if model2.adapter.is_trainable:
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
                debug=args.debug,
                debug_dir=str(out_root / f"debug_train_adapter_{month_tag}"),
                debug_every=args.debug_every,
                debug_num_seq=args.debug_num_seq,
                debug_latent_ch=args.debug_latent_ch,
                earlystop_patience=args.earlystop_patience,
            )
            logger.info(">>> Done training adapter.")
        else:
            logger.info(">>> Skip train_adapter (pipeline is not trainable).")

        # ---- test metrics + plots ----
        vis_dir = out_root / f"vis_{month_tag}"
        metrics = predict_and_metrics_one_month(
            model=model2,
            scaler_y=scaler_y,
            y_p_te=y_p_te,
            x_p_te=x_p_te,
            x_f_te=x_f_te,
            y_f_te=y_f_te,
            t_f_te=t_f_te,
            hist_window=wcfg.hist_window,
            pred_window=wcfg.pred_window,
            ltm_batch_size=cfg.ltm_batch_size_pred,
            n_samples=cfg.n_samples,
            vis_dir=vis_dir,
            target_name=args.target_col,
            patch_size=cfg.patch_size,
            z_dim=cfg.z_dim,
            target_hour=args.target_hour,
            calendar_obj=None,
            max_vis_windows=cfg.max_vis_windows,
        )

        logger.info(
            f"[Test {month_tag}] iMSE={metrics['iMSE']:.4f}  iMAE={metrics['iMAE']:.4f} "
            f"iMAPE={metrics['iMAPE']:.2f}% iMAPE-adj={metrics['iMAPE_ad']:.2f}% "
            f"coverage={metrics['coverage']:.3f}  n_windows={metrics['n_windows']}  "
            f"n_days_used={metrics.get('n_days_used', 0)}"
        )
        results.append(dict(month=month_tag, **metrics))

    # ---- save summary ----
    res_df = pd.DataFrame(results)
    res_path = out_root / "results_summary.csv"
    res_df.to_csv(res_path, index=False)

    logger.info("\n==================== SUMMARY (per month) ====================")
    logger.info(res_df)
    logger.info(f"[Saved] {res_path}")

    if len(res_df) > 0:
        cols = ["iMSE", "iMAE", "iMAPE", "iMAPE_ad", "coverage"]
        overall = {c: float(np.nanmean(res_df[c].to_numpy(dtype=np.float64))) for c in cols if c in res_df.columns}
        logger.info("\n==================== SUMMARY (nan-mean over months) ====================")
        logger.info(overall)

    logger.info("=================================================\n")


if __name__ == "__main__":
    main()