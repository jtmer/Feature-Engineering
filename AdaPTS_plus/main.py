import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

adapts_repo_root = "/data/mahaoke/AdaPTS"
import sys
sys.path.insert(0, adapts_repo_root)

from adapts.icl.sundial import SundialICLTrainer

from conditional_adapters import ConditionalPatchVAEAdapter, ConditionalPatchPCAAdapter, NeuralTimeAdapter
from conditional_adapts import ConditionalAdaPTS
from conditional_adapters_easy import SmoothPatchAdapter
from conditional_adapts_easy import SimplePatchAdaPTS

def normalize_cov_per_sample(x: np.ndarray, eps: float = 1e-5):
    # x: (B, C, T)
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


path = "/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv"
data = pd.read_csv(path)
data = data.dropna(axis=1, how="any")
data = data.drop(columns=["time", "realtime_clearing_price"])

target_index = data.columns.get_indexer(["day_ahead_clearing_price"])[0]
cols = list(data.columns)
cols.append(cols.pop(target_index))
cov_cols = cols[:-1]
target_col = cols[-1]

price_all = data[target_col].values.astype(np.float32)
X_all = data[cov_cols].values.astype(np.float32)

train_start = 0
train_number = 3600 * 15
vali_number  = 3600 * 5
test_number  = 3600 * 2

train_end = train_start + train_number
val_start = train_end
val_end = val_start + vali_number
test_start = val_end
test_end = test_start + test_number

hist_window = 2880
pred_window = 720

scaler_X = StandardScaler().fit(X_all[train_start:train_end])
scaler_y = StandardScaler().fit(price_all[train_start:train_end].reshape(-1, 1))

print("train raw mean:", price_all[train_start:train_end].mean())
print("val   raw mean:", price_all[val_start:val_end].mean())
print("test  raw mean:", price_all[test_start:test_end].mean())

stds = scaler_X.scale_
print("scaler_X std min/median/max:", stds.min(), np.median(stds), stds.max())
print("num near-zero std cols:", np.sum(stds < 1e-6))

X_train_raw = X_all[train_start:train_end]
print("X_train_raw min/mean/max:", np.nanmin(X_train_raw), np.nanmean(X_train_raw), np.nanmax(X_train_raw))
print("Any inf?", np.isinf(X_train_raw).any(), "Any nan?", np.isnan(X_train_raw).any())


# TODO 是否去掉
def scale_xy(x_cov: np.ndarray, y: np.ndarray):
    x = scaler_X.transform(x_cov)
    
    x = np.clip(x, -20.0, 20.0)
    
    yy = scaler_y.transform(y.reshape(-1,1)).reshape(-1)
    return x.astype(np.float32), yy.astype(np.float32)

X_all_s, y_all_s = scale_xy(X_all, price_all)
# X_all_s, y_all_s = X_all, price_all

def make_windows(x_cov: np.ndarray, y: np.ndarray, L: int, H: int, stride: int = 1):
    """
    x_cov: (T,Cx)
    y:     (T,)
    return:
      y_past:   (N,1,L)
      x_past:   (N,Cx,L)
      x_future: (N,Cx,H)
      y_future: (N,1,H)
    """
    T, Cx = x_cov.shape
    max_i = T - L - H
    ys_p, xs_p, xs_f, ys_f = [], [], [], []
    for i in range(0, max_i + 1, stride):
        x_p = x_cov[i:i+L].T              # (Cx,L)
        y_p = y[i:i+L][None, :]           # (1,L)
        x_f = x_cov[i+L:i+L+H].T          # (Cx,H)
        y_f = y[i+L:i+L+H][None, :]       # (1,H)
        xs_p.append(x_p); ys_p.append(y_p); xs_f.append(x_f); ys_f.append(y_f)
    return (
        np.stack(ys_p).astype(np.float32),
        np.stack(xs_p).astype(np.float32),
        np.stack(xs_f).astype(np.float32),
        np.stack(ys_f).astype(np.float32),
    )

stride = 720
# 训练集
y_p_tr, x_p_tr, x_f_tr, y_f_tr = make_windows(
    X_all_s[train_start:train_end], y_all_s[train_start:train_end],
    hist_window, pred_window, stride
)
# 验证集
y_p_va, x_p_va, x_f_va, y_f_va = make_windows(
    X_all_s[val_start:val_end], y_all_s[val_start:val_end],
    hist_window, pred_window, int(stride/6)
)
# 测试集
y_p_te, x_p_te, x_f_te, y_f_te = make_windows(
    X_all_s[test_start:test_end], y_all_s[test_start:test_end],
    hist_window, pred_window, int(stride/6)
)


# # ====== ablation: no covariates (all zeros) ======
# x_p_tr = np.zeros_like(x_p_tr)
# x_f_tr = np.zeros_like(x_f_tr)

# x_p_va = np.zeros_like(x_p_va)
# x_f_va = np.zeros_like(x_f_va)

# x_p_te = np.zeros_like(x_p_te)
# x_f_te = np.zeros_like(x_f_te)

# x_p_tr = normalize_cov_per_sample(x_p_tr)
# x_p_va = normalize_cov_per_sample(x_p_va)


seed = 13
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

model_name = "thuml/sundial-base-128m"
forecast_horizon = pred_window

# 注意：此处 n_features = z_dim（代理通道数），因为 FM 是在代理空间做预测
z_dim = 12
iclearner = SundialICLTrainer(
    sundial_name=model_name,
    n_features=z_dim,
    forecast_horizon=pred_window,   # 先用 pred_window；如果 patch_size>1，后面会在 ConditionalAdaPTS 里换算 token horizon
    device=device,
    trust_remote_code=True,
)

cov_dim = x_p_tr.shape[1]
# patch_size = 16
patch_size = 24

assert hist_window % patch_size == 0 and pred_window % patch_size == 0, "L/H must be divisible by patch_size"

adapter = NeuralTimeAdapter(
    # covariates_dim=cov_dim+2,
    covariates_dim=cov_dim,
    latent_dim=z_dim,
    revin_patch_size_past=24,
    revin_patch_size_future=24,
    hidden_dim=256,
    encoder_layers=2,
    decoder_layers=2,
    dropout=0.0,
    stats_hidden_dim=256,
    normalize_latents=True
)
# adapter = SmoothPatchAdapter(
#     cov_dim=cov_dim,
#     z_dim=z_dim,
#     patch_size=patch_size,
#     hidden_dim=256,
#     enc_layers=2,
#     dec_layers=2,
#     dropout=0.0,
# )

coeff_kl = 0.0

model2 = ConditionalAdaPTS(adapter=adapter, iclearner=iclearner, device=device)
# model2 = SimplePatchAdaPTS(adapter=adapter, iclearner=iclearner, device=device)

val_data = dict(
    past_target=y_p_va,
    past_covariates=x_p_va,
    future_target=y_f_va,
    future_covariates=x_f_va,
)
print(">>> Pretrain future_stats_predictor ...")
stats_val_data = dict(
    future_target=y_f_va,
    future_covariates=x_f_va,
)
model2.pretrain_stats_predictor(
    future_target=y_f_tr,      # (N,1,H)
    future_covariates=x_f_tr,  # (N,Cx,H)
    n_epochs=100,
    batch_size=64,
    lr=1e-4,
    weight_decay=1e-4,
    patience=20,
    val_data=stats_val_data,
    verbose=True,
    use_swanlab=False,
    # swanlab_run=run,
)
# for p in adapter.future_stats_predictor.parameters():
#     p.requires_grad = False
print(">>> Done pretraining stats predictor.")
model2.train_adapter(
    past_target=y_p_tr,
    past_covariates=x_p_tr,
    future_target=y_f_tr,
    future_covariates=x_f_tr,
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-4,
    lambda_past_recon=0.4,
    lambda_future_pred=2.0,
    lambda_latent_stats=0,
    lambda_stats_pred=0.4,
    lambda_y_patch_std=0,
    val_data=val_data,
    verbose=True,
    use_swanlab=False,
    # swanlab_run=run,
)
# model2.train_adapter(
#     past_target=y_p_tr,
#     past_covariates=x_p_tr,
#     future_target=y_f_tr,
#     future_covariates=x_f_tr,
#     n_epochs=100,
#     batch_size=32,
#     lr=1e-4,
#     weight_decay=1e-4,
#     lambda_past_recon=0.6,
#     lambda_future_pred=1.0,
#     lambda_latent_smooth=0.1,
#     lambda_latent_align=0.1,
#     ltm_batch_size=32,
#     val_data=val_data,
#     verbose=True,
#     use_swanlab=False,
# )

# =====================测试
B = min(256, y_p_te.shape[0])

x_future_true = x_f_te[:B].copy()
x_future_zero = np.zeros_like(x_future_true)

# 打乱 cov（保持分布但打乱配对）
perm = np.random.permutation(B)
x_future_shuffle = x_future_true[perm]

pack_true = model2.predict(
    past_target=y_p_te[:B], past_covariates=x_p_te[:B], future_covariates=x_future_true,
    pred_horizon=pred_window, fm_batch_size=32, n_samples=30,
)

pack_zero = model2.predict(
    past_target=y_p_te[:B], past_covariates=x_p_te[:B], future_covariates=x_future_zero,
    pred_horizon=pred_window, fm_batch_size=32, n_samples=30,
)

pack_shuf = model2.predict(
    past_target=y_p_te[:B], past_covariates=x_p_te[:B], future_covariates=x_future_shuffle,
    pred_horizon=pred_window, fm_batch_size=32, n_samples=30,
)

mean_true = pack_true.mean[:, 0, :]  # (B,H) scaled
mean_zero = pack_zero.mean[:, 0, :]
mean_shuf = pack_shuf.mean[:, 0, :]

# 反标准化看尺度
mean_true_inv = scaler_y.inverse_transform(mean_true.reshape(-1,1)).reshape(mean_true.shape)
mean_zero_inv = scaler_y.inverse_transform(mean_zero.reshape(-1,1)).reshape(mean_zero.shape)
mean_shuf_inv = scaler_y.inverse_transform(mean_shuf.reshape(-1,1)).reshape(mean_shuf.shape)

print("\n==== x_future sanity check (inverse scale) ====")
print("pred(mean) with TRUE cov  min/mean/max:", mean_true_inv.min(), mean_true_inv.mean(), mean_true_inv.max())
print("pred(mean) with ZERO cov  min/mean/max:", mean_zero_inv.min(), mean_zero_inv.mean(), mean_zero_inv.max())
print("pred(mean) with SHUF cov  min/mean/max:", mean_shuf_inv.min(), mean_shuf_inv.mean(), mean_shuf_inv.max())

# 也看“相对差异”：
delta_true_zero = np.mean(np.abs(mean_true_inv - mean_zero_inv))
delta_true_shuf = np.mean(np.abs(mean_true_inv - mean_shuf_inv))
print("MAE(pred_true, pred_zero):", float(delta_true_zero))
print("MAE(pred_true, pred_shuf):", float(delta_true_shuf))
# =====================测试


# x_f_te = normalize_cov_per_sample(x_f_te[:B].copy())
# x_p_te   = normalize_cov_per_sample(x_p_te[:B].copy())

# 推理：用真 x_future
pack = model2.predict(
    past_target=y_p_te[:256],
    past_covariates=x_p_te[:256],
    future_covariates=x_f_te[:256],
    pred_horizon=pred_window,
    fm_batch_size=32,
    n_samples=30,
)

mean_y = pack.mean[:, 0, :]   # (B,H)
lb_y   = pack.lb[:, 0, :]
ub_y   = pack.ub[:, 0, :]
true_y = y_f_te[:256, 0, :]


print("\n==== Sanity check scales ====")
print("scaled y_train mean/std:", y_all_s[train_start:train_end].mean(), y_all_s[train_start:train_end].std())
print("raw   y_train mean/std:", price_all[train_start:train_end].mean(), price_all[train_start:train_end].std())

print("mean_y (scaled?)  min/mean/max:", float(mean_y.min()), float(mean_y.mean()), float(mean_y.max()))
print("true_y (scaled)   min/mean/max:", float(true_y.min()), float(true_y.mean()), float(true_y.max()))


# 反标准化回真实尺度
mean_y_inv = scaler_y.inverse_transform(mean_y.reshape(-1,1)).reshape(mean_y.shape)
true_y_inv = scaler_y.inverse_transform(true_y.reshape(-1,1)).reshape(true_y.shape)

print("shape:")
print("mean_y_inv:", mean_y_inv.shape, "true_y_inv:", true_y_inv.shape)
print("前10个预测值（inverse scale）:")
print(mean_y_inv[0, :10], true_y_inv[0, :10])


import matplotlib.pyplot as plt

save_dir = Path("./vis_conditional_adapts")
save_dir.mkdir(parents=True, exist_ok=True)

# y_p_te: (N,1,L) scaled
# mean_y_inv/true_y_inv: (B,H) already inverse
B_plot = min(256, y_p_te.shape[0], mean_y_inv.shape[0])
L = hist_window
H = pred_window

# 历史真值 (inverse)
past_y_scaled = y_p_te[:B_plot, 0, :]  # (B,L)
past_y_inv = scaler_y.inverse_transform(past_y_scaled.reshape(-1, 1)).reshape(B_plot, L)

# 未来真值和预测 (inverse)
future_true_inv = true_y_inv[:B_plot]   # (B,H)
future_mean_inv = mean_y_inv[:B_plot]   # (B,H)
future_lb_inv   = scaler_y.inverse_transform(lb_y[:B_plot].reshape(-1,1)).reshape(B_plot, H)
future_ub_inv   = scaler_y.inverse_transform(ub_y[:B_plot].reshape(-1,1)).reshape(B_plot, H)

# ========= 1) 单窗口可视化（保存多张） =========
num_examples = min(6, B_plot)
t_past = np.arange(L)
t_future = np.arange(L, L + H)

for idx in range(num_examples):
    plt.figure(figsize=(14, 4))

    plt.plot(t_past, past_y_inv[idx], label="History (true)")
    plt.plot(t_future, future_true_inv[idx], label="Future (true)")
    plt.plot(t_future, future_mean_inv[idx], label="Forecast (mean)")
    plt.fill_between(
        t_future,
        future_lb_inv[idx],
        future_ub_inv[idx],
        alpha=0.2,
        label="Uncertainty band (lb~ub)"
    )
    plt.axvline(L - 1, linestyle="--")

    plt.title(f"ConditionalAdaPTS+Sundial Forecast (idx={idx}) | patch={patch_size} | z_dim={z_dim}")
    plt.xlabel("Time index within window")
    plt.ylabel(target_col)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = save_dir / f"forecast_window_{idx:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

print(f"[Saved] example forecast plots -> {save_dir}")

# ========= 2) 整体误差可视化 =========
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))  # per window

def mae(a, b):
    return np.mean(np.abs(a - b), axis=1)          # per window

# per-window metrics on inverse scale
rmses = rmse(future_true_inv, future_mean_inv)  # (B,)
maes  = mae(future_true_inv, future_mean_inv)   # (B,)

print(f"Test windows used for plots: {B_plot}")
print(f"RMSE mean={rmses.mean():.4f}, std={rmses.std():.4f}")
print(f"MAE  mean={maes.mean():.4f}, std={maes.std():.4f}")

# 2.1 曲线图
plt.figure(figsize=(12, 3))
plt.plot(rmses, label="RMSE per window")
plt.plot(maes, label="MAE per window")
plt.title(f"Error across test windows | patch={patch_size} | z_dim={z_dim}")
plt.xlabel("Test window index")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "error_curve.png", dpi=150)
plt.close()

# 2.2 直方图
plt.figure(figsize=(12, 3))
plt.hist(rmses, bins=20, alpha=0.7, label="RMSE")
plt.hist(maes, bins=20, alpha=0.7, label="MAE")
plt.title(f"Error distribution | patch={patch_size} | z_dim={z_dim}")
plt.legend()
plt.tight_layout()
plt.savefig(save_dir / "error_hist.png", dpi=150)
plt.close()

print(f"[Saved] error plots -> {save_dir}")

# ========= 3) 可选：画“覆盖率”（区间是否覆盖真值） =========
# coverage = mean over all points: true in [lb, ub]
covered = (future_true_inv >= future_lb_inv) & (future_true_inv <= future_ub_inv)
coverage = covered.mean()
print(f"Interval coverage (lb~ub): {coverage:.3f}")

plt.figure(figsize=(6, 3))
plt.bar(["coverage"], [coverage])
plt.ylim(0, 1)
plt.title("Interval coverage (lb~ub)")
plt.tight_layout()
plt.savefig(save_dir / "interval_coverage.png", dpi=150)
plt.close()