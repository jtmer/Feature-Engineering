import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

adapts_repo_root = "/data/mahaoke/AdaPTS"
if adapts_repo_root is not None:
    import sys
    sys.path.insert(0, adapts_repo_root)

from adapts import adapts, adapters
from adapts.icl import iclearner as icl
from adapts.icl.sundial import SundialICLTrainer
from adapts.adapters import betaVAE

def info(name, arr):
    mb = arr.nbytes / (1024**2)
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, size={mb:.1f}MB")


path = "/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv"
data = pd.read_csv(path)
data = data.dropna(axis=1, how="any")
data = data.drop(columns=["time", "realtime_clearing_price"])

target_index = data.columns.get_indexer(["day_ahead_clearing_price"])[0]
cols = list(data.columns)
cols.append(cols.pop(target_index))
target_columns = cols

cov_cols = target_columns[:-1]
target_col = target_columns[-1]


train_start = 0
train_number = 3600 * 15
vali_number = 3600 * 5
test_number = 3600 * 2

hist_window = 2880
pred_window = 720
scale = True

price_all = data[target_col].values.astype(np.float32)
X_all = data[cov_cols].values.astype(np.float32)

train_end = train_start + train_number
val_start = train_end
val_end = val_start + vali_number
test_start = val_end
test_end = test_start + test_number

X_train_raw = X_all[train_start:train_end]
y_train_raw = price_all[train_start:train_end]

scaler_X = StandardScaler().fit(X_train_raw)
scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))


def build_multivariate_series(X_cov: np.ndarray, y: np.ndarray, do_scale: bool) -> np.ndarray:
    """
    返回 multivariate time series: shape (T, n_features)
    其中最后一个特征是目标 y。
    """
    if do_scale:
        X_cov = scaler_X.transform(X_cov)
        y = scaler_y.transform(y.reshape(-1, 1)).reshape(-1)

    ts = np.concatenate([X_cov, y.reshape(-1, 1)], axis=1).astype(np.float32)
    return ts


ts_train = build_multivariate_series(X_all[train_start:train_end], price_all[train_start:train_end], scale)
ts_val   = build_multivariate_series(X_all[val_start:val_end],     price_all[val_start:val_end],     scale)
ts_test  = build_multivariate_series(X_all[test_start:test_end],   price_all[test_start:test_end],   scale)

n_features = ts_train.shape[1]  # cov_dim + 1（目标作为最后一维）


# 构造监督学习窗口（匹配示例的 channels-first 形状）
#    X: (N, n_features, context_length)
#    y: (N, n_features, forecast_horizon)
def make_supervised_windows(ts: np.ndarray, context_length: int, horizon: int, stride: int = 1):
    """
    ts: (T, n_features)
    returns:
      X: (N, n_features, context_length)
      Y: (N, n_features, horizon)
    """
    T, C = ts.shape
    max_i = T - context_length - horizon
    if max_i < 0:
        raise ValueError(f"T={T} is too short for context={context_length} and horizon={horizon}")

    X_list, Y_list = [], []
    for i in range(0, max_i + 1, stride):
        x = ts[i : i + context_length].T              # (C, context)
        y = ts[i + context_length : i + context_length + horizon].T  # (C, horizon)
        X_list.append(x)
        Y_list.append(y)

    X = np.stack(X_list, axis=0).astype(np.float32)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y


# stride = 3600
stride = 720

X_train, y_train = make_supervised_windows(ts_train, hist_window, pred_window, stride=stride)
X_val,   y_val   = make_supervised_windows(ts_val,   hist_window, pred_window, stride=stride)
X_test,  y_test  = make_supervised_windows(ts_test,  hist_window, pred_window, stride=stride)

print("\n==== Dataset windows ====")
info("X_train", X_train); info("y_train", y_train)
info("X_val",   X_val);   info("y_val",   y_val)
info("X_test",  X_test);  info("y_test",  y_test)

# test 用于 predict_multi_step 的拼接（按示例）
time_series_test = np.concatenate([X_test, y_test], axis=-1)  # (N, n_features, hist+pred)


seed = 13
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_name = 'thuml/sundial-base-128m'

context_length = hist_window
forecast_horizon = pred_window

# AdaPTS 里 adapter 的“降维通道数”（示例用 n_components=7）
# 经验：不要大于 n_features；也可以直接设成 n_features（等价不降维）
n_components = min(12, n_features)
# n_components = n_features

iclearner = SundialICLTrainer(
    sundial_name=model_name,
    n_features=n_components,          # 代理空间通道数
    forecast_horizon=forecast_horizon,
    device=device,
    trust_remote_code=True,
)

use_revin = True

# 配置 adapter（按示例：betaVAE + MultichannelProjector）
#      adapter_base = None   # 不用学习 adapter
#      adapter_base = "pca"  # PCA adapter
adapter_params = {
    "input_dim": n_features,
    "device": device,
    "context_length": context_length,
    "forecast_horizon": forecast_horizon,
    "n_components": n_components,
    "use_revin": use_revin,
    "num_layers": 2,
    "hidden_dim": 128,
    "beta": 1.0,
}

adapter_base = betaVAE(**adapter_params).to(device)
# adapter_base = None
# adapter_base = "pca"

adapter = adapters.MultichannelProjector(
    num_channels=n_features,
    new_num_channels=n_components,
    patch_window_size=None,
    base_projector=adapter_base,
    device=device,
    use_revin=use_revin,
    context_length=context_length,
    forecast_horizon=forecast_horizon,
)

pca_in_preprocessing = False

adapts_model = adapts.ADAPTS(
    adapter=adapter,
    iclearner=iclearner,
    n_features=n_features,
    n_components=n_components,
    pca_in_preprocessing=pca_in_preprocessing,
)

batch_size = 32
learning_rate = 1e-3
n_epochs_fine_tuning = 50
n_epochs_adapter = 100
log_dir = Path("./logs") / "adapts_your_data"
log_dir.mkdir(parents=True, exist_ok=True)

if adapter_base in [None, "pca"]:
    # 不需要监督学习 adapter：直接 fit
    adapts_model.fit_adapter(X=np.concatenate([X_train, X_val], axis=0))
else:
    # 监督微调 adapter
    adapts_model.adapter_supervised_fine_tuning(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=1,
        n_epochs=n_epochs_adapter,
        log_dir=log_dir,
    )
    
print("\n==== Training adapter supervised fine-tuning ====")
print(f"epochs={n_epochs_adapter}, batch_size={batch_size}, lr={learning_rate}")
print("device:", device)

inference_batch_size = 128
n_samples = 25

with torch.no_grad():
    mean, _, lb, ub = adapts_model.predict_multi_step(
        X=time_series_test,
        prediction_horizon=forecast_horizon,
        batch_size=inference_batch_size,
        n_samples=n_samples,
    )

# mean/lb/ub 形状通常是 (N, n_features, forecast_horizon) 或类似
# 我们只关心最后一个特征（目标列）
target_ch = n_features - 1

mean_y = mean[:, target_ch, :]  # (N, horizon)
lb_y   = lb[:, target_ch, :]
ub_y   = ub[:, target_ch, :]

# 反标准化到原始价格尺度
# scaler_y 是对 shape (T,1) 拟合的，所以这里要 reshape
mean_y_inv = scaler_y.inverse_transform(mean_y.reshape(-1, 1)).reshape(mean_y.shape)
lb_y_inv   = scaler_y.inverse_transform(lb_y.reshape(-1, 1)).reshape(lb_y.shape)
ub_y_inv   = scaler_y.inverse_transform(ub_y.reshape(-1, 1)).reshape(ub_y.shape)

# 真值（test 的 y_test 里最后一个通道）
true_y = y_test[:, target_ch, :]  # scaled
true_y_inv = scaler_y.inverse_transform(true_y.reshape(-1, 1)).reshape(true_y.shape)

print("mean_y_inv:", mean_y_inv.shape, "true_y_inv:", true_y_inv.shape)

# 可选：调用示例里的指标
metrics = adapts_model.compute_metrics()
print(metrics)

import matplotlib.pyplot as plt
import numpy as np

# ===== 可视化一个测试窗口 =====
idx = 0  # 你可以改成 0 ~ len(mean_y_inv)-1
L = context_length
H = forecast_horizon

# 过去真实 y：从 X_test 的 target_ch 取出来（scaled -> inverse）
past_y_scaled = X_test[idx, target_ch, :]  # (L,)
past_y_inv = scaler_y.inverse_transform(past_y_scaled.reshape(-1, 1)).reshape(-1)  # (L,)

# 未来：真值 + 预测均值 + 区间
future_true = true_y_inv[idx]  # (H,)
future_mean = mean_y_inv[idx]  # (H,)
future_lb = lb_y_inv[idx]
future_ub = ub_y_inv[idx]

t_past = np.arange(L)
t_future = np.arange(L, L + H)

plt.figure(figsize=(12, 4))
plt.plot(t_past, past_y_inv, label="History (true)")
plt.plot(t_future, future_true, label="Future (true)")
plt.plot(t_future, future_mean, label="Forecast (mean)")
plt.fill_between(t_future, future_lb, future_ub, alpha=0.2, label="Uncertainty band (lb~ub)")
plt.axvline(L - 1, linestyle="--")
plt.title(f"AdaPTS+Sundial Forecast (test window idx={idx})")
plt.xlabel("Time index within window")
plt.ylabel(target_col)
plt.legend()
plt.tight_layout()
plt.savefig('AdaPTS_betaVAE.png')