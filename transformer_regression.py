
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
# import TransformerClassifier from transformer_classification_withC.py

from gnn_moe_transformer_torch_new import GNNMoETransformerRegressor, ModelFlags, compute_losses, LossWeights

# fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# write dataloader for pytorch
class PriceLoader(Dataset):
    """
    输出：
      X:        (N, L_pred, D)      未来协变量
      y_hist:   (N, L_hist, 1)      过去目标序列
      y_future: (N, L_pred, 1)      未来目标序列（监督目标）
    """
    def __init__(
        self,
        path,
        cov_cols,
        target_col,
        hist_window: int,
        pred_window: int,
        split: str = 'train',            # 'train' / 'vali' / 'test'
        train_start: int = 0,
        train_number: int = 0,
        vali_number: int = 0,
        test_number: int = 0,
        scaler_X=None,
        scaler_y=None,
        scale: bool = True,
        step_size: int = None            # 默认等于 pred_window（不重叠）
    ):
        super().__init__()
        if step_size is None:
            step_size = pred_window

        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df[cov_cols + [target_col]]

        price = df[target_col].values
        X_all = df[cov_cols].values

        self.hist_window = hist_window
        self.pred_window = pred_window

        # ==== 选择时间段 ====
        if split == 'train':
            start = train_start
            length = train_number
        elif split == 'vali':
            start = train_start + train_number
            length = vali_number
        else:  # 'test'
            start = train_start + train_number + vali_number
            length = test_number

        X_seg = X_all[start:start+length]
        y_seg = price[start:start+length]

        if scale:
            assert scaler_X is not None and scaler_y is not None, \
                "scale=True 时必须传入预先在 train 上拟合好的 scaler_X, scaler_y"
            X_seg = scaler_X.transform(X_seg)
            y_seg = scaler_y.transform(y_seg.reshape(-1, 1))
        else:
            y_seg = y_seg.reshape(-1, 1)

        # ==== 构造 (X_future, y_hist, y_future) 序列 ====
        self.X, self.y_hist, self.y_future = self.create_sequences_futureX_pasty(
            X_seg, y_seg,
            hist_window=hist_window,
            pred_window=pred_window,
            step_size=step_size
        )

        self.scaler_X = scaler_X
        self.y_scaler = scaler_y   # 方便外面 inverse_transform 使用

        assert len(self.X) == len(self.y_hist) == len(self.y_future), \
            f"X({len(self.X)}), y_hist({len(self.y_hist)}), y_future({len(self.y_future)}) 长度不一致"

    @staticmethod
    def create_sequences_futureX_pasty(
        data, labels,
        hist_window: int,
        pred_window: int,
        step_size: int = 1
    ):
        """
        data:   (T, D)   协变量
        labels: (T, 1)   目标
        返回:
          X_seq:      (N, pred_window, D)
          y_hist_seq: (N, hist_window, 1)
          y_fut_seq:  (N, pred_window, 1)
        """
        T = len(data)
        X_seq = []
        y_hist_seq = []
        y_fut_seq = []

        for start in range(hist_window, T - pred_window + 1, step_size):
            X_future = data[start:start+pred_window]          # (L_pred, D)
            y_hist   = labels[start-hist_window:start]        # (L_hist, 1)
            y_future = labels[start:start+pred_window]        # (L_pred, 1)

            X_seq.append(X_future)
            y_hist_seq.append(y_hist)
            y_fut_seq.append(y_future)

        return np.array(X_seq), np.array(y_hist_seq), np.array(y_fut_seq)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_hist[idx], self.y_future[idx]
    
    # def create_sequences(self, data, labels, window_size, step_size=1):
    #     sequences = []
    #     seq_labels = []
    #     for start in range(0, len(data) - window_size + 1, step_size):
    #         end = start + window_size
    #         sequences.append(data[start:end])
    #         seq_labels.append(labels[start:end])
    #     return np.array(sequences), np.array(seq_labels)

# Write Transformer to classify the price category
class TransformerRegressor(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, output_size, dim_feedforward=2048, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.input_size = input_size
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, input_size))  # max seq length 5000
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        src = src + self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        B, L, C = output.shape
        output = output.reshape(-1, self.input_size)
        output = self.fc_out(self.dropout(self.relu(output)))
        output = output.reshape(B, L, -1)
        return output


path = '/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv'
data = pd.read_csv(path)
data = data.dropna(axis=1, how='any')
data = data.drop(columns=['time', 'realtime_clearing_price'])
# get index of target column
target_index = data.columns.get_indexer(['day_ahead_clearing_price'])[0]
# put target column to the last
cols = list(data.columns)
cols.append(cols.pop(target_index))
print(len(cols))
target_columns = cols

# Pytorch Dataset and DataLoader
cov_cols = target_columns[:-1]
target_col = target_columns[-1]

train_start = 192 * 160
train_number = 96 * 600
vali_number = 96
test_number = 192 * 2

price_all = data[target_col].values
X_all = data[cov_cols].values

X_train_raw = X_all[train_start:train_start+train_number]
y_train_raw = price_all[train_start:train_start+train_number]

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler().fit(X_train_raw)
scaler_y = StandardScaler().fit(y_train_raw.reshape(-1, 1))

# # ===== 新增：是否打乱特征顺序 =====
# SHUFFLE_FEATURES = True   # 想测试打乱就设 True，不想打乱就 False
# SHUFFLE_SEED = 2024       # 固定种子，保证可复现

# if SHUFFLE_FEATURES:
#     rng = np.random.RandomState(SHUFFLE_SEED)
#     perm = rng.permutation(len(cov_cols))
#     cov_cols = [cov_cols[i] for i in perm]
#     print("[Feature Order] Shuffled with seed", SHUFFLE_SEED)
#     print("[Feature Order] New order (first 10):", cov_cols[:10])
# else:
#     print("[Feature Order] Keep original order")
# # ===== 新增结束 =====

hist_window = 2880
pred_window = 720
scale = True

train_dataset = PriceLoader(
    path, cov_cols, target_col,
    hist_window=hist_window,
    pred_window=pred_window,
    split='train',
    train_start=train_start,
    train_number=train_number,
    vali_number=vali_number,
    test_number=test_number,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    scale=scale,
    step_size=pred_window
)

vali_dataset = PriceLoader(
    path, cov_cols, target_col,
    hist_window=hist_window,
    pred_window=pred_window,
    split='vali',
    train_start=train_start,
    train_number=train_number,
    vali_number=vali_number,
    test_number=test_number,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    scale=scale,
    step_size=pred_window
)

test_dataset = PriceLoader(
    path, cov_cols, target_col,
    hist_window=hist_window,
    pred_window=pred_window,
    split='test',
    train_start=train_start,
    train_number=train_number,
    vali_number=vali_number,
    test_number=test_number,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    scale=scale,
    step_size=pred_window
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
vali_loader = DataLoader(vali_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

input_size = train_dataset.X.shape[2]
output_size = 1

# model = TransformerRegressor(input_size=input_size, num_heads=4, num_layers=4, output_size=output_size, dim_feedforward=128, dropout=0.1)
# model.to('cuda')
flags = ModelFlags(
    use_gnn=True,          # 使用GNN建模协变量
    use_vsn=False,         # 启用特征选择Variable Selection Network
    use_trend=True,        # 启用趋势专家
    use_zone_soft=True,    # 区间分类采用soft限制
    use_value=True,        # 启用值专家
    use_moe=True,          # 启用MoE门控
    graph_mode="learnable" # 学习型图结构
)
model = GNNMoETransformerRegressor(
    input_size=input_size,
    d_model=128, num_heads=4, num_layers=4,
    dim_feedforward=256, dropout=0.1,
    gnn_type="gat", gnn_hidden=64, num_gnn_layers=2,
    num_bins=5, downsample_stride=4, moe_hidden=128,
    flags=flags, posenc="learnable"
).to('cuda')

# class imbalance handling
criterion = nn.MSELoss()
criterion = criterion.cuda() if torch.cuda.is_available() else criterion

epoch_num = 500
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

best_vali_loss = float('inf')
patience = 50
patience_counter = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(epoch_num):
    model.train()
    loss_avg = 0.0

    for X_batch, y_hist_batch, y_future_batch in train_loader:
        X_batch      = X_batch.float().to(device)          # (B,L_pred,D)
        y_hist_batch = y_hist_batch.float().to(device)     # (B,L_hist,1)
        y_future_batch = y_future_batch.float().to(device) # (B,L_pred,1)

        optimizer.zero_grad()

        pred, aux = model(X_batch, y_hist_batch, y_future=y_future_batch, return_aux=True)

        loss_weights = LossWeights(lv=0.5, lz=0.5, lt=0.2, lc=0.1)
        total_loss, metrics, _ = compute_losses(
            y_future_batch, pred, aux, loss_weights,
            hard_clamp=False,
            bin_edges=model.zone_expert.bin_edges,
            trend_target=model.trend_expert.supervised_target(y_future_batch).detach()
        )
        loss = total_loss

        loss_avg += loss.item()
        loss.backward()
        optimizer.step()

    loss_avg /= len(train_loader)

    # ===== 验证 =====
    model.eval()
    vali_loss_avg = 0.0
    with torch.no_grad():
        for X_batch, y_hist_batch, y_future_batch in vali_loader:
            X_batch      = X_batch.float().to(device)
            y_hist_batch = y_hist_batch.float().to(device)
            y_future_batch = y_future_batch.float().to(device)

            outputs = model(X_batch, y_hist_batch, y_future=y_future_batch)  # (B,L_pred,1)
            loss = criterion(
                outputs.reshape(outputs.shape[0], -1),
                y_future_batch.reshape(y_future_batch.shape[0], -1)
            )
            vali_loss_avg += loss.item()

    vali_loss_avg /= len(vali_loader)
    scheduler.step()

    print(f'Epoch {epoch+1}/{epoch_num}, Train Loss: {loss_avg:.6f}, '
          f'Vali Loss: {vali_loss_avg:.6f}, lr: {scheduler.get_last_lr()}')

    # early stopping
    if vali_loss_avg < best_vali_loss:
        best_vali_loss = vali_loss_avg
        torch.save(model.state_dict(), 'best_mlp_regressier.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping')
            break
    sys.stdout.flush()
    
    
with torch.no_grad():
    X_batch, y_hist_batch, y_future_batch = next(iter(train_loader))
    X_batch      = X_batch.float().to(device)
    y_hist_batch = y_hist_batch.float().to(device)
    y_future_batch = y_future_batch.float().to(device)

    pred, aux = model(X_batch, y_hist_batch, y_future=y_future_batch, return_aux=True)

# 如果你启用了 VSN，且希望基于未来窗口的协变量做重要性：
if model.vsn is not None:
    Xt_for_vsn = model.time_linear(X_batch)  # (B,L_pred,d_model)
    w = model.vsn(Xt_for_vsn)                # (B,L_pred,D)
    importance = w.mean(dim=(0,1)).detach().cpu().numpy()
    print("Feature importance (VSN):")
    for i, v in enumerate(importance):
        print(f"  var{i:02d}: {v:.4f}")
        
    plt.bar(range(len(importance)), importance)
    plt.xlabel("Feature index"); plt.ylabel("Importance")
    plt.title("Variable importance (VSN)")
    plt.savefig("feature_importance.png")
        

model.load_state_dict(torch.load('best_mlp_regressier.pth', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_hist_batch, y_future_batch in test_loader:
        X_batch      = X_batch.float().to(device)
        y_hist_batch = y_hist_batch.float().to(device)
        y_future_batch = y_future_batch.float().to(device)

        outputs = model(X_batch, y_hist_batch, y_future=y_future_batch)  # (1,L_pred,1)
        all_preds.append(outputs.cpu().numpy().reshape(-1))
        all_labels.append(y_future_batch.cpu().numpy().reshape(-1))

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# 反缩放
if scale:
    all_preds = scaler_y.inverse_transform(all_preds.reshape(-1, 1))
    all_labels = scaler_y.inverse_transform(all_labels.reshape(-1, 1))

all_preds = all_preds.reshape(-1)
all_labels = all_labels.reshape(-1)

mse = np.mean((all_labels - all_preds) ** 2)
base = np.where(all_labels < 200, 200, all_labels)
mape_adj = np.mean(np.abs((all_labels - all_preds) / base)) * 100
print(f'Test MSE: {mse}, MAPE-ADJ: {mape_adj}')

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(all_labels, label='True Prices', color='blue')
plt.plot(all_preds, label='Predicted Prices', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.title('GNN+MoE Transformer: True vs Predicted Prices')
plt.legend()
plt.savefig('gnn_moe_transformer_results_2_without_vsn.png')
