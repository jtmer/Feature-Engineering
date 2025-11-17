
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# import TransformerClassifier from transformer_classification_withC.py

from gnn_moe_transformer_torch_new import GNNMoETransformerRegressor, ModelFlags, compute_losses, LossWeights

# fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# write dataloader for pytorch
class PriceLoader:
    def __init__(self, path, cov_cols, target_col, window_size, type='train', scale=True):
        path = '/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv'
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df[cov_cols + [target_col]]
        
        price = df[target_col]
        
        # TODO 增加验证集和测试集的时间
        train_start = 192 * 160
        train_number = 96 * 600
        
        vali_number = 96
        test_number = 192 * 2
        
        if scale:
            X_train = df[train_start:train_start+train_number].drop(columns=[target_col]).values
            y_train = price[train_start:train_start+train_number].values
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(y_train.reshape(-1, 1))
            
        if type == 'train':
            self.X = df[train_start:train_start+train_number].drop(columns=[target_col]).values
            self.y = price[train_start:train_start+train_number].values
            if scale:
                self.X = self.scaler.transform(self.X)
                self.y = self.y_scaler.transform(self.y.reshape(-1, 1))
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        elif type == 'vali':
            self.X = df[train_start+train_number:train_start+train_number+vali_number].drop(columns=[target_col]).values
            self.y = price[train_start+train_number:train_start+train_number+vali_number].values
            if scale:
                self.X = self.scaler.transform(self.X)
                self.y = self.y_scaler.transform(self.y.reshape(-1, 1))
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        else: # test
            self.X = df[train_start+train_number+vali_number:train_start+train_number+vali_number+test_number].drop(columns=[target_col]).values
            self.y = price[train_start+train_number+vali_number:train_start+train_number+vali_number+test_number].values
            if scale:
                self.X = self.scaler.transform(self.X)
                self.y = self.y_scaler.transform(self.y.reshape(-1, 1))
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        
        assert len(self.X) == len(self.y), f"Length of X ({len(self.X)}) and y ({len(self.y)}) do not match."

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def create_sequences(self, data, labels, window_size, step_size=1):
        sequences = []
        seq_labels = []
        for start in range(0, len(data) - window_size + 1, step_size):
            end = start + window_size
            sequences.append(data[start:end])
            seq_labels.append(labels[start:end])
        return np.array(sequences), np.array(seq_labels)

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

window_size = 96
scale = True
train_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='train', scale=scale)
vali_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='vali', scale=scale)
test_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='test', scale=scale)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
vali_dataset = DataLoader(vali_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
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
for epoch in range(epoch_num):
    model.train()
    loss_avg = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.float()
        X_batch = X_batch.cuda() if torch.cuda.is_available() else X_batch
        y_batch = y_batch.float() 
        y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch
        optimizer.zero_grad()
        
        # outputs = model(X_batch)
        # loss = criterion(outputs.reshape(outputs.shape[0], -1), y_batch.reshape(y_batch.shape[0], -1))
        pred, aux = model(X_batch, y_batch, return_aux=True)
        loss_weights = LossWeights(lv=0.5, lz=0.5, lt=0.2, lc=0.1)
        total_loss, metrics, _ = compute_losses(
            y_batch, pred, aux, loss_weights,
            hard_clamp=False,
            bin_edges=model.zone_expert.bin_edges,
            trend_target=model.trend_expert.supervised_target(y_batch).detach()
        )
        loss = total_loss
        
        loss_avg += loss.item()
        loss.backward()
        optimizer.step()
    loss_avg /= len(train_loader)
    model.eval()
    vali_loss_avg = 0
    with torch.no_grad():
        for X_batch, y_batch in vali_dataset:
            X_batch = X_batch.float()
            X_batch = X_batch.cuda() if torch.cuda.is_available() else X_batch
            y_batch = y_batch.float()
            y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch
            outputs = model(X_batch)
            loss = criterion(outputs.reshape(outputs.shape[0], -1), y_batch.reshape(y_batch.shape[0], -1))
            vali_loss_avg += loss.item()
    vali_loss_avg /= len(vali_dataset)
    scheduler.step()
    print(f'Epoch {epoch+1}/{epoch_num}, Train Loss: {loss_avg}, Vali Loss: {vali_loss_avg}, lr: {scheduler.get_last_lr()}')
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
    
    
# 可视化
with torch.no_grad():
    X_batch, y_batch = next(iter(train_loader))
    X_batch = X_batch.float().to('cuda')
    y_batch = y_batch.float().to('cuda')
    pred, aux = model(X_batch, y=y_batch, return_aux=True)

# 提取变量权重
if model.vsn is not None:
    w = model.vsn(model.time_linear(X_batch))  # (B,L,D)
    importance = w.mean(dim=(0,1)).detach().cpu().numpy()
    print("Feature importance (VSN):")
    for i, v in enumerate(importance):
        print(f"  var{i:02d}: {v:.4f}")
        
    plt.bar(range(len(importance)), importance)
    plt.xlabel("Feature index"); plt.ylabel("Importance")
    plt.title("Variable importance (VSN)")
    plt.savefig("feature_importance.png")
        

model.load_state_dict(torch.load('best_mlp_regressier.pth'))
model.eval()
all_preds = []
all_preds = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch.float().cuda() if torch.cuda.is_available() else X_batch.float())
        y_batch = y_batch.float()
        # record prediction
        all_preds.append(outputs.cpu().numpy().reshape(-1))
all_labels = test_dataset.y.reshape(-1)
# retransform the predictions and labels
if scale:
    all_preds = train_dataset.y_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1))
    all_labels = train_dataset.y_scaler.inverse_transform(all_labels.reshape(-1, 1))
all_preds = np.array(all_preds).reshape(-1)
all_labels = all_labels.reshape(-1)
# calculate MSE and MAPE
mse = np.mean((all_labels - all_preds) ** 2)
base = np.where(all_labels < 200, 200, all_labels)
mape_adj = np.mean(np.abs((all_labels - all_preds) / base)) * 100
print(f'Test MSE: {mse}, MAPE-ADJ: {mape_adj}')

# visualize the predictions vs labels
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(all_labels, label='True Prices', color='blue')
plt.plot(all_preds, label='Predicted Prices', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.title('Transformer Regression: True vs Predicted Prices')
plt.legend()
plt.show()
plt.savefig('transformer_regression_results_2_without_vsn.png')


