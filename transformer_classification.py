
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# write dataloader for pytorch
class PriceLoader:
    def __init__(self, path, cov_cols, target_col, window_size, type='train', scale=True):
        path = '/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv'
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.dropna(axis=1, how='any')
        df = df.drop(columns=['time','realtime_clearing_price'])
        df = df[cov_cols + [target_col]]
        
        price = df[target_col]
        labels = pd.DataFrame(np.zeros(price.shape))
        low_bound = 100 # -1
        high_bound = 800 # 2
        labels[price <= low_bound] = 0  # low
        labels[(price > low_bound) & (price <= high_bound)] = 1  # normal
        labels[price > high_bound] = 2  # high
        
        train_start = 192 * 162
        train_number = 96 * 600
        vali_number = 96
        test_number = 192 * 2
        
        if scale:
            X_train = df[train_start:train_start+train_number].drop(columns=['day_ahead_clearing_price']).values
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            
        if type == 'train':
            self.X = df[train_start:train_start+train_number].drop(columns=['day_ahead_clearing_price']).values
            if scale:
                self.X = self.scaler.transform(self.X)
            self.y = labels[train_start:train_start+train_number].values
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        elif type == 'vali':
            self.X = df[train_start+train_number:train_start+train_number+vali_number].drop(columns=['day_ahead_clearing_price']).values
            if scale:
                self.X = self.scaler.transform(self.X)
            self.y = labels[train_start+train_number:train_start+train_number+vali_number].values
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        else: # test
            self.X = df[train_start+train_number+vali_number:train_start+train_number+vali_number+test_number].drop(columns=['day_ahead_clearing_price']).values
            if scale:
                self.X = self.scaler.transform(self.X)
            self.y = labels[train_start+train_number+vali_number:train_start+train_number+vali_number+test_number].values
            self.X, self.y = self.create_sequences(self.X, self.y, window_size, step_size=window_size)
        
        y_history = labels[0:train_start+train_number]
        classes = np.unique(labels)
        # calculate weight for each class
        self.class_weights = compute_class_weight('balanced', classes=classes, y=y_history)
        print(f'Class weights: {self.class_weights}')
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
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, output_size, dim_feedforward=2048, dropout=0.1, d_model=128):
        super(TransformerClassifier, self).__init__()
        self.input_size = input_size
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        src_embeddings = self.embedding(src)
        # import pdb; pdb.set_trace()
        pos_embeddings = self.pos_embedding(torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0),1).to(src.device))
        src = src_embeddings + pos_embeddings
        output = self.transformer_encoder(src)
        B, L, H = output.shape
        output = output.reshape(-1, H)
        output = self.fc_out(self.dropout(self.relu(output)))
        output = output.reshape(B, L, -1)
        return output


path = '/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv'
data = pd.read_csv(path)
data = data.dropna(axis=1, how='any')
print(data.columns)
data = data.drop(columns=['time','realtime_clearing_price'])
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
window_size = 96
scale = False
train_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='train', scale=scale)
vali_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='vali', scale=scale)
test_dataset = PriceLoader(path, cov_cols, target_col, window_size=window_size, type='test', scale=scale)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
vali_dataset = DataLoader(vali_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

input_size = train_dataset.X.shape[2]
output_size = 3

model = TransformerClassifier(input_size=input_size, num_heads=4, num_layers=4, output_size=output_size, dim_feedforward=128, dropout=0.1, d_model=128)


# to GPU
if torch.cuda.is_available():
    model = model.cuda()

# class imbalance handling
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(train_dataset.class_weights))
criterion = criterion.cuda() if torch.cuda.is_available() else criterion

epoch_num = 500
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

best_vali_loss = float('inf')
patience = 50
patience_counter = 0
for epoch in range(epoch_num):
    model.train()
    loss_avg = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.float()
        X_batch = X_batch.cuda() if torch.cuda.is_available() else X_batch
        y_batch = y_batch.long() 
        y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        # calucalte cross entropy loss for each time step and average
        loss = 0
        for t in range(outputs.shape[1]):
            loss += criterion(outputs[:, t, :], y_batch[:, t, :].reshape(-1))
        loss /= outputs.shape[1]
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
            y_batch = y_batch.long()
            y_batch = y_batch.cuda() if torch.cuda.is_available() else y_batch
            outputs = model(X_batch)
            # calucalte cross entropy loss for each time step and average
            loss = 0
            for t in range(outputs.shape[1]):
                loss += criterion(outputs[:, t, :], y_batch[:, t, :].reshape(-1))
            loss /= outputs.shape[1]
            vali_loss_avg += loss.item()
    vali_loss_avg /= len(vali_dataset)
    scheduler.step()
    print(f'Epoch {epoch+1}/{epoch_num}, Train Loss: {loss_avg}, Vali Loss: {vali_loss_avg}, lr: {scheduler.get_last_lr()}')
    if vali_loss_avg < best_vali_loss:
        best_vali_loss = vali_loss_avg
        torch.save(model.state_dict(), 'best_mlp_classifier.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print('Early stopping')
            break
    sys.stdout.flush()

model.load_state_dict(torch.load('best_mlp_classifier.pth'))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch.float().cuda() if torch.cuda.is_available() else X_batch.float())
        y_batch = y_batch.long()
        _, preds = torch.max(outputs, 2)
        all_preds.extend(preds.squeeze().cpu().numpy())
        all_labels.extend(y_batch.squeeze().numpy())  # Convert to 1D array

print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds))

# visualize the predictions vs labels
import matplotlib.pyplot as plt
path = '/data/Xiexin/EPF/shanxi_spot_trading_data-predValue.csv'
df = pd.read_csv(path)        
price = df[target_col]
price = price.reset_index(drop=True)
train_start = 192 * 160
train_number = 96 * 600
vali_number = 96
test_number = 192 * 2
true_prices = price[train_start+train_number+vali_number:train_start+train_number+vali_number+test_number]

# plot prices and predicted labels by time in different y-axis
plt.figure(figsize=(15,5))
plt.plot(true_prices.values, label='True Prices', color='blue')
plt.scatter(range(len(all_preds)), [200 if p==0 else 500 if p==1 else 900 for p in all_preds], label='Predicted Labels', color='red', marker='x')
plt.xlabel('Time Steps')
plt.ylabel('Price / Predicted Label')
plt.legend()
plt.show()
plt.savefig('transformer_classifier_results.png')



