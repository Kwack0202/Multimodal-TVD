import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
from tqdm import tqdm
import argparse
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

# 데이터셋 클래스 (TA 데이터만 사용)
class TADataset(Dataset):
    def __init__(self, csv_path, label_col='Signal_origin', mode='train', train_end_idx=None, sequence_length=60):
        self.csv_data = pd.read_csv(csv_path)
        self.label_col = label_col
        self.mode = mode
        self.sequence_length = sequence_length
        
        # 제외 컬럼
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Signal_origin', 'Signal_trend']
        self.ta_cols = [col for col in self.csv_data.columns if col not in exclude_cols]
        
        # 데이터 전처리: 결측값 0으로 채우기 및 정규화
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].fillna(0)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.csv_data[self.ta_cols] = scaler.fit_transform(self.csv_data[self.ta_cols])
        
        # Train-Test 분할
        self.all_indices = list(range(len(self.csv_data)))
        if train_end_idx is None:
            train_end_idx = len(self.all_indices) - 753  # 2021년까지
        
        if mode == 'train':
            self.valid_indices = self.all_indices[:train_end_idx]
        else:  # mode == 'test'
            self.valid_indices = self.all_indices[train_end_idx:]
        
        # 시퀀스 데이터 준비 (딥러닝용)
        self.sequences = []
        self.labels = []
        for idx in self.valid_indices:
            if idx >= self.sequence_length - 1:  # 충분한 시퀀스 길이 보장
                start_idx = idx - self.sequence_length + 1
                seq_data = self.csv_data[self.ta_cols].iloc[start_idx:idx+1].values
                label = self.csv_data[self.label_col].iloc[idx]
                self.sequences.append(seq_data)
                self.labels.append(label)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_data = self.sequences[idx]  # (sequence_length, num_features)
        label = self.labels[idx]
        return torch.tensor(seq_data, dtype=torch.float), torch.tensor(label, dtype=torch.float)

# LSTM 모델
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝 (batch_size, 1)
        return out

# GRU 모델
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Transformer 모델
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, num_heads=4):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_projection(x)
        out = self.transformer(x)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

# 1D-CNN 모델
class CNN1DModel(nn.Module):
    def __init__(self, input_size):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(32 * 15, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# TCN 모델
class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=32):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=4, dilation=4)
        self.fc = nn.Linear(num_channels * 60, 1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 머신러닝 학습 및 평가
def train_eval_ml(model, X_train, y_train, X_test, y_test, model_name, save_path):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # 예측 결과 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_prob})
    results_df.to_csv(save_path, index=False)
    
    return metrics

# 딥러닝 학습 및 평가
def train_eval_dl(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, model_name, save_path):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for seq_data, labels in train_loader:
            seq_data, labels = seq_data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq_data).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    y_true = []
    y_prob = []
    with torch.no_grad():
        for seq_data, labels in test_loader:
            seq_data, labels = seq_data.to(device), labels.to(device)
            outputs = model(seq_data).squeeze(1)
            probs = torch.sigmoid(outputs)
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
    
    y_pred = (np.array(y_prob) > 0.5).astype(float)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # 예측 결과 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_prob})
    results_df.to_csv(save_path, index=False)
    
    torch.cuda.empty_cache()
    return metrics

# 메인 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline study for stock prediction')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length for deep learning models')
    args = parser.parse_args()

    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", 
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
        "WMT", "UNH", "V", "XOM", "MA", 
        "PG", "COST", "JNJ", "ORCL", "HD", 
        "BAC", "KO", "NFLX", "MRK", "CVX",
        "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]
    label_cols = ['Signal_origin', 'Signal_trend']
    
    for ticker in tickers:
        # 데이터 경로
        csv_path = f'./data/csv/TA_csv/{ticker}.csv'

        # 결과 저장 디렉토리
        metrics_dir = './results/baseline'
        results_dir = './Backtesting/pred_results'  # 예측 결과 저장 경로 수정
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        for label_col in label_cols:
            print(f'Processing {ticker} for {label_col}...')
            
            # 데이터셋 준비
            train_dataset = TADataset(csv_path, label_col=label_col, mode='train', sequence_length=args.sequence_length)
            test_dataset = TADataset(csv_path, label_col=label_col, mode='test', sequence_length=args.sequence_length)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # 머신러닝용 데이터 준비
            X_train = train_dataset.sequences.reshape(len(train_dataset), -1)
            y_train = train_dataset.labels
            X_test = test_dataset.sequences.reshape(len(test_dataset), -1)
            y_test = test_dataset.labels
            
            # 모델 리스트
            ml_models = {
                'XGB': XGBClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            dl_models = {
                'LSTM': LSTMModel(input_size=len(train_dataset.ta_cols)),
                'GRU': GRUModel(input_size=len(train_dataset.ta_cols)),
                'Transformer': TransformerModel(input_size=len(train_dataset.ta_cols)),
                '1D-CNN': CNN1DModel(input_size=len(train_dataset.ta_cols)),
                'TCN': TCNModel(input_size=len(train_dataset.ta_cols))
            }
            
            # 디바이스 설정
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 결과 저장용
            metrics_results = []
            
            # 머신러닝 모델 학습/평가
            for model_name, model in ml_models.items():
                print(f'Training {model_name}...')
                save_path = f'{results_dir}/{model_name}/{label_col}/{ticker}.csv'  # 새로운 저장 경로
                
                metrics = train_eval_ml(model, X_train, y_train, X_test, y_test, model_name, save_path)
                metrics_results.append({'model': model_name, **metrics})
                print(f'{model_name} Metrics: {metrics}')
            
            # 딥러닝 모델 학습/평가
            for model_name, model in dl_models.items():
                print(f'Training {model_name}...')
                save_path = f'{results_dir}/{model_name}/{label_col}/{ticker}.csv'  # 새로운 저장 경로
                
                model = model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                metrics = train_eval_dl(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, model_name, save_path)
                metrics_results.append({'model': model_name, **metrics})
                print(f'{model_name} Metrics: {metrics}')
            
            """ # 평가지표 저장
            metrics_df = pd.DataFrame(metrics_results)
            os.makedirs(f'{metrics_dir}/{label_col}', exist_ok=True)
            metrics_df.to_csv(f'{metrics_dir}/{label_col}/{ticker}.csv', index=False)
            print(f'Metrics saved to {metrics_dir}/{label_col}/{ticker}.csv') """