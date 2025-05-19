import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTConfig, ViTModel
import pandas as pd
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.ndimage import zoom

# 데이터셋 클래스
class MultiStockDatasetSingleWindow(Dataset):
    def __init__(self, csv_path, img_base_path, transform=None, window_size=5, label_col='Signal_origin', mode='train', train_end_idx=None):
        self.csv_data = pd.read_csv(csv_path)
        self.img_base_path = img_base_path
        self.transform = transform
        self.window_size = window_size
        self.label_col = label_col
        self.mode = mode
        
        # 제외 컬럼
        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Signal_origin', 'Signal_trend']
        self.ta_cols = [col for col in self.csv_data.columns if col not in exclude_cols]
        
        # 데이터 전처리: 결측값 0으로 채우기 및 정규화
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].fillna(0)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.csv_data[self.ta_cols] = scaler.fit_transform(self.csv_data[self.ta_cols])
        
        # CSV는 2010년 마지막 119일치 데이터 포함 -> CSV 인덱스 119가 0.png(2011년 첫 거래일)에 대응
        self.csv_offset = 119
        self.all_indices = list(range(len(self.csv_data) - self.csv_offset))
        
        # Train-Test 분할 (인덱스 기준)
        if train_end_idx is None:
            train_end_idx = len(self.all_indices) - 753
        
        if mode == 'train':
            self.valid_indices = self.all_indices[:train_end_idx]
        else:
            self.valid_indices = self.all_indices[train_end_idx:]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        csv_idx = img_idx + self.csv_offset
        
        # TA 데이터 (단일 윈도우)
        start_idx = csv_idx - self.window_size + 1
        window_str = str(self.window_size)
        ta_data = self.csv_data[self.ta_cols].iloc[start_idx:csv_idx+1].values
        ta_data = torch.tensor(ta_data, dtype=torch.float)
        ta_dict = {window_str: ta_data}

        # 이미지 데이터 (단일 윈도우)
        img_path = os.path.join(self.img_base_path, window_str, f"{img_idx}.png")
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        img_dict = {window_str: img}

        # 라벨
        label = torch.tensor(self.csv_data[self.label_col].iloc[csv_idx], dtype=torch.float)
        
        return ta_dict, img_dict, label

# 이미지 전처리 (ViT 입력용, 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 클래스
class StockPredictorSingleWindow(nn.Module):
    def __init__(self, window_size, input_size=25, hidden_unit=256, num_layers=4, num_attention_heads=16, intermediate_size=512, dropout_prob=0.5, mhal_num_heads=16, mlp_hidden_unit=512):
        super(StockPredictorSingleWindow, self).__init__()
        
        self.window_size = str(window_size)  # 예: '5', '20', '60', '120'
        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.input_size = input_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhal_num_heads = mhal_num_heads
        self.mlp_hidden_unit = mlp_hidden_unit
        
        # LSTM 모듈 (지정된 윈도우만)
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_unit, 
                            num_layers=self.num_layers, 
                            batch_first=True, 
                            dropout=self.dropout_prob)
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        # ViT 모듈 (지정된 윈도우만)
        config = ViTConfig(hidden_size=self.hidden_unit, 
                           num_hidden_layers=self.num_layers, 
                           num_attention_heads=self.num_attention_heads, 
                           intermediate_size=self.intermediate_size, 
                           hidden_dropout_prob=self.dropout_prob)
        self.vit = ViTModel(config)

        # Cross Attention
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads)
        
        # 최종 MLP (단일 윈도우이므로 hidden_unit 입력)
        self.fc1 = nn.Linear(self.hidden_unit, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, ta_dict, img_dict):
        # 단일 윈도우의 LSTM 처리
        ta, _ = self.lstm(ta_dict[self.window_size])
        ta = self.fc_ts(ta[:, -1, :])  # 마지막 타임스텝 출력

        # 단일 윈도우의 ViT 처리
        vit_out = self.vit(img_dict[self.window_size]).last_hidden_state[:, 0, :]

        # Cross Attention
        ta = ta.unsqueeze(0)  # [1, batch_size, hidden_unit]
        vit_out = vit_out.unsqueeze(0)  # [1, batch_size, hidden_unit]
        attn_output, _ = self.attention_layer(ta, vit_out, vit_out)
        attn_output = attn_output.squeeze(0)  # [batch_size, hidden_unit]

        # 최종 MLP
        output = self.fc1(attn_output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

    def get_model_name(self):
        model_name = f"MM_Causal_ViT_LSTM_Window{self.window_size}_"
        model_name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        model_name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size})_"
        model_name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        model_name += f"(MLP_{self.hidden_unit}_{self.mlp_hidden_unit})"
        return model_name

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for ta_dict, img_dict, labels in train_loader:
            ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
            img_dict = {k: v.to(device) for k, v in img_dict.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(ta_dict, img_dict).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    
    torch.cuda.empty_cache()

# 테스트 함수
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    results = []

    with torch.no_grad():
        for ta_dict, img_dict, labels in test_loader:
            ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
            img_dict = {k: v.to(device) for k, v in img_dict.items()}
            labels = labels.to(device)

            outputs = model(ta_dict, img_dict).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            results.extend(zip(labels.cpu().numpy(), probabilities.cpu().numpy()))

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    torch.cuda.empty_cache()
    return results

# 모델 저장 함수 (경로 유지)
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# 메인 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock prediction model training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
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
    window_sizes = [5, 20, 60, 120]
    train_end_idx = None
    
    for window_size in window_sizes:
        for ticker in tickers:
            csv_path = f'./data/csv/TA_csv/{ticker}.csv'
            img_base_path = f'./data/candle_img/{ticker}'
            
            for label_col in label_cols:
                print(f'Processing {ticker} for {label_col} with window {window_size}...')
                train_dataset = MultiStockDatasetSingleWindow(
                    csv_path, img_base_path, transform=transform, window_size=window_size, 
                    label_col=label_col, mode='train', train_end_idx=train_end_idx
                )
                test_dataset = MultiStockDatasetSingleWindow(
                    csv_path, img_base_path, transform=transform, window_size=window_size, 
                    label_col=label_col, mode='test', train_end_idx=train_end_idx
                )
                
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = StockPredictorSingleWindow(
                    window_size=window_size, input_size=len(train_dataset.ta_cols)
                ).to(device)
                model_name = model.get_model_name()
                print(model_name)

                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

                train_model(model, train_loader, criterion, optimizer, args.epochs, device)
                save_model(model, f'./saved_model/{model_name}/{label_col}/{ticker}.pth')

                loaded_model = StockPredictorSingleWindow(
                    window_size=window_size, input_size=len(train_dataset.ta_cols)
                ).to(device)
                loaded_model.load_state_dict(torch.load(f'./saved_model/{model_name}/{label_col}/{ticker}.pth'))
                results = test_model(loaded_model, test_loader, criterion, device)

                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                os.makedirs(f'./Backtesting/pred_results/{model_name}/{label_col}/', exist_ok=True)
                results_df.to_csv(f'./Backtesting/pred_results/{model_name}/{label_col}/{ticker}.csv', index=False)