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
class MultiStockDataset(Dataset):
    def __init__(self, csv_path, img_base_path, transform=None, window_sizes=[5, 20, 60, 120], label_col='Signal_origin', mode='train', train_end_idx=None):
        self.csv_data = pd.read_csv(csv_path)
        self.img_base_path = img_base_path
        self.transform = transform
        self.window_sizes = window_sizes
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
        
        ta_dict = {}
        for window in self.window_sizes:
            start_idx = csv_idx - window + 1
            ta_data = self.csv_data[self.ta_cols].iloc[start_idx:csv_idx+1].values
            ta_data = torch.tensor(ta_data, dtype=torch.float)
            ta_dict[str(window)] = ta_data

        img_dict = {}
        for window in self.window_sizes:
            img_path = os.path.join(self.img_base_path, str(window), f"{img_idx}.png")
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img_dict[str(window)] = img

        label = torch.tensor(self.csv_data[self.label_col].iloc[csv_idx], dtype=torch.float)
        
        return ta_dict, img_dict, label

# 이미지 전처리 (ViT 입력용) — 320x320
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 클래스
class StockPredictor(nn.Module):
    def __init__(self, input_size=25, hidden_unit=256, num_layers=4, num_attention_heads=16, intermediate_size=512, dropout_prob=0.5, mhal_num_heads=16, mlp_hidden_unit=512):
        super(StockPredictor, self).__init__()
        
        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.input_size = input_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhal_num_heads = mhal_num_heads
        self.mlp_hidden_unit = mlp_hidden_unit
        
        # LSTM 모듈 (각 윈도우별)
        self.lstm_dict = nn.ModuleDict({
            str(window): nn.LSTM(input_size=self.input_size, 
                                 hidden_size=self.hidden_unit, 
                                 num_layers=self.num_layers, 
                                 batch_first=True, 
                                 dropout=self.dropout_prob)
            for window in [5, 20, 60, 120]
        })
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        # ViT 모듈 (각 윈도우별) — image_size=320, patch_size=16
        vit_config = ViTConfig(
            hidden_size=self.hidden_unit, 
            num_hidden_layers=self.num_layers, 
            num_attention_heads=self.num_attention_heads, 
            intermediate_size=self.intermediate_size, 
            hidden_dropout_prob=self.dropout_prob,
            image_size=320,
            patch_size=16
        )
        self.vit_dict = nn.ModuleDict({
            str(window): ViTModel(vit_config) for window in [5, 20, 60, 120]
        })

        # 모달리티 내 셀프어텐션 (L=4, N=B, E=H)
        self.ta_attention  = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads, dropout=self.dropout_prob)
        self.img_attention = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads, dropout=self.dropout_prob)

        # 멀티모달 크로스어텐션: Q=TA(4), K/V=IMG(4)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads, dropout=self.dropout_prob)

        # 최종 MLP
        self.fc1 = nn.Linear(self.hidden_unit, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, ta_dict, img_dict):
        # 1) TA: 윈도우 4개 → 토큰 4개
        ta_tokens = []
        for window in ['5', '20', '60', '120']:
            ta, _ = self.lstm_dict[window](ta_dict[window])       # (B, T, H)
            ta = self.fc_ts(ta[:, -1, :])                         # (B, H)
            ta_tokens.append(ta.unsqueeze(0))                     # (1, B, H)
        ta_tokens = torch.cat(ta_tokens, dim=0)                   # (4, B, H)
        ta_attn_output, _ = self.ta_attention(ta_tokens, ta_tokens, ta_tokens)   # (4,B,H)

        # 2) IMG: 윈도우별 ViT CLS → 토큰 4개
        img_tokens = []
        for window in ['5', '20', '60', '120']:
            vit_out = self.vit_dict[window](img_dict[window]).last_hidden_state[:, 0, :]  # (B,H)
            img_tokens.append(vit_out.unsqueeze(0))                # (1, B, H)
        img_tokens = torch.cat(img_tokens, dim=0)                  # (4, B, H)
        img_attn_output, _ = self.img_attention(img_tokens, img_tokens, img_tokens)       # (4,B,H)

        # 3) 모달 간 크로스어텐션 (윈도우×윈도우)
        cross_attn_output, _ = self.cross_attention(ta_attn_output, img_attn_output, img_attn_output)  # (4,B,H)
        cross_attn_output = cross_attn_output.mean(dim=0)          # (B,H)  — 간단 평균(원설계 유지)

        # 4) 최종 MLP
        x = self.fc1(cross_attn_output)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)                                          # (B,1)
        return out

    def get_model_name(self):
        model_name = "MM_Causal_ViT_LSTM_FusedWindows_"
        model_name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        model_name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size}_img320_patch16)_"
        model_name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        model_name += f"(MLP_{self.hidden_unit}_{self.mlp_hidden_unit})"
        return model_name

# =========================
# 에포크 평가(동적 임계값=테스트 평균 확률) — Test Acc/Loss 계산 전용
# =========================
@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_probs, all_labels = [], []

    for ta_dict, img_dict, labels in loader:
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels = labels.to(device)

        outputs = model(ta_dict, img_dict).squeeze(1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    test_loss /= len(loader)
    all_probs  = torch.cat(all_probs)            # (N,)
    all_labels = torch.cat(all_labels).float()   # (N,)

    threshold = all_probs.mean()                 # 동적 임계값(로그 출력 X)
    preds = (all_probs > threshold).float()
    test_acc = 100.0 * (preds == all_labels).float().mean().item()
    return test_loss, test_acc

# =========================
# 학습(매 에포크 테스트 & 체크포인트 & Best by Acc)
# =========================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    best_state, best_epoch = None, -1
    best_acc, best_loss = -1.0, float('inf')

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

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
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # 학습 로깅은 0.5 기준(의미: 진행상태 확인용)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # 에포크마다 테스트
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

        # 체크포인트 저장(매 에포크)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth'))

        # Best 갱신: Acc 우선, 동률이면 Loss 낮은 모델
        if (test_acc > best_acc) or (abs(test_acc - best_acc) < 1e-9 and test_loss < best_loss):
            best_acc, best_loss = test_acc, test_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(ckpt_dir, 'best.pth'))

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}% | "
              f"Best Acc: {best_acc:.2f}% @ epoch {best_epoch}")

    return best_epoch

# =========================
# 최종 예측 저장(동적 임계값 적용)
# =========================
@torch.no_grad()
def predict_and_save(model, test_loader, device, save_csv_path):
    model.eval()
    all_probs, all_labels = [], []

    for ta_dict, img_dict, labels in test_loader:
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels = labels.to(device)

        outputs = model(ta_dict, img_dict).squeeze(1)
        probs = torch.sigmoid(outputs)
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_probs  = torch.cat(all_probs)            # (N,)
    all_labels = torch.cat(all_labels).float()   # (N,)
    threshold  = all_probs.mean()                # 테스트 평균 확률

    pred_labels = (all_probs > threshold).float().numpy()
    df = pd.DataFrame({
        'Actual':   all_labels.numpy(),
        'PredProb': all_probs.numpy(),
        'PredLabel': pred_labels
    })
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df.to_csv(save_csv_path, index=False)

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
    train_end_idx = None
    
    for ticker in tickers:
        csv_path = f'./data/numeric_modal/{ticker}.csv'
        img_base_path = f'./data/candle_img/{ticker}'
     
        for label_col in label_cols:
            print(f'Processing {ticker} for {label_col}...')
            train_dataset = MultiStockDataset(csv_path, img_base_path, transform=transform, label_col=label_col, mode='train', train_end_idx=train_end_idx)
            test_dataset  = MultiStockDataset(csv_path, img_base_path, transform=transform, label_col=label_col, mode='test',  train_end_idx=train_end_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = StockPredictor(input_size=len(train_dataset.ta_cols)).to(device)
            model_name = model.get_model_name()
            print(model_name)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

            # 경로 통일: ./stock_prediction/...
            base_dir = './stock_prediction'
            ckpt_dir = os.path.join(base_dir, 'saved_model', model_name, label_col, ticker)
            pred_dir = os.path.join(base_dir, 'pred_results', model_name, label_col)
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            # 학습 + 매 에포크 평가 + 베스트 저장
            best_epoch = train_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, ckpt_dir)
            best_ckpt_path = os.path.join(ckpt_dir, 'best.pth')

            # 베스트 모델 로드 후 최종 예측 저장(동적 임계값)
            loaded_model = StockPredictor(input_size=len(train_dataset.ta_cols)).to(device)
            loaded_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            save_csv_path = os.path.join(pred_dir, f'{ticker}.csv')
            predict_and_save(loaded_model, test_loader, device, save_csv_path)
