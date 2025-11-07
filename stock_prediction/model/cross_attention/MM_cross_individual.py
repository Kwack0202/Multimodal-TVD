import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTConfig, ViTModel
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

# =========================
# 데이터셋 클래스 (활성 윈도우만 로딩)
# =========================
class MultiStockDataset(Dataset):
    def __init__(self, csv_path, img_base_path, transform=None,
                 window_sizes=[5], label_col='Signal_origin',
                 mode='train', train_end_idx=None):
        self.csv_data = pd.read_csv(csv_path)
        self.img_base_path = img_base_path
        self.transform = transform
        self.window_sizes = window_sizes  # <- 활성 윈도우만 사용
        self.label_col = label_col
        self.mode = mode

        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Signal_origin', 'Signal_trend']
        self.ta_cols = [col for col in self.csv_data.columns if col not in exclude_cols]

        # 간단 전처리 (기존 로직 유지)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].fillna(0)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.csv_data[self.ta_cols] = scaler.fit_transform(self.csv_data[self.ta_cols])

        # 인덱스 매핑 (CSV 119 ↔ 이미지 0.png)
        self.csv_offset = 119
        self.all_indices = list(range(len(self.csv_data) - self.csv_offset))

        if train_end_idx is None:
            train_end_idx = len(self.all_indices) - 753
        self.valid_indices = self.all_indices[:train_end_idx] if mode == 'train' else self.all_indices[train_end_idx:]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        csv_idx = img_idx + self.csv_offset

        # TA 시퀀스(활성 윈도우만)
        ta_dict = {}
        for window in self.window_sizes:
            start_idx = csv_idx - window + 1
            ta = self.csv_data[self.ta_cols].iloc[start_idx:csv_idx+1].values
            ta = torch.tensor(ta, dtype=torch.float)
            ta_dict[str(window)] = ta

        # 이미지(활성 윈도우만)
        img_dict = {}
        for window in self.window_sizes:
            img_path = os.path.join(self.img_base_path, str(window), f"{img_idx}.png")
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img_dict[str(window)] = img

        label = torch.tensor(self.csv_data[self.label_col].iloc[csv_idx], dtype=torch.float)
        return ta_dict, img_dict, label

# =========================
# 이미지 전처리 (ViT 입력용) — 320x320
# =========================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 모델 (활성 윈도우만 구성)
# =========================
class StockPredictor(nn.Module):
    def __init__(self, windows=[5], input_size=25, hidden_unit=256, num_layers=4,
                 num_attention_heads=16, intermediate_size=512, dropout_prob=0.5,
                 mhal_num_heads=16, mlp_hidden_unit=512):
        super().__init__()
        self.windows = [int(w) for w in windows]
        self.windows_str = [str(w) for w in self.windows]

        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.input_size = input_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhal_num_heads = mhal_num_heads
        self.mlp_hidden_unit = mlp_hidden_unit

        # LSTM (활성 윈도우만)
        self.lstm_dict = nn.ModuleDict({
            w: nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit,
                       num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)
            for w in self.windows_str
        })
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        # ViT (활성 윈도우만)
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
            w: ViTModel(vit_config) for w in self.windows_str
        })

        # TA last (Q) vs ViT patches (K,V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads,
            dropout=self.dropout_prob, batch_first=True
        )

        # 윈도우 개수에 맞춰 MLP 입력 차원 조정
        self.fc1 = nn.Linear(self.hidden_unit * len(self.windows), self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, ta_dict, img_dict):
        fused = []
        for w in self.windows_str:
            # TA → 마지막 스텝 요약
            ta_seq, _ = self.lstm_dict[w](ta_dict[w])         # (B,T,H)
            q = self.fc_ts(ta_seq[:, -1, :]).unsqueeze(1)     # (B,1,H)

            # ViT → 패치 토큰
            vit_out = self.vit_dict[w](img_dict[w])
            vit_tokens = vit_out.last_hidden_state             # (B,1+P,H)
            kv = vit_tokens[:, 1:, :]                          # (B,P,H)

            # Cross-Attn (Q vs K/V)
            fused_w, _ = self.cross_attn(q, kv, kv)            # (B,1,H)
            fused.append(fused_w.squeeze(1))                   # (B,H)

            # 메모리 관리(선택): vit_out 참조 해제
            del vit_out

        z = torch.cat(fused, dim=1)                            # (B, H*|W|)
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)                                      # (B,1)
        return out

    def get_model_name(self):
        ws = "-".join([str(w) for w in self.windows])
        name = f"MM_Causal_ViT_LSTM_OnlyW[{ws}]_"
        name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size}_img320_patch16)_"
        name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        name += f"(MLP_{self.hidden_unit*len(self.windows)}_{self.mlp_hidden_unit})"
        return name

# =========================
# 학습 함수 (요청한 간단 버전)
# =========================
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss, correct, total = 0.0, 0, 0
        for ta_dict, img_dict, labels in train_loader:
            ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
            img_dict = {k: v.to(device) for k, v in img_dict.items()}
            labels  = labels.to(device)

            optimizer.zero_grad()
            outputs = model(ta_dict, img_dict).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    torch.cuda.empty_cache()

# =========================
# 테스트 함수 (요청한 간단 버전)
# =========================
@torch.no_grad()
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    results = []
    for ta_dict, img_dict, labels in test_loader:
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels  = labels.to(device)

        outputs = model(ta_dict, img_dict).squeeze(1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        # CSV 저장용 (정답, 확률)
        results.extend(zip(labels.cpu().numpy(), probs.cpu().numpy()))

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    torch.cuda.empty_cache()
    return results

# =========================
# 메인: Ablation (윈도우 하나만 사용하는 4회 실험)
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single-window ablation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
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

    ablation_windows = [5, 20, 60, 120]  # ← 윈도우 하나만 사용하는 4회 실험

    for w in ablation_windows:
        print(f"\n==== Ablation: ONLY window={w} ====\n")

        for ticker in tickers:
            csv_path = f'./data/numeric_modal/{ticker}.csv'
            img_base_path = f'./data/candle_img/{ticker}'

            for label_col in label_cols:
                print(f'Processing {ticker} for {label_col} (only {w})...')

                train_dataset = MultiStockDataset(csv_path, img_base_path, transform=transform,
                                                  window_sizes=[w], label_col=label_col, mode='train',
                                                  train_end_idx=train_end_idx)
                test_dataset  = MultiStockDataset(csv_path, img_base_path, transform=transform,
                                                  window_sizes=[w], label_col=label_col, mode='test',
                                                  train_end_idx=train_end_idx)

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = StockPredictor(windows=[w], input_size=len(train_dataset.ta_cols)).to(device)
                model_name = model.get_model_name()
                print(model_name)

                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

                # ✅ 경로를 기존과 동일하게 통일
                base_dir = './stock_prediction'
                ckpt_dir = os.path.join(base_dir, 'saved_model', model_name, label_col, ticker)
                pred_dir = os.path.join(base_dir, 'pred_results', model_name, label_col)
                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)

                # 학습 (에포크 중간 테스트 없음)
                train_model(model, train_loader, criterion, optimizer, args.epochs, device)

                # 최종 체크포인트 저장 (기존 구조 유지)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pth'))

                # 테스트 & CSV 저장 (기존 구조 유지)
                results = test_model(model, test_loader, criterion, device)
                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                results_df.to_csv(os.path.join(pred_dir, f'{ticker}.csv'), index=False)