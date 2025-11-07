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
# 데이터셋 클래스
# =========================
class MultiStockDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_base_path,
        transform=None,
        window_sizes=[5, 20, 60, 120],
        label_col='Signal_origin',
        mode='train',
        train_end_idx=None,
        modalities=('ta','img') 
    ):
        self.csv_data = pd.read_csv(csv_path)
        self.img_base_path = img_base_path
        self.transform = transform
        self.window_sizes = window_sizes
        self.label_col = label_col
        self.mode = mode
        self.modalities = set(modalities)

        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Signal_origin', 'Signal_trend']
        self.ta_cols = [c for c in self.csv_data.columns if c not in exclude_cols]

        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].fillna(0)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.csv_data[self.ta_cols] = scaler.fit_transform(self.csv_data[self.ta_cols])

        # 인덱스 매핑
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

        ta_dict, img_dict = {}, {}

        if 'ta' in self.modalities:
            for w in self.window_sizes:
                start_idx = csv_idx - w + 1
                ta = self.csv_data[self.ta_cols].iloc[start_idx:csv_idx+1].values
                ta_dict[str(w)] = torch.tensor(ta, dtype=torch.float)

        if 'img' in self.modalities:
            for w in self.window_sizes:
                img_path = os.path.join(self.img_base_path, str(w), f"{img_idx}.png")
                img = Image.open(img_path).convert('RGB')
                if self.transform: img = self.transform(img)
                img_dict[str(w)] = img

        label = torch.tensor(self.csv_data[self.label_col].iloc[csv_idx], dtype=torch.float)
        return ta_dict, img_dict, label

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 모델 클래스
# =========================
class StockPredictorAblation(nn.Module):
    def __init__(
        self,
        windows=[5,20,60,120],
        modality_mode='ta_only',  # 'ta_only' or 'img_only'
        input_size=25,
        hidden_unit=256,
        num_layers=4,
        num_attention_heads=16,
        intermediate_size=512,
        dropout_prob=0.5,
        mhal_num_heads=16,
        mlp_hidden_unit=512
    ):
        super().__init__()
        self.windows = [int(w) for w in windows]
        self.wstr = [str(w) for w in self.windows]
        self.modality_mode = modality_mode

        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.input_size = input_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhal_num_heads = mhal_num_heads
        self.mlp_hidden_unit = mlp_hidden_unit

        self.use_ta  = (modality_mode == 'ta_only')
        self.use_img = (modality_mode == 'img_only')

        if self.use_ta:
            # LSTM per window + 윈도우 간 self-attn
            self.lstm_dict = nn.ModuleDict({
                s: nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit,
                           num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)
                for s in self.wstr
            })
            self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)
            self.ta_attention = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads, dropout=self.dropout_prob)

        if self.use_img:
            # ViT per window + 윈도우 간 self-attn
            vit_cfg = ViTConfig(
                hidden_size=self.hidden_unit,
                num_hidden_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_dropout_prob=self.dropout_prob,
                image_size=320,
                patch_size=16
            )
            self.vit_dict = nn.ModuleDict({ s: ViTModel(vit_cfg) for s in self.wstr })
            self.img_attention = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads, dropout=self.dropout_prob)

        # 최종 MLP (윈도우 평균(H) → mlp)
        self.fc1 = nn.Linear(self.hidden_unit, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, ta_dict, img_dict):
        if self.use_ta:
            # 윈도우별 TA 토큰 4개 → (4,B,H)
            ta_tokens = []
            for s in self.wstr:
                ta_seq, _ = self.lstm_dict[s](ta_dict[s])   # (B,T,H)
                ta_vec = self.fc_ts(ta_seq[:, -1, :])       # (B,H)
                ta_tokens.append(ta_vec.unsqueeze(0))       # (1,B,H)
            ta_tokens = torch.cat(ta_tokens, dim=0)         # (4,B,H)
            ta_attn_out, _ = self.ta_attention(ta_tokens, ta_tokens, ta_tokens)  # (4,B,H)
            fused = ta_attn_out.mean(dim=0)                  # (B,H)

        elif self.use_img:
            # 윈도우별 ViT CLS 4개 → (4,B,H)
            img_tokens = []
            for s in self.wstr:
                vit_out = self.vit_dict[s](img_dict[s]).last_hidden_state[:, 0, :]  # (B,H)
                img_tokens.append(vit_out.unsqueeze(0))                               # (1,B,H)
            img_tokens = torch.cat(img_tokens, dim=0)                                 # (4,B,H)
            img_attn_out, _ = self.img_attention(img_tokens, img_tokens, img_tokens) # (4,B,H)
            fused = img_attn_out.mean(dim=0)                                          # (B,H)

        else:
            raise ValueError("modality_mode must be 'ta_only' or 'img_only'.")

        x = self.fc1(fused)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)  # (B,1)
        return out

    def get_model_name(self):
        tag = "Ablation_TAOnly" if self.use_ta else "Ablation_IMGOnly"
        name = f"MM_Causal_ViT_LSTM_FusedWindows_{tag}_"
        name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        if self.use_img:
            name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size}_img320_patch16)_"
        else:
            name += f"(ViT_skipped)_"
        name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        name += f"(MLP_{self.hidden_unit}_{self.mlp_hidden_unit})"
        return name

# =========================
# Train / Test 
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

        results.extend(zip(labels.cpu().numpy(), probs.cpu().numpy()))

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    torch.cuda.empty_cache()
    return results

# =========================
# main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modality ablation (TA-only / IMG-only)')
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

    ablation_modes = ['ta_only', 'img_only']  # 두 실험 자동 실행

    for mode in ablation_modes:
        print(f"\n==== Ablation: {mode.upper()} ====\n")
        modalities = ('ta',) if mode == 'ta_only' else ('img',)

        for ticker in tickers:
            csv_path = f'./data/numeric_modal/{ticker}.csv'
            img_base_path = f'./data/candle_img/{ticker}'

            for label_col in label_cols:
                print(f'Processing {ticker} for {label_col} ({mode})...')

                train_dataset = MultiStockDataset(
                    csv_path, img_base_path, transform=transform if 'img' in modalities else None,
                    window_sizes=[5,20,60,120], label_col=label_col, mode='train',
                    train_end_idx=train_end_idx, modalities=modalities
                )
                test_dataset = MultiStockDataset(
                    csv_path, img_base_path, transform=transform if 'img' in modalities else None,
                    window_sizes=[5,20,60,120], label_col=label_col, mode='test',
                    train_end_idx=train_end_idx, modalities=modalities
                )

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
                test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = StockPredictorAblation(
                    windows=[5,20,60,120],
                    modality_mode=mode,
                    input_size=len(train_dataset.ta_cols)
                ).to(device)
                model_name = model.get_model_name()
                print(model_name)

                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

                base_dir = './stock_prediction'
                ckpt_dir = os.path.join(base_dir, 'saved_model', model_name, label_col, ticker)
                pred_dir = os.path.join(base_dir, 'pred_results', model_name, label_col)
                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(pred_dir, exist_ok=True)

                train_model(model, train_loader, criterion, optimizer, args.epochs, device)

                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pth'))

                results = test_model(model, test_loader, criterion, device)
                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                results_df.to_csv(os.path.join(pred_dir, f'{ticker}.csv'), index=False)
