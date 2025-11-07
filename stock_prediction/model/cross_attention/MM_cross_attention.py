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
import math

# =========================
# 데이터셋 클래스
# =========================
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
        
        # 데이터 전처리: 결측값 0으로 채우기 및 정규화 (기존 로직 유지)
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

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 모델 클래스
# =========================
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
        
        # 시계열 인코더 (원형 유지)
        self.lstm_dict = nn.ModuleDict({
            str(window): nn.LSTM(input_size=self.input_size, 
                                 hidden_size=self.hidden_unit, 
                                 num_layers=self.num_layers, 
                                 batch_first=True, 
                                 dropout=self.dropout_prob)
            for window in [5, 20, 60, 120]
        })
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        # UPDATED: ViT 설정에 image_size=320, patch_size=16 명시
        vit_config = ViTConfig(
            hidden_size=self.hidden_unit, 
            num_hidden_layers=self.num_layers, 
            num_attention_heads=self.num_attention_heads, 
            intermediate_size=self.intermediate_size, 
            hidden_dropout_prob=self.dropout_prob,
            image_size=320,            # NEW
            patch_size=16              # NEW -> 20x20=400 patches
        )
        self.vit_dict = nn.ModuleDict({
            str(window): ViTModel(vit_config) for window in [5, 20, 60, 120]
        })

        # UPDATED: 실제 크로스어텐션 (Q: TA last, K/V: ViT 패치)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_unit, 
            num_heads=self.mhal_num_heads,
            dropout=self.dropout_prob,
            batch_first=True  # NEW: (B, L, D)
        )
        
        # 출력 헤드 (원형 유지)
        self.fc1 = nn.Linear(self.hidden_unit * 4, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    # NEW: fused 스택과 로짓을 동시에 얻는 헬퍼 (설명용/중요도 산출용)
    def _mlp_from_fused(self, fused_stack):  # fused_stack: (B,4,D)
        z = fused_stack.reshape(fused_stack.size(0), -1)  # (B,4D)
        h = self.fc1(z)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)  # (B,1)

    def forward_return_fused(self, ta_dict, img_dict):  # NEW
        fused_outputs = []
        for window in ['5', '20', '60', '120']:
            # ----- TA 요약 (Q) -----
            ta_seq, _ = self.lstm_dict[window](ta_dict[window])           # (B, T, H)
            ta_last = self.fc_ts(ta_seq[:, -1, :]).unsqueeze(1)           # (B, 1, D)

            # ----- ViT 토큰 (K/V) -----
            vit_out = self.vit_dict[window](img_dict[window])              # outputs
            vit_tokens = vit_out.last_hidden_state                         # (B, 1+P, D)
            vit_patches = vit_tokens[:, 1:, :]                             # (B, P, D)

            # ----- 크로스어텐션 -----
            attn_output, _ = self.attention_layer(
                ta_last, vit_patches, vit_patches
            )                                                              # (B, 1, D)
            fused_outputs.append(attn_output.squeeze(1))                   # (B, D)

        fused_stack = torch.stack(fused_outputs, dim=1)                    # (B,4,D)
        logits = self._mlp_from_fused(fused_stack)                         # (B,1)
        return logits, fused_stack

    def forward(self, ta_dict, img_dict):
        logits, _ = self.forward_return_fused(ta_dict, img_dict)
        return logits

    def get_model_name(self):
        model_name = "MM_Causal_ViT_LSTM_"
        model_name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        model_name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size}_img320_patch16)_"
        model_name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        model_name += f"(MLP_{self.hidden_unit*4}_{self.mlp_hidden_unit})"
        return model_name

# =========================
# Train / Test 
# =========================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)

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
            preds = (probs > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # 전체 학습 종료 후 마지막 모델만 저장
    final_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(final_state, os.path.join(ckpt_dir, 'final.pth'))

    print(f"Training complete. Final model saved at {os.path.join(ckpt_dir, 'final.pth')}")

@torch.no_grad()
def test_model(model, test_loader, criterion, device):
    model.eval()
    results = []
    for ta_dict, img_dict, labels in test_loader:
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels = labels.to(device)

        outputs = model(ta_dict, img_dict).squeeze(1)
        probabilities = torch.sigmoid(outputs)
        results.extend(zip(labels.cpu().numpy(), probabilities.cpu().numpy()))
    return results

# =========================
# Explainable
# =========================
def visualize_attention_maps(model, data_loader, device, img_base_path, mode='train', sample_idx=0, save_dir='./stock_prediction/attention_maps'):
    import math
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, (ta_dict, img_dict, labels) in enumerate(data_loader):
        if i != sample_idx:
            continue
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels = labels.to(device)

        # 단일 샘플만 사용
        ta_dict = {k: v[0:1] for k, v in ta_dict.items()}
        img_dict = {k: v[0:1] for k, v in img_dict.items()}

        # 이미지 인덱스 (파일명 매칭용)
        img_idx = data_loader.dataset.valid_indices[sample_idx]

        with torch.no_grad():
            for window in ['5', '20', '60', '120']:
                # ===== ViT 어텐션 =====
                vit_model = model.vit_dict[window]
                vit_outputs = vit_model(img_dict[window], output_attentions=True)
                vit_attention = vit_outputs.attentions[-1]       # (B, heads, L, L)
                vit_attention = vit_attention[0].cpu().numpy()   # (heads, L, L)
                num_heads = vit_attention.shape[0]

                # 원본 이미지 로드/보정
                img_path = os.path.join(img_base_path, str(window), f"{img_idx}.png")
                img_raw = Image.open(img_path).convert('RGB')
                img_raw = ImageEnhance.Brightness(img_raw).enhance(1.2)
                img_raw = ImageEnhance.Contrast(img_raw).enhance(1.5)
                img_raw = img_raw.resize((512, 512))
                img_array = np.array(img_raw) / 255.0

                # (A) ViT 전체 어텐션 히트맵
                fig, axes = plt.subplots(2, 4, figsize=(20, 8))
                fig.suptitle(f'ViT Attention Maps (Window={window}, Mode={mode}, Sample={sample_idx})')
                for h in range(min(num_heads, 8)):
                    ax = axes[h // 4, h % 4]
                    sns.heatmap(vit_attention[h, :, :], ax=ax, cmap='jet')
                    ax.set_title(f'Head {h+1}')
                    ax.set_xlabel('Key Tokens')
                    ax.set_ylabel('Query Tokens')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'vit_attention_window_{window}_{mode}_sample_{sample_idx}.png'))
                plt.close()

                # (B) ViT CLS→패치 오버레이 (그리드 자동)
                cls_to_patches = vit_attention[:, 0, 1:]  # (heads, P)
                P = cls_to_patches.shape[-1]
                grid = int(math.sqrt(P))
                assert grid * grid == P, f"Patch count {P} is not a perfect square."

                for h in range(min(num_heads, 8)):
                    attention_map = cls_to_patches[h].reshape(grid, grid)
                    attention_map = zoom(attention_map, 512 / grid, order=1)
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_array)
                    ax.imshow(attention_map, cmap='jet', alpha=0.5)
                    ax.set_title(f'ViT Attention Overlay (Window={window}, Mode={mode}, Head={h+1})')
                    ax.axis('off')
                    plt.savefig(os.path.join(save_dir, f'vit_overlay_window_{window}_{mode}_sample_{sample_idx}_head_{h+1}.png'))
                    plt.close()

                # ===== MHAL(크로스) 패치 주의 =====
                ta_seq, _ = model.lstm_dict[window](ta_dict[window])
                ta_last = model.fc_ts(ta_seq[:, -1, :]).unsqueeze(1)     # (1,1,D)
                vit_tokens = vit_outputs.last_hidden_state               # (1, 1+P, D)
                vit_patches = vit_tokens[:, 1:, :]                       # (1, P, D)

                _, attn_weights = model.attention_layer(
                    ta_last, vit_patches, vit_patches,
                    need_weights=True, average_attn_weights=False
                )  # (B, heads, 1, P)
                mhal_attention = attn_weights[0].cpu().numpy().squeeze(1)  # (heads, P)

                # (C) 헤드 평균 주의 강도 막대
                head_mean = mhal_attention.mean(axis=-1)  # (heads,)
                plt.figure(figsize=(10, 6))
                plt.bar(range(model.mhal_num_heads), head_mean[:model.mhal_num_heads], color='skyblue')
                plt.title(f'MHAL Mean Attention (Window={window}, Mode={mode}, Sample={sample_idx})')
                plt.xlabel('Head')
                plt.ylabel('Mean Attention')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'mhal_attention_window_{window}_{mode}_sample_{sample_idx}.png'))
                plt.close()

                # (D) 특정 헤드 오버레이
                show_head = 0
                attn_map = mhal_attention[show_head].reshape(grid, grid)
                attn_map = zoom(attn_map, 512 / grid, order=1)
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_array)
                ax.imshow(attn_map, cmap='jet', alpha=0.5)
                ax.set_title(f'MHAL Overlay (Window={window}, Mode={mode}, Head={show_head+1})')
                ax.axis('off')
                plt.savefig(os.path.join(save_dir, f'mhal_overlay_window_{window}_{mode}_sample_{sample_idx}_head_{show_head+1}.png'))
                plt.close()

        break  # 한 샘플만
    torch.cuda.empty_cache()

# =========================
# main
# =========================
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
            test_dataset = MultiStockDataset(csv_path, img_base_path, transform=transform, label_col=label_col, mode='test', train_end_idx=train_end_idx)
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) 
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = StockPredictor(input_size=len(train_dataset.ta_cols)).to(device)
            model_name = model.get_model_name()
            print(model_name)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.RAdam(model.parameters(), lr=2e-4)

            base_dir = f'./stock_prediction'
            ckpt_dir = os.path.join(base_dir, 'saved_model', model_name, label_col, ticker)
            pred_dir = os.path.join(base_dir, 'pred_results', model_name, label_col)
            attn_dir = os.path.join(base_dir, 'attention_maps', model_name, label_col, ticker)
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(attn_dir, exist_ok=True)

            train_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, ckpt_dir)
            final_ckpt_path = os.path.join(ckpt_dir, 'final.pth')

            loaded_model = StockPredictor(input_size=len(train_dataset.ta_cols)).to(device)
            loaded_model.load_state_dict(torch.load(final_ckpt_path, map_location=device))
            results = test_model(loaded_model, test_loader, criterion, device)

            results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
            results_df.to_csv(os.path.join(pred_dir, f'{ticker}.csv'), index=False)

            print(f'Visualizing attention maps & window importance for {label_col}...')
            visualize_attention_maps(loaded_model, train_loader, device, img_base_path, mode='train', sample_idx=0, save_dir=attn_dir)
            visualize_attention_maps(loaded_model, test_loader, device, img_base_path, mode='test', sample_idx=0, save_dir=attn_dir)
