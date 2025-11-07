# visualize_extra_attention_all.py
import os
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTConfig, ViTModel

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom

# =========================
# 데이터셋 (원형과 동일 인터페이스)
# =========================
class MultiStockDataset(Dataset):
    def __init__(self, csv_path, img_base_path, transform=None, window_sizes=[5, 20, 60, 120],
                 label_col='Signal_origin', mode='train', train_end_idx=None):
        self.csv_data = pd.read_csv(csv_path)
        self.img_base_path = img_base_path
        self.transform = transform
        self.window_sizes = window_sizes
        self.label_col = label_col
        self.mode = mode

        exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Signal_origin', 'Signal_trend']
        self.ta_cols = [c for c in self.csv_data.columns if c not in exclude_cols]

        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].fillna(0)
        self.csv_data[self.ta_cols] = self.csv_data[self.ta_cols].replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.csv_data[self.ta_cols] = scaler.fit_transform(self.csv_data[self.ta_cols])

        # 이미지-CSV 오프셋
        self.csv_offset = 119
        self.all_indices = list(range(len(self.csv_data) - self.csv_offset))

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
            ta = self.csv_data[self.ta_cols].iloc[start_idx:csv_idx+1].values
            ta = torch.tensor(ta, dtype=torch.float)
            ta_dict[str(window)] = ta

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
# 이미지 전처리 (학습과 동일)
# =========================
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 모델 (원형과 동일 + helper)
# =========================
class StockPredictor(nn.Module):
    def __init__(self, input_size=25, hidden_unit=256, num_layers=4, num_attention_heads=16,
                 intermediate_size=512, dropout_prob=0.5, mhal_num_heads=16, mlp_hidden_unit=512):
        super().__init__()
        self.hidden_unit = hidden_unit
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.input_size = input_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.mhal_num_heads = mhal_num_heads
        self.mlp_hidden_unit = mlp_hidden_unit

        self.lstm_dict = nn.ModuleDict({
            str(w): nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_unit,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)
            for w in [5, 20, 60, 120]
        })
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        vit_config = ViTConfig(
            hidden_size=self.hidden_unit,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout_prob,
            image_size=320,
            patch_size=16  # 20x20=400 patches
        )
        self.vit_dict = nn.ModuleDict({
            str(w): ViTModel(vit_config) for w in [5, 20, 60, 120]
        })

        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads,
            dropout=self.dropout_prob, batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_unit * 4, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def _mlp_from_fused(self, fused_stack):  # (B,4,D)
        z = fused_stack.reshape(fused_stack.size(0), -1)
        h = self.fc1(z)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)
        return self.fc2(h)  # (B,1)

    def forward_return_fused(self, ta_dict, img_dict):
        fused_outputs = []
        for window in ['5', '20', '60', '120']:
            ta_seq, _ = self.lstm_dict[window](ta_dict[window])     # (B,T,H)
            ta_last = self.fc_ts(ta_seq[:, -1, :]).unsqueeze(1)     # (B,1,D)

            vit_out = self.vit_dict[window](img_dict[window])       # base forward
            vit_tokens = vit_out.last_hidden_state                  # (B,1+P,D)
            vit_patches = vit_tokens[:, 1:, :]                      # (B,P,D)

            attn_output, _ = self.attention_layer(ta_last, vit_patches, vit_patches)
            fused_outputs.append(attn_output.squeeze(1))            # (B,D)

        fused_stack = torch.stack(fused_outputs, dim=1)             # (B,4,D)
        logits = self._mlp_from_fused(fused_stack)                  # (B,1)
        return logits, fused_stack

    def forward(self, ta_dict, img_dict):
        logits, _ = self.forward_return_fused(ta_dict, img_dict)
        return logits

    def get_model_name(self):
        mn = "VER_2_MM_Causal_ViT_LSTM_"
        mn += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        mn += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size}_img320_patch16)_"
        mn += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        mn += f"(MLP_{self.hidden_unit*4}_{self.mlp_hidden_unit})"
        return mn

# =========================
# (1) ViT Attention Rollout
# =========================
@torch.no_grad()
def vit_attention_rollout(vit_outputs):
    # vit_outputs.attentions: list of tensors [L] with shape (B, heads, T, T)
    attns = [a[0].mean(0).cpu().numpy() for a in vit_outputs.attentions]  # (T,T) per layer (mean over heads)
    # 잔여 연결 추가 & 행 정규화
    attns = [a + np.eye(a.shape[-1]) for a in attns]
    attns = [a / a.sum(axis=-1, keepdims=True) for a in attns]
    rollout = attns[0]
    for a in attns[1:]:
        rollout = a @ rollout
    cls_to_patches = rollout[0, 1:]  # (P,)
    return cls_to_patches

# =========================
# (2) LSTM 시계열 사후 어텐션 (Q=fc_ts(h_T), K=h_t)
# =========================
@torch.no_grad()
def compute_temporal_attention_for_window(model: StockPredictor, ta_tensor, window_key: str):
    # ta_tensor: (1,T,F)
    h_seq, _ = model.lstm_dict[window_key](ta_tensor)        # (1,T,H)
    q = model.fc_ts(h_seq[:, -1, :])                          # (1,H)
    k = h_seq.squeeze(0)                                      # (T,H)
    score = (k @ q.squeeze(0)) / math.sqrt(k.size(-1))        # (T,)
    w = torch.softmax(score, dim=0).cpu().numpy()             # (T,)
    return w  # (T,)

# =========================
# (3) Feature×Time saliency (grad×input) — cuDNN eval-backward 회피
# =========================
def feature_time_saliency_grad(model: StockPredictor, ta_dict, img_dict, device, window_key='20'):
    model.eval()
    ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
    img_dict = {k: v.to(device) for k, v in img_dict.items()}

    ta_w = ta_dict[window_key].detach().clone().requires_grad_(True)  # (1,T,F)
    ta_clone = {k: (ta_w if k == window_key else v.detach()) for k, v in ta_dict.items()}

    import torch.backends.cudnn as cudnn
    with cudnn.flags(enabled=False):           # ✅ eval에서도 backward 되도록
        logit = model(ta_clone, img_dict).squeeze(1).mean()
        model.zero_grad(set_to_none=True)
        logit.backward()

    sal = (ta_w.grad * ta_w).abs().detach().cpu().numpy()[0]  # (T,F)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal  # (T,F)

# =========================
# (4) 윈도우 중요도 (fused grad×input)
# =========================
def window_importance_from_fused(model: StockPredictor, ta_dict, img_dict, device):
    model.eval()
    ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
    img_dict = {k: v.to(device) for k, v in img_dict.items()}

    logits, fused = model.forward_return_fused(ta_dict, img_dict)  # fused: (B,4,D)
    fused1 = fused[:1].detach().clone().requires_grad_(True)
    logit = model._mlp_from_fused(fused1).squeeze(1).mean()

    model.zero_grad(set_to_none=True)
    logit.backward()

    imp = (fused1.grad * fused1).abs().mean(dim=-1).detach().cpu().numpy()[0]  # (4,)
    imp = imp / (imp.sum() + 1e-8)
    return {'5': imp[0], '20': imp[1], '60': imp[2], '120': imp[3]}

# =========================
# 유틸: 저장 & 플롯 (+ 축 라벨 간격 자동)
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def overlay_patch_map_on_image(img_path, patch_map, save_path, title='', out_size=512):
    img = Image.open(img_path).convert('RGB')
    img = ImageEnhance.Brightness(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.resize((out_size, out_size))
    img_arr = np.array(img) / 255.0

    P = patch_map.shape[0]
    grid = int(math.sqrt(P))
    assert grid * grid == P, f"Patch count {P} not square."
    attn = patch_map.reshape(grid, grid)
    attn = zoom(attn, out_size / grid, order=1)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_arr)
    plt.imshow(attn, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_tick_positions(T: int):
    """시퀀스 길이 T에 따라 축 라벨 간격 자동 설정:
       T>=60 → 10단위, 20<=T<60 → 5단위, 그 외 모두 표시"""
    if T >= 60:
        step = 10
    elif T >= 20:
        step = 5
    else:
        step = 1
    idxs = list(range(0, T, step))
    labels = [str(i+1) for i in idxs]
    if (T-1) not in idxs:
        idxs.append(T-1)
        labels.append(str(T))
    return idxs, labels

def barplot(values, labels, save_path, title, xlabel='', ylabel='', tick_positions=None, tick_labels=None):
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(values)), values)
    if tick_positions is not None and tick_labels is not None:
        plt.xticks(tick_positions, tick_labels, rotation=0)
    else:
        plt.xticks(range(len(labels)), labels, rotation=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def heatmap(matrix, xlabels, ylabels, save_path, title):
    T = len(ylabels)
    tick_pos, tick_lbl = compute_tick_positions(T)

    # 모든 yticklabels를 '' 처리 후 필요한 위치만 채움
    yticks = [''] * T
    for pos, lbl in zip(tick_pos, tick_lbl):
        yticks[pos] = lbl

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, xticklabels=xlabels, yticklabels=yticks, cmap='jet')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# 시각화 파이프라인(한 종목 + 한 라벨) — 여러 sample_idx 처리
# =========================
def run_for_one(ticker, label_col, args):
    csv_path = args.csv_path or f'./data/numeric_modal/{ticker}.csv'
    img_base_path = args.img_base_path or f'./data/candle_img/{ticker}'
    device = args.device

    ds_train = MultiStockDataset(csv_path, img_base_path, transform=transform, label_col=label_col, mode='train')
    ds_test  = MultiStockDataset(csv_path, img_base_path, transform=transform, label_col=label_col, mode='test')
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    model = StockPredictor(input_size=len(ds_train.ta_cols)).to(device)
    model_name = model.get_model_name()
    ckpt_path = os.path.join(args.ckpt_root, model_name, label_col, ticker, 'best.pth')
    if not os.path.isfile(ckpt_path):
        print(f'[WARN] checkpoint not found, skip: {ckpt_path}')
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    out_dir = ensure_dir(os.path.join(args.out_root, model_name, label_col, ticker))
    print(f'[Save Dir] {out_dir}')

    loader = dl_train if args.sample_mode == 'train' else dl_test
    dataset = ds_train if args.sample_mode == 'train' else ds_test

    N = len(dataset)
    if N == 0:
        print(f'[WARN] empty dataset for {ticker} ({label_col}, {args.sample_mode})')
        return

    if args.sample_idxs:
        sample_indices = [min(max(0, int(s.strip())), N-1) for s in args.sample_idxs.split(',') if s.strip()!='']
    else:
        sample_indices = [0, N//4, N//2, (3*N)//4] if N >= 4 else list(range(N))

    # === 여러 인덱스 순회 ===
    for sample_idx in sample_indices:
        print(f'  - sample_idx = {sample_idx}')
        # 배치에서 해당 sample_idx를 추출
        found_batch = None
        base_index = 0
        for i, (ta_dict, img_dict, labels) in enumerate(loader):
            bsz = labels.size(0)
            if base_index + bsz > sample_idx:
                found_batch = (i, base_index, (ta_dict, img_dict, labels))
                break
            base_index += bsz
        if found_batch is None:
            print(f'  [WARN] sample_idx out of range for {ticker} ({label_col})')
            continue

        i, base_index, batch = found_batch
        ta_dict, img_dict, labels = batch
        offset = sample_idx - base_index

        # 단일 샘플 슬라이스
        for k in ta_dict: ta_dict[k] = ta_dict[k][offset:offset+1].to(device)  # (1,T,F)
        for k in img_dict: img_dict[k] = img_dict[k][offset:offset+1].to(device)
        img_file_idx = dataset.valid_indices[sample_idx]

        # (1) Rollout
        for w in ['5', '20', '60', '120']:
            vit_out = model.vit_dict[w](img_dict[w], output_attentions=True)
            rollout = vit_attention_rollout(vit_out)
            img_path = os.path.join(img_base_path, w, f"{img_file_idx}.png")
            save_path = os.path.join(out_dir, f'rollout_window_{w}_{args.sample_mode}_idx{sample_idx}.png')
            overlay_patch_map_on_image(img_path, rollout, save_path,
                                       title=f'ViT Attention Rollout (W={w}, {args.sample_mode}, idx={sample_idx})')

        # (2) Temporal Attention (+ 축 라벨 간격 자동)
        for w in ['5', '20', '60', '120']:
            wts = compute_temporal_attention_for_window(model, ta_dict[w], w)
            T = len(wts)
            tick_pos, tick_lbl = compute_tick_positions(T)
            save_path = os.path.join(out_dir, f'temporal_attn_window_{w}_{args.sample_mode}_idx{sample_idx}.png')
            barplot(
                values=wts,
                labels=[str(i+1) for i in range(T)],
                save_path=save_path,
                title=f'Temporal Attention (W={w})',
                xlabel='Time step (old→new)',
                ylabel='Weight',
                tick_positions=tick_pos,
                tick_labels=tick_lbl
            )

        # (3) Feature×Time saliency
        ta_cols = ds_train.ta_cols
        for w in ['5', '20', '60', '120']:
            sal = feature_time_saliency_grad(model, ta_dict, img_dict, device, window_key=w)
            save_path = os.path.join(out_dir, f'feat_time_saliency_W{w}_{args.sample_mode}_idx{sample_idx}.png')
            heatmap(sal, xlabels=ta_cols, ylabels=[f't{-len(sal)+i+1}' for i in range(len(sal))],
                    save_path=save_path, title=f'Feature×Time Saliency (grad×input) — W={w}')

        # (4) 윈도우 중요도
        imp = window_importance_from_fused(model, ta_dict, img_dict, device)
        save_path = os.path.join(out_dir, f'window_importance_{args.sample_mode}_idx{sample_idx}.png')
        barplot([imp['5'], imp['20'], imp['60'], imp['120']], ['5','20','60','120'], save_path,
                title='Window Importance (fused grad×input)', xlabel='Window', ylabel='Importance')

    print(f'[OK] {ticker} / {label_col} done.')

# =========================
# 메인: 모든 종목 × 모든 라벨 반복
# =========================
def main():
    parser = argparse.ArgumentParser(description='Extra attention visualizations for all tickers & labels (no training)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--sample_idx', type=int, default=0, help='(deprecated) 단일 인덱스 — sample_idxs가 있으면 무시')
    parser.add_argument('--sample_idxs', type=str, default=None,
                        help='쉼표구분 sample indices, 예: "0,50,200" (지정 없으면 균등 4개 자동 선택)')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--img_base_path', type=str, default=None)
    parser.add_argument('--ckpt_root', type=str, default='./stock_prediction/saved_model')
    parser.add_argument('--out_root', type=str, default='./stock_prediction/attention_plus')
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Device: {args.device}')

    tickers = [
        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
        "BRK-B", "LLY", "AVGO", "TSLA", "JPM",
        "WMT", "UNH", "V", "XOM", "MA",
        "PG", "COST", "JNJ", "ORCL", "HD",
        "BAC", "KO", "NFLX", "MRK", "CVX",
        "CRM", "ADBE", "AMD", "PEP", "TMO"
    ]
    label_cols = ['Signal_origin', 'Signal_trend']

    for label_col in label_cols:
        for ticker in tickers:
            print(f'\n=== Processing {ticker} | {label_col} | mode={args.sample_mode} ===')
            try:
                run_for_one(ticker, label_col, args)
            except Exception as e:
                print(f'[ERROR] {ticker} / {label_col}: {e}')

    print('Done. (no training performed)')

if __name__ == '__main__':
    main()
