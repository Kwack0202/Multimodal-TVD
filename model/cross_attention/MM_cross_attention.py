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

# 이미지 전처리 (ViT 입력용, 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
        
        self.lstm_dict = nn.ModuleDict({
            str(window): nn.LSTM(input_size=self.input_size, 
                                 hidden_size=self.hidden_unit, 
                                 num_layers=self.num_layers, 
                                 batch_first=True, 
                                 dropout=self.dropout_prob)
            for window in [5, 20, 60, 120]
        })
        self.fc_ts = nn.Linear(self.hidden_unit, self.hidden_unit)

        config = ViTConfig(hidden_size=self.hidden_unit, 
                           num_hidden_layers=self.num_layers, 
                           num_attention_heads=self.num_attention_heads, 
                           intermediate_size=self.intermediate_size, 
                           hidden_dropout_prob=self.dropout_prob)
        self.vit_dict = nn.ModuleDict({
            str(window): ViTModel(config) for window in [5, 20, 60, 120]
        })

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.hidden_unit, num_heads=self.mhal_num_heads)
        
        self.fc1 = nn.Linear(self.hidden_unit * 4, self.mlp_hidden_unit)
        self.bn1 = nn.BatchNorm1d(self.mlp_hidden_unit)
        self.fc2 = nn.Linear(self.mlp_hidden_unit, 1)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, ta_dict, img_dict):
        fused_outputs = []
        
        for window in ['5', '20', '60', '120']:
            ta, _ = self.lstm_dict[window](ta_dict[window])
            ta = self.fc_ts(ta[:, -1, :])
            vit_out = self.vit_dict[window](img_dict[window]).last_hidden_state[:, 0, :]
            ta = ta.unsqueeze(0)
            vit_out = vit_out.unsqueeze(0)
            attn_output, _ = self.attention_layer(ta, vit_out, vit_out)
            attn_output = attn_output.squeeze(0)
            fused_outputs.append(attn_output)

        attn_output = torch.cat(fused_outputs, dim=1)
        attn_output = self.fc1(attn_output)
        attn_output = self.bn1(attn_output)
        attn_output = self.relu(attn_output)
        attn_output = self.dropout(attn_output)
        output = self.fc2(attn_output)
        return output

    def get_model_name(self):
        model_name = "MM_Causal_ViT_LSTM_"
        model_name += f"(LSTM_{self.input_size}_{self.hidden_unit}_{self.num_layers})_"
        model_name += f"(ViT_{self.hidden_unit}_{self.num_layers}_{self.num_attention_heads}_{self.intermediate_size})_"
        model_name += f"(MHAL_{self.hidden_unit}_{self.mhal_num_heads})_"
        model_name += f"(MLP_{self.hidden_unit*4}_{self.mlp_hidden_unit})"
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

# ViT와 MHAL 어텐션 맵 시각화 함수 (저장 경로 수정)
def visualize_attention_maps(model, data_loader, device, img_base_path, mode='train', sample_idx=0, save_dir='./Backtesting/attention_maps'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (ta_dict, img_dict, labels) in enumerate(data_loader):
        if i != sample_idx:
            continue
        ta_dict = {k: v.to(device) for k, v in ta_dict.items()}
        img_dict = {k: v.to(device) for k, v in img_dict.items()}
        labels = labels.to(device)
        
        ta_dict = {k: v[0:1] for k, v in ta_dict.items()}
        img_dict = {k: v[0:1] for k, v in img_dict.items()}
        labels = labels[0:1]
        
        # 이미지 인덱스 계산
        img_idx = data_loader.dataset.valid_indices[sample_idx]
        
        with torch.no_grad():
            for window in ['5', '20', '60', '120']:
                # ViT 어텐션 맵
                vit_model = model.vit_dict[window]
                vit_inputs = img_dict[window]
                vit_outputs = vit_model(vit_inputs, output_attentions=True)
                vit_attention = vit_outputs.attentions[-1]
                vit_attention = vit_attention[0].cpu().numpy()
                
                # MHAL 어텐션 맵
                ta, _ = model.lstm_dict[window](ta_dict[window])
                ta = model.fc_ts(ta[:, -1, :])
                vit_out = vit_outputs.last_hidden_state[:, 0, :]
                ta = ta.unsqueeze(0)
                vit_out = vit_out.unsqueeze(0)
                _, attn_weights = model.attention_layer(ta, vit_out, vit_out)
                mhal_attention = attn_weights[0].cpu().numpy()
                
                # 원본 캔들스틱 이미지 로드 및 선명도 조정 (512x512 해상도)
                img_path = os.path.join(img_base_path, str(window), f"{img_idx}.png")
                img_raw = Image.open(img_path).convert('RGB')
                # 밝기와 대비 조정
                img_enhancer = ImageEnhance.Brightness(img_raw)
                img_raw = img_enhancer.enhance(1.2)
                img_enhancer = ImageEnhance.Contrast(img_raw)
                img_raw = img_enhancer.enhance(1.5)
                # 512x512로 리사이즈
                img_raw = img_raw.resize((512, 512))
                img_array = np.array(img_raw) / 255.0
                
                # ViT 어텐션 맵 시각화 (전체 매트릭스, 197x197)
                num_heads = vit_attention.shape[0]
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
                
                # ViT 어텐션 맵 오버레이 시각화 (512x512로 업샘플링)
                for h in range(min(num_heads, 8)):
                    cls_attention = vit_attention[h, 0, 1:]  # CLS 토큰의 패치별 어텐션 (196,)
                    attention_map = cls_attention.reshape(14, 14)  # 14x14 격자로 재구성
                    # 14x14 → 512x512로 부드럽게 업샘플링 (확대 비율: 512/14 ≈ 36.57)
                    attention_map = zoom(attention_map, 512/14, order=1)  # bilinear 보간
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)  # 정규화
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(img_array)
                    ax.imshow(attention_map, cmap='jet', alpha=0.5)
                    ax.set_title(f'ViT Attention Overlay (Window={window}, Mode={mode}, Head={h+1})')
                    ax.axis('off')
                    plt.savefig(os.path.join(save_dir, f'vit_overlay_window_{window}_{mode}_sample_{sample_idx}_head_{h+1}.png'))
                    plt.close()
                
                # MHAL 어텐션 맵 시각화 (바 플롯)
                plt.figure(figsize=(10, 6))
                mhal_weights = mhal_attention.squeeze()
                if mhal_weights.ndim > 1:
                    mhal_weights = mhal_weights[0, 0]
                plt.bar(range(model.mhal_num_heads), mhal_weights, color='skyblue')
                plt.title(f'MHAL Attention Weights (Window={window}, Mode={mode}, Sample={sample_idx})')
                plt.xlabel('Head')
                plt.ylabel('Attention Weight')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'mhal_attention_window_{window}_{mode}_sample_{sample_idx}.png'))
                plt.close()
        
        break
    
    torch.cuda.empty_cache()

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
        csv_path = f'./data/csv/TA_csv/{ticker}.csv'
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
            optimizer = torch.optim.RAdam(model.parameters(), lr=1e-4)

            train_model(model, train_loader, criterion, optimizer, args.epochs, device)
            save_model(model, f'./saved_model/{model_name}/{label_col}/{ticker}.pth')

            loaded_model = StockPredictor(input_size=len(train_dataset.ta_cols)).to(device)
            loaded_model.load_state_dict(torch.load(f'./saved_model/{model_name}/{label_col}/{ticker}.pth'))
            results = test_model(loaded_model, test_loader, criterion, device)

            results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
            os.makedirs(f'./Backtesting/pred_results/{model_name}/{label_col}/', exist_ok=True)
            results_df.to_csv(f'./Backtesting/pred_results/{model_name}/{label_col}/{ticker}.csv', index=False)

            print(f'Visualizing attention maps for {label_col}...')
            visualize_attention_maps(loaded_model, train_loader, device, img_base_path, mode='train', sample_idx=0, save_dir=f'./Backtesting/attention_maps/{model_name}/{label_col}/{ticker}')
            visualize_attention_maps(loaded_model, test_loader, device, img_base_path, mode='test', sample_idx=0, save_dir=f'./Backtesting/attention_maps/{model_name}/{label_col}/{ticker}')