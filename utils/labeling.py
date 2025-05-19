from common_imports import *

# 라벨 생성 함수
def add_labels(df, num_days = 5):
    df['Signal_origin'] = np.where((df['Close'].shift(-1) - df['Close']) / df['Close'] >= 0.00, 1, 0)
    df['Signal_trend'] = np.where(df['Close'].rolling(window=num_days).mean().shift(-num_days) > df['Close'], 1, 0)
    return df
