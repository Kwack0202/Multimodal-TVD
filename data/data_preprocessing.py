from common_imports import *
from utils.candlestick_img import *
from utils.technical_indicator import *

# ============================================
'''
step 1.origin data download
'''
def data_download(output_dir, stock_codes, start_day, end_day):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for stock_code in tqdm(stock_codes, desc="Downloading stock data"):
        
        try:
            stock_data = pd.DataFrame(fdr.DataReader(stock_code, start_day, end_day))
            stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            stock_data = stock_data.reset_index()
            stock_data.to_csv(os.path.join(output_dir, f"{stock_code}.csv"), encoding='utf-8', index=False)
            
        except Exception as e:
            print(f"Failed to download data for {stock_code}: {e}")
            
'''
step 2.candlestick image (visual modal.)
'''
def candlestick_image(output_dir, stock_codes, seq_lens):
    
    for stock_code in tqdm(stock_codes):
        stock_data = pd.read_csv(f"./data/csv/origin_data/{stock_code}.csv")
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data = stock_data[stock_data['Date'] <= '2024-01-01']

        mask = (stock_data['Date'].dt.year == 2011)
        filtered_stock_data = stock_data[mask]

        if not filtered_stock_data.empty:
            idx = filtered_stock_data.index[0] - max(seq_lens) + 1
            if idx < 0:
                idx = 0
            stock_data = stock_data.iloc[idx:].reset_index(drop=True)

        route_new = os.path.join(output_dir, stock_code)
        print(f"\n캔들스틱 차트 이미지 생성 : [ {stock_code} ]")

        for seq_len in seq_lens:
            for i in tqdm(range(0, len(stock_data) - max(seq_lens) + 1)):
                if seq_len == max(seq_lens):
                    candlestick_data = stock_data.iloc[i:i + seq_len]
                else:
                    candlestick_data = stock_data.iloc[i + max(seq_lens) - seq_len:i + max(seq_lens)]
                candlestick_data = candlestick_data.reset_index(drop=True)

                seq_path = os.path.join(route_new, str(seq_len))
                os.makedirs(seq_path, exist_ok=True)

                fig = plot_candles(candlestick_data, trend_line=False)
                fig.savefig(os.path.join(seq_path, f'{i}.png'), dpi=100)
                plt.close(fig)
                
'''
step 3.Technical indicators (Numerical modal.)
'''
def numeric_modal(stock_dir, output_dir, seq_lens):
    csv_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {stock_dir}")
        return

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        stock_code = os.path.splitext(csv_file)[0] 
        input_path = os.path.join(stock_dir, csv_file)
        
        try:
            stock_data = pd.read_csv(input_path)
            
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # 기술적 지표 추가
            stock_data = calculate_indicators(stock_data)
            
            # 라벨 추가
            stock_data['Signal_origin'] = np.where((stock_data['Close'].shift(-1) - stock_data['Close']) / stock_data['Close'] >= 0.00, 1, 0)
            stock_data['Signal_trend'] = np.where(stock_data['Close'].rolling(window=5).mean().shift(-5) > stock_data['Close'], 1, 0)
             
            mask = (stock_data['Date'].dt.year == 2011)
            filtered_stock_data = stock_data[mask]

            if not filtered_stock_data.empty:
                idx = filtered_stock_data.index[0] - max(seq_lens) + 1
                if idx < 0:
                    idx = 0
                stock_data = stock_data.iloc[idx:].reset_index(drop=True)
                
            stock_data = stock_data[stock_data['Date'] < '2024-01-01']
            stock_data = stock_data.reset_index(drop=True)
                   
            # 결과 저장
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{stock_code}.csv")
            stock_data.to_csv(output_path, encoding='utf-8', index=False)
                    
        except Exception as e:
            print(f"Failed to process {stock_code}: {e}")