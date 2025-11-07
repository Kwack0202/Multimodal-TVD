from common_imports import *

from data.data_preprocessing import *
from stock_prediction.backtesting import Backtesting

## ==================================================
parser = argparse.ArgumentParser(description="Stock prediction")

## ==================================================
## basic config
## ==================================================
parser.add_argument("--task_name", type=str, required=True, default="data_preprocessing", help="task name")

## ==================================================
## step 1. data preprocessing (csv, image)
## ==================================================
parser.add_argument('--output_dir', type=str, default='./data/csv/origin_data/', help='origin data directory')
parser.add_argument('--stock_codes',
                    type=str,
                    nargs='+',
                    default=[
                        "AAPL", "MSFT", "NVDA", "GOOG", "AMZN",
                        "BRK-B", "LLY", "AVGO", "TSLA", "JPM", 
                        "WMT", "UNH", "V", "XOM", "MA", 
                        "PG", "COST", "JNJ", "ORCL", "HD", 
                        "BAC", "KO", "NFLX", "MRK",  "CVX", 
                        "CRM", "ADBE", "AMD", "PEP", "TMO"
                        ],
                    help='List of stock tickers'
                    )
parser.add_argument('--start_day', type=str, default='2010-06-01', help='Start date (format : YYYY-MM-DD)') 
parser.add_argument('--end_day', type=str, default='2025-01-10', help='End date (format : YYYY-MM-DD)')

parser.add_argument('--stock_dir', type=str, default='./data/csv/origin_data/', help='origin data directory')
parser.add_argument('--seq_lens', type=int, nargs='+', default=[5, 20, 60, 120], help='list of top N values for prediction')

parser.add_argument('--bt_name', type=str, default='pred_results', help='merge the pred results with origin stock data')

## ======================================================================================================================================================
## Multi processing Helper
def process_candle_task(stock_code, seq_len, output_dir):
    try:
        # candlestick_image는 stock_codes 리스트를 받음
        candlestick_image(output_dir, [stock_code], [seq_len])
    except Exception as e:
        print(f"[ERROR] {stock_code}-{seq_len}: {e}")
        
## ======================================================================================================================================================
if __name__ == "__main__":
    args = parser.parse_args()
    num_workers = max(1, cpu_count() - 2)
    
    # Step 1: Data Preprocessing
    if args.task_name == "data_preprocessing":
        data_download(args.output_dir, args.stock_codes, args.start_day, args.end_day)

    elif args.task_name == "candle_img":
        freeze_support()

        tasks = [(stock_code, seq_len, args.output_dir) 
                 for stock_code in args.stock_codes 
                 for seq_len in args.seq_lens]

        print(f"[INFO] Total tasks: {len(tasks)} (stocks={len(args.stock_codes)} × seq_lens={len(args.seq_lens)})")
        print(f"[INFO] Using {num_workers} parallel workers")

        with Pool(processes=num_workers) as pool:
            for _ in tqdm(pool.starmap(process_candle_task, tasks), total=len(tasks), desc="Candlestick Image Tasks"):
                pass

    elif args.task_name == "numeric_modal":
        numeric_modal(args.stock_dir, args.output_dir, args.seq_lens)
    
    elif args.task_name == "backtesting":
        exp = Backtesting(args)
        
        if args.bt_name == 'UpDown_Signal':
            exp.UpDown_Signal()
            
        elif args.bt_name == 'BuySell_Signal':
            exp.BuySell_Signal()
        
        elif args.bt_name == 'Simulation':
            exp.Simulation()
        
        elif args.bt_name == 'Benchmark_BuyAndHold':
            exp.Benchmark_BuyAndHold()
            
        elif args.bt_name == 'BacktestSummary':
            exp.BacktestSummary()
            
        elif args.bt_name == 'PlotResults':
            exp.PlotResults()