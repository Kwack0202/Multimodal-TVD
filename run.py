from common_imports import *

from data.data_preprocessing import *
from Backtesting.trading import *
from Backtesting.ploting import *

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

parser.add_argument('--base_dir', type=str, default='./data/csv/origin_data/', help='origin data directory')
parser.add_argument('--labels', type=str, nargs='+', default=['Signal_origin', 'Signal_trend'], help='list of top N values for prediction')


## ======================================================================================================================================================
args = parser.parse_args()

## ======================================================================================================================================================
## Data preparing

# Step 1: Data Preprocessing
if args.task_name == "data_preprocessing":
    data_preprocessing(args.output_dir, args.stock_codes, args.start_day, args.end_day)
    
elif args.task_name == "TA_label":
    numeric_data(args.stock_dir, args.output_dir)

elif args.task_name == "candlestick_img":
    candlestick_image(args.output_dir, args.stock_codes, args.seq_lens)
    
# Step 2: trading
elif args.task_name == "up_down_signal":
    up_down_signal(args.output_dir, args.base_dir, args.stock_codes, args.labels)

elif args.task_name == "buy_sell_signal":
    buy_sell_signal(args.output_dir, args.base_dir, args.stock_codes, args.labels)
    buy_sell_signal_YoY(args.output_dir, args.base_dir, args.stock_codes, args.labels)
    
elif args.task_name == "simulation":
    simulation(args.output_dir, args.base_dir, args.stock_codes, args.labels)
    simulation_YoY(args.output_dir, args.base_dir, args.stock_codes, args.labels)

elif args.task_name == "backtesting":
    backtesting(args.output_dir, args.base_dir, args.stock_codes, args.labels)
    backtesting_YoY(args.output_dir, args.base_dir, args.stock_codes, args.labels)

# Step 3: ploting
elif args.task_name == "ploting":
    # plot_backtesting_metrics(args.base_dir, args.stock_codes, args.labels)
    
    # plot_cumulative_profit(args.base_dir, args.stock_codes, args.labels)
    # plot_cumulative_profit_by_model(args.base_dir, args.stock_codes, args.labels)
    plot_cumulative_profit_baselines_mm(args.base_dir, args.stock_codes, args.labels)
    plot_cumulative_profit_cross_comparisons(args.base_dir, args.stock_codes, args.labels)

    # plot_drawdown(args.base_dir, args.stock_codes, args.labels)
    # plot_trading_signal(args.base_dir, args.stock_codes, args.labels)
    # plot_return_size(args.base_dir, args.stock_codes, args.labels)

    
    
    
    
    