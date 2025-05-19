#!/bin/bash
python run.py \
    --task_name simulation \
    --output_dir ./Backtesting/simulation \
    --base_dir ./Backtesting/model_results \
    --stock_codes AAPL MSFT NVDA GOOG AMZN BRK-B LLY AVGO TSLA JPM WMT UNH V XOM MA PG COST JNJ ORCL HD BAC KO NFLX MRK CVX CRM ADBE AMD PEP TMO \
    --labels Signal_origin Signal_trend
