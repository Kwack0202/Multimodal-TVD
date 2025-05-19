#!/bin/bash
python run.py \
    --task_name data_preprocessing \
    --output_dir ./data/csv/origin_data/ \
    --stock_codes AAPL MSFT NVDA GOOG AMZN BRK-B LLY AVGO TSLA JPM WMT UNH V XOM MA PG COST JNJ ORCL HD BAC KO NFLX MRK CVX CRM ADBE AMD PEP TMO \
    --start_day 2010-06-01 \
    --end_day 2024-01-10 
