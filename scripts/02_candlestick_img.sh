#!/bin/bash
python run.py \
    --task_name candle_img \
    --output_dir ./data/candle_img \
    --stock_codes AAPL MSFT NVDA GOOG AMZN BRK-B LLY AVGO TSLA JPM WMT UNH V XOM MA PG COST JNJ ORCL HD BAC KO NFLX MRK CVX CRM ADBE AMD PEP TMO \
    --seq_lens 5 20 60 120
