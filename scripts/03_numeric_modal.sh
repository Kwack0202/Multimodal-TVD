#!/bin/bash
python run.py \
    --task_name numeric_modal \
    --stock_dir ./data/csv/origin_data/ \
    --output_dir ./data/numeric_modal/ \
    --seq_lens 5 20 60 120

