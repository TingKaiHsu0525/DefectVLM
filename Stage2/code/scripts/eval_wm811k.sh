#!/bin/bash
set -e
cd "$(dirname "$0")/.."

EVAL_JSON=${1:-eval_result_WM-811k_test.json}

echo "[1/2] Running inference on WM-811K test set..."
CUDA_VISIBLE_DEVICES=1 python test_wm811k.py --output "$EVAL_JSON"

echo "[2/2] Calculating accuracy..."
CUDA_VISIBLE_DEVICES=1 python cal_acc_wm811k.py --input "$EVAL_JSON"
