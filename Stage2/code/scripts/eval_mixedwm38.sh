#!/bin/bash
set -e
cd "$(dirname "$0")/.."

EVAL_JSON=${1:-mixedwm38_eval_result_test.json}

echo "[1/2] Running inference on MixedWM38 test set..."
python test_mixedwm38.py --output "$EVAL_JSON"

echo "[2/2] Calculating accuracy..."
python cal_acc.py --input "$EVAL_JSON"
