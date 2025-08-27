#!/usr/bin/env bash

set -euo pipefail

# Simple, balanced SGD (no XGBoost), modest cap for speed
SGD_EXP="sgd_cap50000"
echo "=== Running $SGD_EXP ==="
PYTHONPATH="." python T7/T7_modeling_service_types.py \
  --cap 50000 \
  --exp_name "${SGD_EXP}"


for cap in 10000 20000 40000 50000; do
    for thr in 1 2 4 8; do
        EXP="xgb_cap${cap}_thr${thr}"
        echo "=== Running $EXP ==="
        PYTHONPATH="." python T7/T7_modeling_service_types.py \
          --only_xgb \
          --cap ${cap} \
          --xgb_threads ${thr} \
          --xgb_rounds 50 \
          --exp_name "${EXP}"
    echo
  done
done

