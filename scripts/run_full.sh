#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Usage: bash scripts/run_full.sh runs/full_v1 cuda
OUT="${1:-runs/full_v1}"
DEVICE="${2:-auto}"
python -m uji.run prepare --out "$OUT" --profile full
python -m uji.run develop --out "$OUT" --device "$DEVICE"
python -m uji.run final --out "$OUT" --device "$DEVICE" --allow-test
python -m uji.run final --out "$OUT" --device "$DEVICE" --allow-test --matched
python scripts/report.py --out "$OUT"
