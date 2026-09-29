#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT="${1:-runs/smoke_fresh}"
python -m uji.run prepare --out "$OUT" --profile smoke
python -m uji.run develop --out "$OUT" --device cpu
python -m uji.run final --out "$OUT" --device cpu --allow-test
python -m uji.run final --out "$OUT" --device cpu --allow-test --matched
python scripts/report.py --out "$OUT"
