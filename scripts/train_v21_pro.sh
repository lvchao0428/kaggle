#!/usr/bin/env bash
# v21_pro (~6h-scale: more games per iter; still CPU rollout by default).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3.13}"
# shellcheck source=/dev/null
. "${ROOT}/scripts/_train_v21_common.sh"
v21_require_interpreter

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs runs/v21_pro
nohup "$PY" tools/v21/train_supervisor.py \
  --runs-dir runs/v21_pro \
  --tier pro \
  --submission v20 \
  --iterations 8 \
  --workers 6 \
  --games-per-worker 20 \
  --learner-updates 1 \
  --lr 2e-4 \
  --wait-secs 300 \
  --python "$PY" \
  --opponents v20 v19 v17 \
  > "logs/v21_pro_${STAMP}.log" 2>&1 &
echo "PID=$!  log=logs/v21_pro_${STAMP}.log"
echo "Watch: ./scripts/watch_v21_training.sh runs/v21_pro"
