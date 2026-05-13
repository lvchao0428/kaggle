#!/usr/bin/env bash
# Background-friendly v21_lite training (~1h-scale defaults; tune --iterations/games).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3.13}"
# shellcheck source=/dev/null
. "${ROOT}/scripts/_train_v21_common.sh"
v21_require_interpreter

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs runs/v21_lite
nohup "$PY" tools/v21/train_supervisor.py \
  --runs-dir runs/v21_lite \
  --tier lite \
  --submission v20 \
  --iterations 6 \
  --workers 4 \
  --games-per-worker 12 \
  --learner-updates 1 \
  --lr 3e-4 \
  --wait-secs 180 \
  --python "$PY" \
  --opponents v20 v19 \
  > "logs/v21_lite_${STAMP}.log" 2>&1 &
echo "PID=$!  log=logs/v21_lite_${STAMP}.log"
echo "Watch live: ./scripts/watch_v21_training.sh runs/v21_lite"
echo "Or:        tail -f logs/v21_lite_${STAMP}.log runs/v21_lite/train.log runs/v21_lite/rollout_progress_w*.jsonl"
