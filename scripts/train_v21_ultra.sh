#!/usr/bin/env bash
# v21_ultra (~24h-scale): heavier net; reduce games/worker if GPU rollout is slow.
# After policy_latest.pth exists, you may add e.g. --opponent-mix "self:0.45,v20:0.35,v19:0.2"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3.13}"
# shellcheck source=/dev/null
. "${ROOT}/scripts/_train_v21_common.sh"
v21_require_interpreter

ARCH_EXTRA=()
if [[ "${V21_ARCHIVE_SHARDS:-0}" == "1" ]]; then
  ARCH_EXTRA=(--archive-shards)
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs runs/v21_ultra
nohup "$PY" tools/v21/train_supervisor.py \
  --runs-dir runs/v21_ultra \
  --tier ultra \
  --submission v20 \
  --iterations 12 \
  --workers 4 \
  --games-per-worker 10 \
  --learner-updates 1 \
  --lr 1.5e-4 \
  --wait-secs 600 \
  --rollout-device cuda \
  --python "$PY" \
  --opponents v20 v19 \
  "${ARCH_EXTRA[@]}" \
  > "logs/v21_ultra_${STAMP}.log" 2>&1 &
echo "PID=$!  log=logs/v21_ultra_${STAMP}.log"
echo "Watch: ./scripts/watch_v21_training.sh runs/v21_ultra"
