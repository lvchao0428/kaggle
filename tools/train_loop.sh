#!/usr/bin/env bash
# Interleaved rollout + learner driver.
# Usage: tools/train_loop.sh <runs_dir> <iters> <workers> <games_per_worker_per_iter>
#
# Each iter: spawn workers (write shards) -> learner consumes shards -> repeat.

set -euo pipefail

RUNS_DIR="${1:-runs/exp1}"
ITERS="${2:-6}"
WORKERS="${3:-4}"
GAMES="${4:-15}"

mkdir -p "${RUNS_DIR}"
echo "=== train_loop dir=${RUNS_DIR} iters=${ITERS} workers=${WORKERS} games/worker=${GAMES} ==="

PY=/opt/local/bin/python3.12

for it in $(seq 1 "${ITERS}"); do
  echo "--- iter ${it}/${ITERS}: rollout ---"
  WEIGHTS_ARG=""
  if [[ -f "${RUNS_DIR}/policy_latest.npz" ]]; then
    WEIGHTS_ARG="--weights ${RUNS_DIR}/policy_latest.npz"
  fi
  ${PY} tools/rollout_worker.py \
      --workers "${WORKERS}" \
      --games-per-worker "${GAMES}" \
      --runs-dir "${RUNS_DIR}" \
      --opponents v9 v10 v11 \
      ${WEIGHTS_ARG}

  echo "--- iter ${it}/${ITERS}: learner ---"
  ${PY} tools/learner.py \
      --runs-dir "${RUNS_DIR}" \
      --updates 1 \
      --wait-secs 5
done

echo "=== train_loop done ==="
