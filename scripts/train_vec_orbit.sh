#!/usr/bin/env bash
# One-click vec_orbit GPU training (simplified env; not the v21 submission checkpoint).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3.13}"

mkdir -p runs/vec_orbit logs

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-runs/vec_orbit/policy_${STAMP}.pth}"
LOG="${LOG:-logs/vec_orbit_${STAMP}.log}"

# Override with env: BATCH=8192 UPDATES=500 HORIZON=64 SEED=0 DEVICE=cuda
BATCH="${BATCH:-4096}"
UPDATES="${UPDATES:-500}"
HORIZON="${HORIZON:-64}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-1}"

if ! "${PY}" -c "import torch" 2>/dev/null; then
  echo "error: ${PY} cannot import torch" >&2
  exit 1
fi

exec > >(tee -a "${LOG}") 2>&1

echo "Logging to ${LOG}"
echo "Checkpoint: ${OUT}"
echo "NOTE: Output .pth is vec_orbit only — not a submittable NeuralVal / v21 checkpoint."
echo "      Submit: ./scripts/train_v21_lite.sh -> distill_to_numpy_v21.py -> _NEURAL_WEIGHTS_B64 (vec_orbit/PIPELINE.md §5)."

"${PY}" -m vec_orbit.train_loop \
  --device "${DEVICE}" \
  --batch "${BATCH}" \
  --horizon "${HORIZON}" \
  --updates "${UPDATES}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  --out "${OUT}"

echo "Done. Weights: ${OUT}"
echo "This file is not a Kaggle submission; use train_v21 + distill for _NEURAL_WEIGHTS_B64."
