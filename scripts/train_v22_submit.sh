#!/usr/bin/env bash
# v22: vec_orbit GPU train -> distill into submission_v22_* NeuralVal (needs real-env shards).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python3.13}"

# Real rollout shards (state_feat), e.g. from ./scripts/train_v21_lite.sh → runs/v21_lite/
SHARDS_DIR="${SHARDS_DIR:?Set SHARDS_DIR e.g. export SHARDS_DIR=runs/v21_lite}"

TIER="${TIER:-lite}"
SUB="submission_v22_${TIER}.py"
OUT_VEC="${OUT_VEC:-runs/vec_orbit/policy_v22_${TIER}.pth}"
OUT_B64="${OUT_B64:-neural_weights_v22_${TIER}.b64.txt}"

if [[ ! -f "$SUB" ]]; then
  echo "Generating $SUB ..."
  "${PY}" tools/gen_v22_submissions.py --tier "${TIER}"
fi

echo "=== 1) vec_orbit (GPU by default) -> $OUT_VEC ==="
OUT="${OUT_VEC}" "${ROOT}/scripts/train_vec_orbit.sh"

echo "=== 2) bridge distill -> $OUT_B64 ==="
"${PY}" tools/distill_vec_bridge_v22.py \
  --vec-checkpoint "${OUT_VEC}" \
  --shards-dir "${SHARDS_DIR}" \
  --target-submission "${SUB}" \
  --out-b64 "${OUT_B64}"

echo ""
echo "Done. Paste contents of ${OUT_B64} into ${SUB} variable _NEURAL_WEIGHTS_B64"
echo "Then: ${PY} -m py_compile ${SUB}"
