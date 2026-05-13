#!/usr/bin/env bash
# Follow v21 training logs: merged train.log + per-worker rollout JSONL lines.
# Usage: ./scripts/watch_v21_training.sh [runs_dir]
# Example: ./scripts/watch_v21_training.sh runs/v21_lite
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${1:-$ROOT/runs/v21_lite}"
cd "$ROOT"
shopt -s nullglob
files=("${RUNS}/train.log" "${RUNS}"/rollout_progress_w*.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "Nothing to tail yet under ${RUNS}/ (no train.log or rollout_progress_w*.jsonl)." >&2
  echo "Start training first, then re-run this script." >&2
  exit 1
fi
echo "Tailing (Ctrl+C to stop): ${files[*]}"
exec tail -n 15 -F "${files[@]}"
