#!/usr/bin/env bash
# One-shot pack Orbit Wars submission: copy agent file -> dist/main.py, compile, optional tar.gz.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SRC="${1:-submission_v19.py}"
OUT_DIR="${OUT_DIR:-$ROOT/dist}"
MAIN="$OUT_DIR/main.py"
TGZ="$OUT_DIR/submission.tar.gz"
COMPETITION_SLUG="${COMPETITION_SLUG:-orbit-wars}"

if [[ ! -f "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cp -f "$SRC" "$MAIN"
python3 -m py_compile "$MAIN"
echo "OK: $MAIN (syntax check passed)"

( cd "$OUT_DIR" && tar -czf submission.tar.gz main.py )
echo "OK: $TGZ"

echo ""
echo "# Suggested Kaggle CLI (after rules accepted + auth):"
echo "kaggle competitions submit $COMPETITION_SLUG -f $TGZ -m \"$(basename "$SRC" .py) heuristic\""
