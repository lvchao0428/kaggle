#!/usr/bin/env python3
"""Generate submission_v21_lite.py, submission_v21_pro.py, submission_v21_ultra.py
from submission_v20.py by patching docstring, version label, and NeuralVal widths.

NeuralVal layout (NumPy, same matmul as v20):
  h1 = ReLU(W1 @ x + b1),  W1: (h1, 14), b1: (h1,)
  h2 = ReLU(W2 @ h1 + b2), W2: (h2, h1), b2: (h2,)
  out = tanh(W3 @ h2 + b3), W3: (1, h2), b3: (1,)

Run from repo root: python3.13 tools/gen_v21_submissions.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# (h1, h2) hidden sizes for NeuralVal
TIER_DIMS = {
    "lite": (64, 32),
    "pro": (128, 64),
    "ultra": (192, 96),
}

TIER_DOC = {
    "lite": (
        "Orbit Wars v21_lite - v20 lineage + compact NeuralVal (64→32→1). "
        "Regenerate weights with tools/distill_to_numpy_v21.py after RL."
    ),
    "pro": (
        "Orbit Wars v21_pro - v20 lineage + medium NeuralVal (128→64→1). "
        "Regenerate weights with tools/distill_to_numpy_v21.py after RL."
    ),
    "ultra": (
        "Orbit Wars v21_ultra - v20 lineage + wide NeuralVal (192→96→1). "
        "Heavier per-step cost; prefer offline RL teacher + distill to lite/pro for submit."
    ),
}


def patch_neural_val_init(src: str, h1: int, h2: int) -> str:
    """Replace W1/W2/W3 rng shapes and bias sizes in NeuralVal.__init__."""
    # self.W1 = rng.normal(0, 0.2, (64, self.N_FEAT))
    src = re.sub(
        r"self\.W1 = rng\.normal\(0, 0\.2, \(\d+, self\.N_FEAT\)\)",
        f"self.W1 = rng.normal(0, 0.2, ({h1}, self.N_FEAT))",
        src,
        count=1,
    )
    src = re.sub(
        r"self\.b1 = np\.zeros\(\d+, dtype=np\.float32\)",
        f"self.b1 = np.zeros({h1}, dtype=np.float32)",
        src,
        count=1,
    )
    src = re.sub(
        r"self\.W2 = rng\.normal\(0, 0\.2, \(\d+, \d+\)\)",
        f"self.W2 = rng.normal(0, 0.2, ({h2}, {h1}))",
        src,
        count=1,
    )
    src = re.sub(
        r"self\.b2 = np\.zeros\(\d+, dtype=np\.float32\)",
        f"self.b2 = np.zeros({h2}, dtype=np.float32)",
        src,
        count=1,
    )
    src = re.sub(
        r"self\.W3 = rng\.normal\(0, 0\.2, \(1, \d+\)\)",
        f"self.W3 = rng.normal(0, 0.2, (1, {h2}))",
        src,
        count=1,
    )
    return src


def strip_neural_weights_b64(src: str) -> str:
    """Set _NEURAL_WEIGHTS_B64 to empty string."""
    return re.sub(
        r'^_NEURAL_WEIGHTS_B64 = .+$',
        '_NEURAL_WEIGHTS_B64 = ""  # run tools/distill_to_numpy_v21.py to fill',
        src,
        flags=re.MULTILINE,
    )


def patch_module_docstring(src: str, title_paragraph: str) -> str:
    """Replace only the **first** module docstring (from file start)."""
    if not src.startswith('"""'):
        return src
    close = src.find('"""', 3)
    if close < 0:
        return src
    inner = src[3:close]
    parts = inner.split("\n\n", 1)
    rest = parts[1] if len(parts) == 2 else ""
    new_inner = title_paragraph.strip() + ("\n\n" + rest if rest else "")
    return '"""' + new_inner + '"""' + src[close + 3 :]


def generate_one(tier: str, dry_run: bool) -> None:
    h1, h2 = TIER_DIMS[tier]
    in_path = ROOT / "submission_v20.py"
    out_path = ROOT / f"submission_v21_{tier}.py"
    text = in_path.read_text(encoding="utf-8")
    text = patch_module_docstring(text, TIER_DOC[tier])
    text = strip_neural_weights_b64(text)
    text = patch_neural_val_init(text, h1, h2)
    # Unique header comment after imports block is heavy; docstring is enough.
    if dry_run:
        print(f"Would write {out_path} (NeuralVal {h1}→{h2}→1)")
        return
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} (NeuralVal {h1}→{h2}→1)")
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", str(out_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tier",
        choices=list(TIER_DIMS) + ["all"],
        default="all",
        help="Which submission to generate (default: all three).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tiers = list(TIER_DIMS) if args.tier == "all" else [args.tier]
    for t in tiers:
        generate_one(t, args.dry_run)


if __name__ == "__main__":
    main()
