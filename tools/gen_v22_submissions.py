#!/usr/bin/env python3
"""Generate submission_v22_lite.py, submission_v22_pro.py, submission_v22_ultra.py
from ``tools/templates/v20_monolith_for_v21_codegen.py`` (same NeuralVal widths as v21).

v22: fill NeuralVal via tools/distill_vec_bridge_v22.py after vec_orbit + real shards.

Run from repo root: python3.13 tools/gen_v22_submissions.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

TIER_DIMS = {
    "lite": (64, 32),
    "pro": (128, 64),
    "ultra": (192, 96),
}

TIER_DOC = {
    "lite": (
        "Orbit Wars v22_lite - v20 lineage + NeuralVal (64→32→1). "
        "Weights: vec_orbit critic + bridge encoder → tools/distill_vec_bridge_v22.py (see vec_orbit/PIPELINE.md v22)."
    ),
    "pro": (
        "Orbit Wars v22_pro - v20 lineage + NeuralVal (128→64→1). "
        "Fill with tools/distill_vec_bridge_v22.py after vec_orbit + shards."
    ),
    "ultra": (
        "Orbit Wars v22_ultra - v20 lineage + NeuralVal (192→96→1). "
        "Fill with tools/distill_vec_bridge_v22.py."
    ),
}


def patch_neural_val_init(src: str, h1: int, h2: int) -> str:
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
    return re.sub(
        r'^_NEURAL_WEIGHTS_B64 = .+$',
        '_NEURAL_WEIGHTS_B64 = ""  # run tools/distill_vec_bridge_v22.py to fill',
        src,
        flags=re.MULTILINE,
    )


def patch_module_docstring(src: str, title_paragraph: str) -> str:
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
    in_path = ROOT / "tools" / "templates" / "v20_monolith_for_v21_codegen.py"
    out_path = ROOT / f"submission_v22_{tier}.py"
    text = in_path.read_text(encoding="utf-8")
    text = patch_module_docstring(text, TIER_DOC[tier])
    text = strip_neural_weights_b64(text)
    text = patch_neural_val_init(text, h1, h2)
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
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tiers = list(TIER_DIMS) if args.tier == "all" else [args.tier]
    for t in tiers:
        generate_one(t, args.dry_run)


if __name__ == "__main__":
    main()
