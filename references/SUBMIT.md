# Kaggle submission paths

The graded bot must expose `agent(obs, config=None)` returning a list of moves.

- **Planet Wars–style scaffold (global candidates + surplus + pulse):** [`submission_v8.py`](../submission_v8.py) — experimental; compare locally with `scripts/eval_compare_v6_v7.py`.
- **当前默认（Planet Wars–style heuristic v7）：** submit [`submission_v7.py`](../submission_v7.py) as a single file, or use the pack script below to produce `dist/main.py` + `dist/submission.tar.gz`.
- **Stable baseline:** [`submission_v6.py`](../submission_v6.py) (rename to `main.py` if the competition requires that exact name).
- **Optional bundle:** use [`kaggle_submit_entry.py`](../kaggle_submit_entry.py) as `main.py` in a tarball that also includes `submission_v6.py` at the same root.

**One-shot pack:** from repo root, `./scripts/package_submission.sh` (defaults to v7). Examples: `./scripts/package_submission.sh submission_v8.py`, `./scripts/package_submission.sh submission_v6.py`. Outputs `dist/main.py`, runs `py_compile`, and `dist/submission.tar.gz` containing only `main.py`.

**Local A/B:** `python3 scripts/eval_compare_v6_v7.py --seeds 0 1 2`

Packaged helpers under [`orbit_wars_bot/`](../orbit_wars_bot/) are for **local** training and evaluation only; they are not required on the server unless you vendor them into the submission archive.
