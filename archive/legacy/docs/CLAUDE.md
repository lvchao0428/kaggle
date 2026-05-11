# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A bot for the Kaggle **Orbit Wars** competition: continuous-2D real-time strategy where fleets capture orbiting planets around a central sun. The game rules (board, physics, combat, observation/action format) live in [README.md](README.md) and are best read before touching strategy code.

Two much deeper docs already exist — prefer reading them over guessing, and keep them in sync when you change things:

- **[AGENTS.md](AGENTS.md)** — full version-by-version changelog (v6 → v13), what each version added, win-rate tables, and the rationale behind each design choice. Update this whenever a new `submission_vN.py` is created or an existing one changes its strategy.
- **[ONBOARDING.md](ONBOARDING.md)** — quick-start: env install, eval commands, replay viewer, RL pipeline walkthrough.

## Python interpreter

Use `/opt/local/bin/python3.12` explicitly for anything that touches `kaggle-environments` — that's the interpreter that has the competition env and torch installed. `train_loop.sh` hardcodes this path. A bare `python3` may not have the deps.

## Submission model — one file, self-contained

Each `submission_vN.py` at repo root is a **standalone Kaggle submission**. Kaggle requires a single file exposing `agent(obs, config=None) -> list[[from_planet_id, angle_rad, num_ships], ...]`. Rules this imposes on anything that ships:

- No imports from `orbit_wars_bot/` or `tools/`. Everything gets inlined.
- Neural net weights are base64-encoded in `_NEURAL_WEIGHTS_B64` and decoded at import time into a NumPy MLP. No PyTorch at inference.
- Per-turn wall budget ≈ 1s; MCTS and sim budgets are tuned in `PhasePolicy` / `PHASE_TABLE` to stay under ~920ms.

**Creating a new version** is `cp submission_v13.py submission_v14.py`, edit in place, then `scripts/eval_head2head.py --a v14 --b v13` to validate. Don't refactor the older versions — they're kept as reproducible baselines for regression comparison.

`submission_v13.py` is the current best and is organized into numbered regions 0–11 (constants, data classes, `GameState`, `Snapshot`, `PhasePolicy`+`PHASE_TABLE`, scoring, sim, Planners, `MCTSEngine`, `NeuralVal`, `PlanArbiter`, `agent()`). The fastest tuning entry point is `PHASE_TABLE` in region 4.

## Common commands

```bash
# Head-to-head eval (double-seated by default, 2 games per seed)
/opt/local/bin/python3.12 scripts/eval_head2head.py --a v13 --b v12 --seeds 0-9

# vs built-in random (single seat)
/opt/local/bin/python3.12 scripts/eval_head2head.py --a v13 --b random --seeds 0-4 --no-swap

# HTML replay (opens in browser)
/opt/local/bin/python3.12 scripts/replay.py --a v13 --b v12 --seed 42

# Package a single submission into dist/main.py + dist/submission.tar.gz
./scripts/package_submission.sh submission_v13.py
```

Quick smoke after a code change: `scripts/eval_head2head.py --a vNEW --b random --seeds 0-4 --no-swap` should return 5:0 wins. Anything less means the new version is broken.

A single v13-vs-v13 game is ~30s on Apple silicon; a 10-seed double-seated run (20 games) is ~6 min.

## RL training pipeline (`tools/`)

Produces new `_NEURAL_WEIGHTS_B64` to drop into the next submission version. The pipeline is:

```
imitation_pretrain.py (BC warm-start from v11)
  → rollout_worker.py (mp self-play → msgpack shards)
  → learner.py (PPO, reads shards, writes .pth + policy_latest.npz)
  → distill_to_numpy.py (pth → base64 NumPy MLP matching NeuralVal's 14→64→32→1 shape)
  → paste base64 into submission_vN.py
```

`tools/train_loop.sh <runs_dir> <iters> <workers> <games_per_worker>` drives iters of (rollout → learner). The student MLP shape in `distill_to_numpy.py` **must** match `NeuralVal` in the submission — that's the only contract for drop-in weight swaps.

## Directory map

| Path | Role |
|------|------|
| `submission_v*.py` | Standalone Kaggle submissions, one per version |
| `scripts/` | Local eval / replay / packaging |
| `tools/` | RL training pipeline (not shipped) |
| `orbit_wars_bot/` | Older heuristic+RL scaffold (v6 era); **not** imported by current submissions |
| `references/` | Read-only reference implementations (halite3, lux-s2, planet-wars) |
| `planet-wars/` | 2010 Planet Wars champion's Common Lisp source — design inspiration for v11's four penalty functions, see AGENTS.md |
| `runs/` | RL artifacts: shards, checkpoints, distilled weights |
| `replays/` | Generated HTML replays (gitignored-ish, safe to delete) |
| `dist/` | Output of `package_submission.sh` |
| `submission.py`, `submission.py` | Notebook `elite_bot_v5` dump used as a local opponent baseline |

## Conventions worth keeping

- Chinese is the working language for strategy notes in `AGENTS.md` / `ONBOARDING.md` and many code comments. Code identifiers and docstrings in `submission_v*.py` are English.
- Win-rate claims in `AGENTS.md` are expected to cite exact seeds and swap convention (e.g. "0-9 双座位 = 20 局"). If you add a new row, match the format.
- Evaluation uses *double-seated* runs by default (each agent plays both player 0 and player 1 per seed). Single-seat results are noisier and should be labeled `--no-swap`.
