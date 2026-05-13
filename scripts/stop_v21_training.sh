#!/usr/bin/env bash
# Stop background v21 training: supervisor, rollout workers, and learner for the given tier.
# Matches process command lines that contain runs/v21_{lite,pro,ultra}.
#
# Usage:
#   ./scripts/stop_v21_training.sh              # stop lite + pro + ultra
#   ./scripts/stop_v21_training.sh lite         # only runs/v21_lite
#   ./scripts/stop_v21_training.sh pro ultra    # multiple tiers
#   ./scripts/stop_v21_training.sh all --force  # SIGKILL immediately (no TERM wait)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

FORCE=0
TIERS=()

for arg in "$@"; do
  case "$arg" in
    -9 | --force) FORCE=1 ;;
    lite | pro | ultra | all) TIERS+=("$arg") ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: $0 [lite|pro|ultra|all] ... [--force|-9]" >&2
      exit 1
      ;;
  esac
done

if ((${#TIERS[@]} == 0)); then
  TIERS=(all)
fi

declare -a want_tags=()
for t in "${TIERS[@]}"; do
  if [[ "$t" == all ]]; then
    want_tags=(v21_lite v21_pro v21_ultra)
    break
  fi
  case "$t" in
    lite) want_tags+=(v21_lite) ;;
    pro) want_tags+=(v21_pro) ;;
    ultra) want_tags+=(v21_ultra) ;;
  esac
done

# Dedupe tags (bash 3.2–friendly)
declare -a tags=()
for tag in "${want_tags[@]}"; do
  dup=0
  for e in "${tags[@]}"; do
    if [[ "$e" == "$tag" ]]; then
      dup=1
      break
    fi
  done
  if [[ "$dup" -eq 0 ]]; then
    tags+=("$tag")
  fi
done

stop_tag() {
  local tag="$1"
  local needle pid pids
  echo "Stopping v21 (*${tag}*): train_supervisor, rollout_worker_v21, learner_v21"
  for needle in "train_supervisor.py" "rollout_worker_v21.py" "learner_v21.py"; do
    pids=$(pgrep -f "${needle}.*${tag}" 2>/dev/null || true)
    for pid in $pids; do
      [[ -z "$pid" ]] && continue
      if ((FORCE)); then
        kill -9 "$pid" 2>/dev/null || true
      else
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
  done
}

for tag in "${tags[@]}"; do
  stop_tag "$tag"
done

if ((FORCE)); then
  echo "Done (--force). Check: pgrep -fl train_supervisor.py"
  exit 0
fi

# Second pass: anything still running gets SIGKILL (stuck rollout pool, etc.)
sleep 1
for tag in "${tags[@]}"; do
  for needle in "train_supervisor.py" "rollout_worker_v21.py" "learner_v21.py"; do
    pids=$(pgrep -f "${needle}.*${tag}" 2>/dev/null || true)
    for pid in $pids; do
      [[ -z "$pid" ]] && continue
      echo "  Force killing PID $pid ($needle *${tag}*)"
      kill -9 "$pid" 2>/dev/null || true
    done
  done
done

echo "Done. Check: pgrep -fl train_supervisor.py"
