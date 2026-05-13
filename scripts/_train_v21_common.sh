# shellcheck shell=bash
# Source from train_v21_*.sh after setting ROOT and PY.
# Usage: . "$(dirname "$0")/_train_v21_common.sh"

v21_require_interpreter() {
    if ! command -v "${PY}" >/dev/null 2>&1; then
        echo "error: Python interpreter not found: ${PY}" >&2
        echo "  Set PY= e.g. PY=python3.13, PY=python3, or PY=/path/to/python" >&2
        exit 1
    fi
    if ! "${PY}" -c "import msgpack, torch" 2>/dev/null; then
        echo "error: ${PY} is missing dependencies (msgpack, torch)." >&2
        echo "  From repo root: ${PY} -m pip install -U msgpack torch" >&2
        echo "  Or full set:   ${PY} -m pip install -r requirements.txt" >&2
        exit 1
    fi
}
