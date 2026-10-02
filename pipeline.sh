#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$script_dir/work/pip-cache}"
export HF_HOME="${HF_HOME:-$script_dir/work/hf-cache}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$script_dir/work/yolo}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/work/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$script_dir/work/cache}"
cd "$script_dir"
if [[ -x "$script_dir/.venv/bin/python" ]]; then
    pipeline_python="$script_dir/.venv/bin/python"
else
    pipeline_python="${PIPELINE_PYTHON:-python3}"
fi
exec "$pipeline_python" -u "$script_dir/tools/bootstrap.py" "$@"
