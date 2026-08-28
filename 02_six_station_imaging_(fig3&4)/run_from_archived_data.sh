#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_fig34_mplconfig}"
mkdir -p "$MPLCONFIGDIR" "$ROOT/generated_outputs"

python3 "$ROOT/code/make_maintext_fig3_fig4.py"
python3 "$ROOT/audit.py"
