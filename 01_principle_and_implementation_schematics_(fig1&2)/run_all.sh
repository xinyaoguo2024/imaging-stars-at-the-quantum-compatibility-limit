#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_fig12_mplconfig}"
mkdir -p "$ROOT/generated_outputs" "$MPLCONFIGDIR"

python3 "$ROOT/code/make_fig1_principle_schematic.py"
python3 "$ROOT/code/make_fig2_implementation_schematic.py"
python3 "$ROOT/audit.py"
