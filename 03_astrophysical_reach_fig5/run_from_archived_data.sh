#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_fig5_mplconfig}"
mkdir -p "$ROOT/generated_outputs" "$MPLCONFIGDIR"
export FIG5_MC_SUMMARY="$ROOT/data/fixed_modulus_phase_gain_summary.json"

python3 "$ROOT/code/make_fig5_from_fixed_modulus_mean.py"
python3 "$ROOT/audit.py"
