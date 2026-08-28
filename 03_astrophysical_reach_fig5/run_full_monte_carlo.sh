#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_fig5_mplconfig}"
mkdir -p "$ROOT/recomputed_data" "$ROOT/recomputed_outputs" "$MPLCONFIGDIR"
export FIG5_RESULT_DIR="$ROOT/recomputed_data"
export FIG5_MC_FIGURE_DIR="$ROOT/recomputed_outputs"

python3 "$ROOT/code/run_fixed_modulus_phase_mc.py" --samples "${FIG5_MC_SAMPLES:-600}" --seed "${FIG5_MC_SEED:-20260716}"
export FIG5_MC_SUMMARY="$ROOT/recomputed_data/fixed_modulus_phase_gain_summary.json"
export FIG5_FINAL_FIGURE_DIR="$ROOT/recomputed_outputs"
python3 "$ROOT/code/make_fig5_from_fixed_modulus_mean.py"
