#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/recomputed_data"
FIGURES="$ROOT/recomputed_outputs"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_appendix_mplconfig}"
mkdir -p "$DATA" "$FIGURES" "$MPLCONFIGDIR"

python3 "$ROOT/scripts/optimize_n4_complex_workpoint_pvm.py" --output "$DATA"
N4_DATA_DIR="$DATA" python3 "$ROOT/scripts/optimize_n4_complex_workpoint_povm.py"
python3 "$ROOT/scripts/optimize_n4_complexwp_holevo.py" \
  --output "$DATA/n4_complexwp_holevo_summary.json"
python3 "$ROOT/scripts/compute_complex_workpoint_asymptotic_diagnostics.py" \
  --output "$DATA/n4_complex_workpoint_asymptotic_score_diagnostic.json"
python3 "$ROOT/scripts/export_n4_outcome_table.py" \
  --input "$DATA/n4_ns2_phase_pvm_complexwp.npz" \
  --output "$DATA/n4_ns2_pvm_outcomes.csv"
python3 "$ROOT/scripts/make_n4_pvm_outcome_atlas.py" \
  --input "$DATA/n4_ns2_phase_pvm_complexwp.npz" \
  --output "$FIGURES/n4_ns2_pvm_outcome_atlas"
N4_DATA_DIR="$DATA" N4_FIGURE_DIR="$FIGURES" \
  python3 "$ROOT/scripts/make_complex_receiver_comparison.py"
