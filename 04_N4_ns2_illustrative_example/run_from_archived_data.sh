#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_appendix_mplconfig}"
mkdir -p "$ROOT/generated_outputs" "$MPLCONFIGDIR"
export N4_DATA_DIR="$ROOT/data"
export N4_FIGURE_DIR="$ROOT/generated_outputs"

python3 "$ROOT/scripts/export_n4_outcome_table.py"
python3 "$ROOT/scripts/make_n4_pvm_outcome_atlas.py" \
  --input "$ROOT/data/n4_ns2_phase_pvm_complexwp.npz" \
  --output "$ROOT/generated_outputs/n4_ns2_pvm_outcome_atlas"
python3 "$ROOT/scripts/make_complex_receiver_comparison.py"
python3 "$ROOT/scripts/make_collective_receiver_schematic.py" \
  --input "$ROOT/data/n4_ns2_phase_pvm_complexwp.npz" \
  --output "$ROOT/generated_outputs/n4_ns2_collective_receiver"
python3 "$ROOT/audit.py"
