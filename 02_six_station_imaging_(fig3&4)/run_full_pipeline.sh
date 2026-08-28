#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/quantum_telescope_fig34_mplconfig}"
export FIG2_RUN_ROOT="${FIG2_RUN_ROOT:-$ROOT/recomputed_12seed_runs/100ms}"
mkdir -p "$MPLCONFIGDIR" "$FIG2_RUN_ROOT"

# The archived per-band receiver cache is deliberately reused across seeds.
# To redesign it, run optimize_finite_ns_povm_per_band.py and
# build_per_band_finite_cache.py first; see their --help output.
"$ROOT/run_12seed_eff002_no_core.sh"

export FIG2_REPRESENTATIVE_RUN_ROOT="$FIG2_RUN_ROOT"
export FIG2_SELECTED_RUN_ROOT="$FIG2_RUN_ROOT"
python3 "$ROOT/select_representative_seed.py"
python3 "$ROOT/code/make_sm_paired_ns10_statistics.py" \
  --run-root "$FIG2_RUN_ROOT" \
  --result-dir "$ROOT/results" \
  --skip-figure \
  --receiver-cache "$ROOT/receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.npz"

# Produce only the current manuscript layouts.  Historical wide-layout
# plotting entry points and their rendered artifacts are intentionally absent
# from this clean archive.
python3 "$ROOT/code/make_maintext_fig3_fig4.py"
python3 "$ROOT/audit.py"
