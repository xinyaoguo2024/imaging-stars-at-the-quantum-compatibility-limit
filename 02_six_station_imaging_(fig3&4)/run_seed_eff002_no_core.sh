#!/bin/zsh
set -euo pipefail

seed="${1:-20260529}"
chi_target="${2:-0.85}"
chi_tag="${chi_target//./p}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$ROOT/code"
SUFFIX="ns3_eff002_perband_localclosure_nogauge_nocoreprior_chi${chi_tag}w075to100"
RUN_ROOT="${FIG2_RUN_ROOT:-$ROOT/recomputed_12seed_runs/100ms}"
output_root="$RUN_ROOT/seed_${seed}"
run_tag="paired_povm_100ms_seed${seed}_${SUFFIX}"
log="$ROOT/logs/${run_tag}.log"
mkdir -p "$output_root" "$ROOT/logs" "$ROOT/mplconfig"

echo "[eff002 no-core-prior] start seed=${seed} chi_target=${chi_target}"
env \
  MPLCONFIGDIR="$ROOT/mplconfig" \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  FIG2_OUTPUT_ROOT="$output_root" \
  FIG2_RECEIVER_CACHE="$ROOT/receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.npz" \
  FIG2_EXPOSURE_S=0.1 \
  FIG2_EXISTING_COUPLING=1.0 \
  FIG2_REMOTE_DIAMETER_M=6 \
  FIG2_COLLECTION_EFFICIENCY=0.02 \
  FIG2_COHERENT_BLOCK_SIZE=3 \
  FIG2_PROMOTION_MODEL=finite_ns3_povm_q2_per_band_near_transit \
  FIG2_STRATEGY_FILTER="${FIG2_STRATEGY_FILTER:-edge_uniform,optimal_singlecopy,promoted_singlecopy}" \
  FIG2_SKIP_PLOT=1 \
  FIG2_RNG_SEED="$seed" \
  RML_ADAM_ITER=2600 \
  RML_CHI2_MAX_PASSES=4 \
  RML_AMP_CHI2_TARGET="$chi_target" \
  RML_PHASE_CHI2_TARGET="$chi_target" \
  RML_AMP_CHI2_MIN=0.75 \
  RML_AMP_CHI2_MAX=1.00 \
  RML_PHASE_CHI2_MIN=0.75 \
  RML_PHASE_CHI2_MAX=1.00 \
  RML_ADAM_TARGET_AMP_CHI2=0.85 \
  RML_ADAM_TARGET_PHASE_CHI2=0.85 \
  RUN_TAG="$run_tag" \
  python3 "$CODE/run_promoted_povm_rml.py" >"$log" 2>&1

echo "[eff002 no-core-prior] complete seed=${seed} chi_target=${chi_target}"
tail -n 24 "$log"
