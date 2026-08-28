#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
seed=20260535
suffix="ns3_eff002_perband_localclosure_commongauge_nocoreprior_chi0p85_multistart"
out="$ROOT/results/paired_povm_100ms_${suffix}_runs/100ms/seed_${seed}"
tag="paired_povm_100ms_seed${seed}_${suffix}"
log="$ROOT/logs/${tag}.log"
mkdir -p "$out" "$ROOT/logs" "$ROOT/mplconfig"

env \
  MPLCONFIGDIR="$ROOT/mplconfig" \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  FIG2_OUTPUT_ROOT="$out" \
  FIG2_RECEIVER_CACHE="$ROOT/receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.npz" \
  FIG2_EXPOSURE_S=0.1 \
  FIG2_EXISTING_COUPLING=1.0 \
  FIG2_REMOTE_DIAMETER_M=6 \
  FIG2_COLLECTION_EFFICIENCY=0.02 \
  FIG2_COHERENT_BLOCK_SIZE=3 \
  FIG2_PROMOTION_MODEL=finite_ns3_povm_q2_per_band_near_transit \
  FIG2_STRATEGY_FILTER=promoted_singlecopy \
  FIG2_SKIP_PLOT=1 \
  FIG2_RNG_SEED="$seed" \
  RML_ADAM_ITER=2600 \
  RML_CHI2_MAX_PASSES=8 \
  RML_EXTRA_JITTER_STARTS=4 \
  RML_JITTER_START_SIGMA=0.35 \
  RML_TRANSLATION_GAUGE_WEIGHT=0.05 \
  RML_GAUGE_SEARCH_RADIUS_UAS=20 \
  RML_EXACT_TRANSLATION_GAUGE=brightest_core_pixel \
  RML_AMP_CHI2_TARGET=0.85 \
  RML_PHASE_CHI2_TARGET=0.85 \
  RML_AMP_CHI2_MIN=0.75 \
  RML_AMP_CHI2_MAX=1.00 \
  RML_PHASE_CHI2_MIN=0.75 \
  RML_PHASE_CHI2_MAX=1.00 \
  RML_ADAM_TARGET_AMP_CHI2=0.90 \
  RML_ADAM_TARGET_PHASE_CHI2=0.90 \
  RUN_TAG="$tag" \
  python3 "$ROOT/code/run_promoted_povm_rml.py" >"$log" 2>&1

tail -n 30 "$log"
