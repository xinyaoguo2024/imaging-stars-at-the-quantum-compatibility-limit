#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CACHE="$ROOT/receiver_cache_eff002_remote6_qlan_100ms_ns3_v1.npz"
mkdir -p "$ROOT/results/receiver_precompute" "$ROOT/mplconfig" "$ROOT/logs"

env \
  MPLCONFIGDIR="$ROOT/mplconfig" \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  FIG2_PRECOMPUTE_ONLY=1 \
  FIG2_OUTPUT_ROOT="$ROOT/results/receiver_precompute" \
  FIG2_RECEIVER_CACHE="$CACHE" \
  FIG2_EXPOSURE_S=0.1 \
  FIG2_EXISTING_COUPLING=1.0 \
  FIG2_REMOTE_DIAMETER_M=6 \
  FIG2_COLLECTION_EFFICIENCY=0.02 \
  FIG2_COHERENT_BLOCK_SIZE=3 \
  FIG2_PROMOTION_MODEL=coherent_score_operator_qlan_surrogate \
  FIG2_FINITE_NS_QLAN_FISHER_FACTOR=1.0 \
  RUN_TAG=receiver_precompute_eff002_100ms_ns3_qlan \
  python3 "$ROOT/code/run_promoted_povm_rml.py"
