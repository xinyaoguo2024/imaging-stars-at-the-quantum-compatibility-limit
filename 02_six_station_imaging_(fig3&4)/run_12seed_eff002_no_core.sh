#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$ROOT/run_seed_eff002_no_core.sh"
export FIG2_STRATEGY_FILTER="edge_uniform,optimal_singlecopy,promoted_singlecopy"

for seed in {20260529..20260540}; do
  echo "$seed"
done | xargs -P 3 -I {} "$RUNNER" {} 0.85
