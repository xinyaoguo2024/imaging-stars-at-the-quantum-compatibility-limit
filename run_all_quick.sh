#!/bin/zsh
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/01_principle_and_implementation_schematics_(fig1&2)/run_all.sh"
"$ROOT/02_six_station_imaging_(fig3&4)/run_from_archived_data.sh"
"$ROOT/03_astrophysical_reach_fig5/run_from_archived_data.sh"
"$ROOT/04_N4_ns2_illustrative_example/run_from_archived_data.sh"
python3 "$ROOT/audit_archive.py"
