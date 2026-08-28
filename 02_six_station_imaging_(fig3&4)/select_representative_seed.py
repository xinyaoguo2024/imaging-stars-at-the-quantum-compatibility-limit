#!/usr/bin/env python3
"""Choose one visualization seed closest to the paired-ensemble mean.

The distance uses the six standardized image-correlation coordinates formed
by three receivers and two image regions.  It is used only to choose the panel
shown in Fig. 3; every inferential statistic retains all twelve seeds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SUFFIX = "ns3_eff002_perband_localclosure_nogauge_nocoreprior_chi0p85w075to100"
RUN_ROOT = Path(
    os.environ.get(
        "FIG2_REPRESENTATIVE_RUN_ROOT",
        str(ROOT / "results_12seed_raw" / "100ms"),
    )
)
SEEDS = tuple(range(20260529, 20260541))
STRATEGIES = ("edge_uniform", "optimal_singlecopy", "promoted_singlecopy")
METRICS = ("blr_corr", "global_corr")


def summary_path(seed: int) -> Path:
    tag = f"paired_povm_100ms_seed{seed}_{SUFFIX}"
    stem = f"broad_plume_split_objective_nmode_rml_{tag}_summary.json"
    return RUN_ROOT / f"seed_{seed}" / "rml_outputs" / stem


def main() -> None:
    coordinates = []
    inputs = []
    for seed in SEEDS:
        path = summary_path(seed)
        payload = json.loads(path.read_text())
        rows = {row["strategy"]: row for row in payload["rows"]}
        coordinates.append(
            [float(rows[strategy][metric]) for strategy in STRATEGIES for metric in METRICS]
        )
        inputs.append(str(path.resolve()))

    values = np.asarray(coordinates, dtype=float)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 0.0, scale, 1.0)
    distances = np.sqrt(np.sum(((values - mean) / scale) ** 2, axis=1))
    index = int(np.argmin(distances))
    selected = int(SEEDS[index])
    payload = {
        "criterion": "minimum Euclidean distance to the ensemble mean in six standardized image-correlation coordinates",
        "selected_seed": selected,
        "seeds": list(SEEDS),
        "coordinate_order": [f"{strategy}:{metric}" for strategy in STRATEGIES for metric in METRICS],
        "ensemble_mean": mean.tolist(),
        "ensemble_sample_std": scale.tolist(),
        "standardized_distances": {
            str(seed): float(distance) for seed, distance in zip(SEEDS, distances)
        },
        "input_summaries": inputs,
    }
    output = ROOT / "results" / f"representative_seed_{SUFFIX}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(selected)
    print(output)


if __name__ == "__main__":
    main()
