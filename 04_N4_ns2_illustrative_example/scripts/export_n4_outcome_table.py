#!/usr/bin/env python3
"""Export the complete 16-outcome table from the complex-workpoint PVM."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SYMMETRIC_CELLS = [
    (1, 1), (1, 2), (1, 3), (1, 4), (2, 2),
    (2, 3), (2, 4), (3, 3), (3, 4), (4, 4),
]
ANTISYMMETRIC_CELLS = [
    (2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (4, 3),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "n4_ns2_phase_pvm_complexwp.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "n4_ns2_pvm_outcomes.csv",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as saved:
        probabilities = np.asarray(saved["probabilities"], dtype=float)
        derivatives = np.asarray(saved["derivatives"], dtype=float)
    scores = derivatives / probabilities[:, None]
    score_norms = np.linalg.norm(scores, axis=1)
    fisher_traces = probabilities * score_norms**2
    cells = SYMMETRIC_CELLS + ANTISYMMETRIC_CELLS
    sectors = ["symmetric"] * 10 + ["antisymmetric"] * 6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "outcome",
                "terminal_a",
                "terminal_b",
                "copy_sector",
                "probability",
                "score_norm",
                "fisher_trace_contribution",
                *[f"score_edge_{edge}" for edge in ("12", "13", "14", "23", "24", "34")],
            ]
        )
        for index, ((a, b), sector) in enumerate(zip(cells, sectors, strict=True)):
            writer.writerow(
                [
                    index + 1,
                    a,
                    b,
                    sector,
                    probabilities[index],
                    score_norms[index],
                    fisher_traces[index],
                    *scores[index],
                ]
            )
    print(args.output)


if __name__ == "__main__":
    main()
