#!/usr/bin/env python3
"""Audit the two wavelength bins bracketing observed broad H-alpha.

The audit separates an optimization anomaly (poor componentwise residuals)
from an ordinary noisy realization.  It never selects a reconstruction solely
because its truth correlation is larger.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


STRATEGIES = ("edge_uniform", "optimal_singlecopy", "promoted_singlecopy")
TARGET = 0.85
CHI_MIN = 0.75
CHI_MAX = 1.00


def target_score(row: dict) -> float:
    return max(
        abs(math.log(max(float(row["amp_chi2"]), 1e-12) / TARGET)),
        abs(math.log(max(float(row["phase_chi2"]), 1e-12) / TARGET)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    diagnostics = payload["band_diagnostics"]
    centers = [float(x) for x in payload["wavelength_bin_centers_nm"]]
    observed_halpha_nm = 656.3 * (1.0 + 0.00332)
    selected = sorted(range(len(centers)), key=lambda i: abs(centers[i] - observed_halpha_nm))[:2]
    selected.sort()

    rows = []
    suspicious = False
    for index in selected:
        by_strategy = {key: diagnostics[key][index] for key in STRATEGIES}
        collective = by_strategy["promoted_singlecopy"]
        single_rows = [by_strategy["edge_uniform"], by_strategy["optimal_singlecopy"]]
        collective_underperforms = any(
            float(collective[metric]) < min(float(item[metric]) for item in single_rows) - 0.03
            for metric in ("blr_corr", "global_corr")
        )
        collective_bad_residual = not (
            CHI_MIN <= float(collective["amp_chi2"]) <= CHI_MAX
            and CHI_MIN <= float(collective["phase_chi2"]) <= CHI_MAX
        )
        collective_score_excess = target_score(collective) > min(target_score(item) for item in single_rows) + 0.12
        convergence_suspect = collective_underperforms and (
            collective_bad_residual or collective_score_excess
        )
        suspicious = suspicious or convergence_suspect
        for key in STRATEGIES:
            item = by_strategy[key]
            rows.append(
                {
                    "lambda_nm": centers[index],
                    "strategy": key,
                    "amp_chi2": float(item["amp_chi2"]),
                    "phase_chi2": float(item["phase_chi2"]),
                    "blr_corr": float(item["blr_corr"]),
                    "global_corr": float(item["global_corr"]),
                    "target_score": target_score(item),
                    "convergence_suspect": bool(convergence_suspect and key == "promoted_singlecopy"),
                }
            )

    report = {
        "summary": str(args.summary.resolve()),
        "observed_halpha_nm": observed_halpha_nm,
        "audited_centers_nm": [centers[i] for i in selected],
        "criterion": (
            "A collective H-alpha result is marked convergence-suspect only when its BLR or "
            "global correlation is more than 0.03 below both single-copy results and its "
            "componentwise residual score is abnormal."
        ),
        "rerun_recommended": suspicious,
        "rows": rows,
    }
    output = args.summary.with_name(args.summary.stem.replace("_summary", "_halpha_audit") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        flag = "  RERUN?" if row["convergence_suspect"] else ""
        print(
            f"{row['lambda_nm']:5.1f} {row['strategy']:24s} "
            f"chi=({row['amp_chi2']:.3f},{row['phase_chi2']:.3f}) "
            f"BLR={row['blr_corr']:.3f} all={row['global_corr']:.3f}{flag}"
        )
    print(f"rerun_recommended={suspicious}")
    print(output)


if __name__ == "__main__":
    main()
