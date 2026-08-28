#!/usr/bin/env python3
"""Regenerate Fig. 5 from the fixed-|g| random-phase mean Fisher gain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import make_fig5_promoted_uniform as base  # noqa: E402


def main() -> None:
    summary_path = Path(
        os.environ.get(
            "FIG5_MC_SUMMARY",
            str(ROOT / "data" / "fixed_modulus_phase_gain_summary.json"),
        )
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"missing {summary_path}; run run_fixed_modulus_phase_mc.py first"
        )
    summary = json.loads(summary_path.read_text())
    records = summary["gain_by_N_and_magnitude"]["20"]
    magnitude_grid = np.asarray(
        [row["magnitude_ab"] for row in records], dtype=float
    )
    mean_gain = np.asarray(
        [row["mean_fisher_gain"] for row in records], dtype=float
    )
    original_fisher_terms = base.fisher_terms

    def mean_fisher_terms(magnitude_ab):
        values = original_fisher_terms(magnitude_ab)
        magnitude = np.asarray(magnitude_ab, dtype=float)
        interpolated_gain = np.interp(
            magnitude,
            magnitude_grid,
            mean_gain,
            left=mean_gain[0],
            right=mean_gain[-1],
        )
        values["F_prom"] = values["F_edge"] * interpolated_gain
        values["fixed_modulus_phase_mean_gain"] = interpolated_gain
        return values

    base.fisher_terms = mean_fisher_terms
    base.FIGURE_DIR = Path(
        os.environ.get("FIG5_FINAL_FIGURE_DIR", str(ROOT / "generated_outputs"))
    )
    base.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base.FIGURE_STEM = "fig5_astrophysical_reach"
    base.PROMOTED_CURVE_LABEL = "collective uniform edge-first measurement"
    promoted_limit, edge_limit = base.make_figure()

    displayed = (magnitude_grid >= base.MAG_MIN) & (
        magnitude_grid <= base.MAG_MAX
    )
    displayed_gain = mean_gain[displayed]
    output = {
        "artifact": base.FIGURE_STEM,
        "source_monte_carlo_summary": str(summary_path.resolve()),
        "target_loop": "S1-S2-S3, fixed before sampling",
        "averaging": "arithmetic mean Fisher gain at each magnitude",
        "edge_first": (
            "exact fixed-|g|=0.5 copy-local uniform edge-first reference"
        ),
        "sigma_cl_equals_pi": {
            "phase_mean_promoted_mag_ab": float(promoted_limit),
            "copy_local_edge_first_mag_ab": float(edge_limit),
        },
        "displayed_gain_audit": {
            "magnitude_range": [base.MAG_MIN, base.MAG_MAX],
            "mean_fisher_gain_range": [
                float(np.min(displayed_gain)),
                float(np.max(displayed_gain)),
            ],
            "baseline_or_snr_gain_range": [
                float(np.sqrt(np.min(displayed_gain))),
                float(np.sqrt(np.max(displayed_gain))),
            ],
        },
        "parameters": {
            "N": base.N_STATION,
            "all_offdiagonal_magnitudes": 0.5,
            "M_total": base.M_TOTAL,
            "eta": base.ETA,
            "epsilon": base.EPSILON_BG,
            "phase_budget_fraction": base.PHASE_BUDGET_FRACTION,
            "reference_black_hole_mass_msun": base.MASS_REFERENCE_MSUN,
            "displayed_magnitude_range": [base.MAG_MIN, base.MAG_MAX],
        },
        "source_mass_and_baseline_audit": [
            {
                "name": src.name,
                "mass_msun": src.mbh_msun,
                "mass_error_minus_msun": src.mbh_err_minus_msun,
                "mass_error_plus_msun": src.mbh_err_plus_msun,
                "mass_reference": src.mass_reference,
                "single_copy_required_baseline_km": float(
                    base.required_baseline_km(
                        src.z, src.mag_ab, "edge", src.mbh_msun
                    )
                ),
                "collective_required_baseline_km": float(
                    base.required_baseline_km(
                        src.z, src.mag_ab, "promoted", src.mbh_msun
                    )
                ),
                "baseline_reduction_factor": float(
                    base.required_baseline_km(
                        src.z, src.mag_ab, "edge", src.mbh_msun
                    )
                    / base.required_baseline_km(
                        src.z, src.mag_ab, "promoted", src.mbh_msun
                    )
                ),
            }
            for src in base.SOURCES
        ],
    }
    output_path = base.FIGURE_DIR / f"{base.FIGURE_STEM}.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(base.FIGURE_DIR / f"{base.FIGURE_STEM}.png")
    print(base.FIGURE_DIR / f"{base.FIGURE_STEM}.pdf")
    print(output_path)


if __name__ == "__main__":
    main()
