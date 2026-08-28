#!/usr/bin/env python3
"""Audit the archived astrophysical-reach Monte Carlo and final curve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summary = json.loads((ROOT / "data" / "fixed_modulus_phase_gain_summary.json").read_text())
    if summary["monte_carlo"]["accepted_samples_per_N"] != 600:
        raise RuntimeError("Monte Carlo sample count is not 600 per N")
    if summary["ensemble"]["N_values"] != [6, 20]:
        raise RuntimeError("missing N=6 or N=20 ensemble")
    if summary["ensemble"]["all_offdiagonal_magnitudes"] != 0.5:
        raise RuntimeError("the fixed-|g| ensemble is not |g|=0.5")
    if summary["ensemble"]["target_loop"] != "S1-S2-S3, fixed before sampling":
        raise RuntimeError("the typical loop was not fixed before sampling")

    with np.load(ROOT / "data" / "fixed_modulus_phase_gain_samples.npz") as saved:
        if saved["gain_N6"].shape != (600, 13):
            raise RuntimeError("N=6 raw gain array is incomplete")
        if saved["gain_N20"].shape != (600, 13):
            raise RuntimeError("N=20 raw gain array is incomplete")
        if np.max(np.abs(saved["minimum_eigenvalue_N20"] < -1e-10)):
            raise RuntimeError("non-PSD coherence draw found")

    reach = json.loads((ROOT / "data" / "fig5_astrophysical_reach.json").read_text())
    params = reach["parameters"]
    expected = {
        "N": 20,
        "all_offdiagonal_magnitudes": 0.5,
        "M_total": 1e11,
        "eta": 0.2,
        "epsilon": 1e-11,
        "phase_budget_fraction": 1.0,
    }
    for key, value in expected.items():
        if params[key] != value:
            raise RuntimeError(f"astrophysical-reach parameter mismatch: {key}")
    gain = reach["displayed_gain_audit"]["baseline_or_snr_gain_range"]
    if not (4.73 < gain[0] < gain[1] < 5.44):
        raise RuntimeError("displayed baseline/SNR gain range mismatch")
    if len(reach["source_mass_and_baseline_audit"]) < 10:
        raise RuntimeError("source catalog is incomplete")

    generated = ROOT / "generated_outputs" / "fig5_astrophysical_reach.png"
    reference = ROOT / "reference_outputs" / "fig5_astrophysical_reach.png"
    if generated.is_file():
        if sha(generated) != sha(reference):
            raise RuntimeError("astrophysical-reach PNG mismatch")
        print("PASS exact PNG: current astrophysical-reach figure")
    print("PASS raw Monte Carlo: N=6,20; 600 accepted draws per N")
    print(f"PASS source catalog: {len(reach['source_mass_and_baseline_audit'])} objects")


if __name__ == "__main__":
    main()
