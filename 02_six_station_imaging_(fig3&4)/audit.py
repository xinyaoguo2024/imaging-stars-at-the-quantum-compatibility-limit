#!/usr/bin/env python3
"""Numerical and artifact audit for the six-station imaging archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SUFFIX = "ns3_eff002_perband_localclosure_nogauge_nocoreprior_chi0p85w075to100"
SEEDS = tuple(range(20260529, 20260541))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(value: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(value - expected) > tolerance:
        raise RuntimeError(f"numeric mismatch: {value} vs {expected}")


def main() -> None:
    summaries = []
    for seed in SEEDS:
        run = ROOT / "results_12seed_raw" / "100ms" / f"seed_{seed}" / "rml_outputs"
        summary = run / f"broad_plume_split_objective_nmode_rml_paired_povm_100ms_seed{seed}_{SUFFIX}_summary.json"
        raw = ROOT / "raw_measurements" / f"seed_{seed}_raw_measurements.npz"
        if not summary.is_file() or not raw.is_file():
            raise FileNotFoundError(f"missing archived seed {seed}")
        summaries.append(summary)
        with np.load(raw, allow_pickle=False) as saved:
            required = {
                "truth", "axis_uas", "band00_u", "band09_v",
                "band00_amp_edge_uniform", "band00_amp_optimal_singlecopy",
                "band00_amp_promoted_singlecopy", "band09_closure_promoted_singlecopy",
                "band09_mixedcov_promoted_singlecopy",
            }
            if not required.issubset(saved.files):
                raise RuntimeError(f"raw measurement fields incomplete for seed {seed}")
            if saved["band00_u"].shape != (540,):
                raise RuntimeError(f"unexpected UV sample count for seed {seed}")
            if saved["band00_closure_edge_uniform"].shape != (36, 10):
                raise RuntimeError(f"unexpected closure shape for seed {seed}")

    receiver_json = json.loads(
        (ROOT / "receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.json").read_text()
    )
    for key, expected in {
        "n_station": 6,
        "n_edge": 15,
        "n_closure": 10,
        "n_wavelength_bins": 10,
        "n_time_windows": 36,
        "exposure_s": 0.1,
        "existing_coupled_area_fraction": 1.0,
        "remote_diameter_m": 6.0,
        "photon_collection_efficiency": 0.02,
        "amplitude_branch_fraction": 0.5,
        "phase_branch_fraction": 0.5,
        "coherent_block_size": 3,
    }.items():
        if receiver_json.get(key) != expected:
            raise RuntimeError(f"receiver setting mismatch: {key}")

    stats_path = ROOT / "results" / f"sm_paired_povm_100ms_{SUFFIX}_statistics.json"
    stats = json.loads(stats_path.read_text())
    metadata = stats["metadata"]
    if metadata["n_seed"] != 12 or metadata["phase_dimension_C"] != 10:
        raise RuntimeError("12-seed/closure-space metadata mismatch")
    if metadata["amplitude_dimension_E"] != 15:
        raise RuntimeError("amplitude edge-space metadata mismatch")

    strategy = stats["strategy_statistics"]
    close(strategy["blr_corr"]["edge_uniform"]["mean"], 0.686)
    close(strategy["blr_corr"]["optimal_singlecopy"]["mean"], 0.726)
    close(strategy["blr_corr"]["promoted_singlecopy"]["mean"], 0.810)
    close(strategy["global_corr"]["edge_uniform"]["mean"], 0.905)
    close(strategy["global_corr"]["optimal_singlecopy"]["mean"], 0.914)
    close(strategy["global_corr"]["promoted_singlecopy"]["mean"], 0.938)

    representative = json.loads(
        (ROOT / "results" / f"representative_seed_{SUFFIX}.json").read_text()
    )
    if representative["selected_seed"] != 20260536:
        raise RuntimeError("representative seed mismatch")

    current_stats = json.loads(
        (ROOT / "plotting_data" / "paired_12seed_100ms_summary.json").read_text()
    )
    if current_stats["seeds"] != list(SEEDS):
        raise RuntimeError("current plotting statistics do not contain the paired 12-seed ensemble")
    for metric, key, expected in (
        ("blr_corr", "edge_uniform", 0.6863622897642722),
        ("blr_corr", "optimal_singlecopy", 0.7259868740113228),
        ("blr_corr", "collective_ns3", 0.8096854665702233),
        ("global_corr", "edge_uniform", 0.9052404400837806),
        ("global_corr", "optimal_singlecopy", 0.9135682506582201),
        ("global_corr", "collective_ns3", 0.9380322584658893),
    ):
        close(current_stats["statistics"][metric][key]["mean"], expected, tolerance=1e-12)

    plotting_audit = ROOT / "generated_outputs" / "audit.json"
    if plotting_audit.is_file():
        plotting_metadata = json.loads(plotting_audit.read_text())
        if plotting_metadata["cutoff"] != 0.015:
            raise RuntimeError("main-text display cutoff mismatch")
        if plotting_metadata["contour_levels"] != [0.015, 0.03, 0.1, 0.3]:
            raise RuntimeError("main-text contour-level mismatch")
        if plotting_metadata["asymptotic_series_in_statistics_figure"]:
            raise RuntimeError("asymptotic series should not appear in current Fig. 4")

    imaging = ROOT / "generated_outputs" / "fig3_fourpanel_singlecolumn.png"
    stat_figure = ROOT / "generated_outputs" / "fig4_statistics_singlecolumn.png"
    pairs = (
        (imaging, ROOT / "reference_outputs" / "fig3_fourpanel_singlecolumn.png"),
        (stat_figure, ROOT / "reference_outputs" / "fig4_statistics_singlecolumn.png"),
    )
    for generated, reference in pairs:
        if not generated.is_file():
            print(f"PASS numerical archive; generated figure not requested: {generated.name}")
        elif sha(generated) != sha(reference):
            raise RuntimeError(f"PNG mismatch: {generated}")
        else:
            print(f"PASS exact PNG: {generated.name}")

    allowed_artifacts = {
        ROOT / "generated_outputs" / f"fig{number}_{stem}.{suffix}"
        for number, stem in (
            (3, "fourpanel_singlecolumn"),
            (4, "statistics_singlecolumn"),
        )
        for suffix in ("pdf", "png")
    } | {
        ROOT / "reference_outputs" / f"fig{number}_{stem}.{suffix}"
        for number, stem in (
            (3, "fourpanel_singlecolumn"),
            (4, "statistics_singlecolumn"),
        )
        for suffix in ("pdf", "png")
    }
    found_artifacts = {
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".png", ".svg"}
    }
    unexpected = sorted(found_artifacts - allowed_artifacts)
    missing = sorted(allowed_artifacts - found_artifacts)
    if unexpected:
        raise RuntimeError(
            "superseded figure artifacts remain in the clean module: "
            + ", ".join(str(path.relative_to(ROOT)) for path in unexpected)
        )
    if missing:
        raise FileNotFoundError(
            "current figure artifacts are incomplete: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        )
    print("PASS clean figure set: current Fig. 3 and Fig. 4 only")
    print(f"PASS complete paired archive: {len(summaries)} seeds, 10 bands, 3 receivers")


if __name__ == "__main__":
    main()
