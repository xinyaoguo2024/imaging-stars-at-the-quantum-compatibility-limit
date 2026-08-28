#!/usr/bin/env python3
"""Apply ten independently calibrated finite-n_s factors to the QLAN cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_NPZ = HERE / "receiver_cache_eff002_remote6_qlan_100ms_ns3_v1.npz"
SOURCE_JSON = SOURCE_NPZ.with_suffix(".json")
TARGET_NPZ = HERE / "receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.npz"
TARGET_JSON = TARGET_NPZ.with_suffix(".json")
PROMOTION_MODEL = "finite_ns3_povm_q2_per_band_near_transit"


def calibration_path(band_index: int) -> Path:
    matches = sorted(
        HERE.glob(f"finite_ns3_povm_q2_band{band_index:02d}_*nm.json")
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one calibration for band {band_index}, got {matches}"
        )
    return matches[0]


def main() -> None:
    calibration_files = [calibration_path(index) for index in range(10)]
    calibrations = [json.loads(path.read_text()) for path in calibration_files]
    factors = np.asarray(
        [
            float(item["global_fisher_factor_povm_over_qlan"])
            for item in calibrations
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0) or np.any(
        factors > 1.0
    ):
        raise ValueError(f"invalid per-band finite-copy factors: {factors}")

    with np.load(SOURCE_NPZ, allow_pickle=False) as source:
        edge = np.asarray(source["cov_edge_uniform"], dtype=float)
        single = np.asarray(source["cov_optimal_singlecopy"], dtype=float)
        qlan = np.asarray(source["cov_promoted_singlecopy"], dtype=float)
    if qlan.shape[0] != len(factors):
        raise ValueError(
            f"QLAN cache has {qlan.shape[0]} bands but {len(factors)} factors"
        )
    finite = qlan / factors[:, None, None, None]
    np.savez_compressed(
        TARGET_NPZ,
        cov_edge_uniform=edge,
        cov_optimal_singlecopy=single,
        cov_promoted_singlecopy=finite,
        cov_promoted_qlan=qlan,
    )

    metadata = json.loads(SOURCE_JSON.read_text())
    old_aggregate = metadata.get("aggregate_efficiency", {}).get(
        "promoted_qfi_efficiency"
    )
    if old_aggregate is not None:
        metadata["aggregate_efficiency"]["qlan_qfi_efficiency"] = old_aggregate
        metadata["aggregate_efficiency"].pop("promoted_qfi_efficiency", None)
    for sample in metadata.get("sample_diagnostics", []):
        old = sample.pop("promoted_qfi_efficiency", None)
        if old is not None:
            sample["qlan_qfi_efficiency"] = old

    metadata.update(
        {
            "coherent_block_size": 3,
            "promotion_model": PROMOTION_MODEL,
            "finite_ns_qlan_fisher_factor": 1.0,
            "finite_ns_scaling_mode": "per_10nm_band_near_transit",
            "finite_ns_band_fisher_factors": factors.tolist(),
            "finite_ns_band_centers_nm": [
                float(item["working_point"]["lambda_center_nm"])
                for item in calibrations
            ],
            "finite_ns_calibration_json": [
                str(path.resolve()) for path in calibration_files
            ],
            "qlan_covariance_reference_key": "cov_promoted_qlan",
            "finite_block_caveat": (
                "Each 10-nm wavelength band retains its per-sample QLAN "
                "covariance and is inflated by an independently calibrated "
                "finite-n_s factor.  Each factor comes from a locally optimized "
                "q=2, n_s=3 overcomplete POVM at that band's near-transit "
                "working point and is shared only across epochs within the band."
            ),
            "source_qlan_cache_npz": str(SOURCE_NPZ.resolve()),
        }
    )
    TARGET_JSON.write_text(json.dumps(metadata, indent=2) + "\n")

    recovered = np.diagonal(qlan, axis1=-2, axis2=-1) / np.diagonal(
        finite, axis1=-2, axis2=-1
    )
    maximum_error = float(
        np.max(np.abs(recovered - factors[:, None, None]))
    )
    print(
        json.dumps(
            {
                "promotion_model": PROMOTION_MODEL,
                "band_centers_nm": metadata["finite_ns_band_centers_nm"],
                "band_fisher_factors": factors.tolist(),
                "factor_min": float(np.min(factors)),
                "factor_max": float(np.max(factors)),
                "roundtrip_max_abs": maximum_error,
                "target_npz": str(TARGET_NPZ.resolve()),
                "target_json": str(TARGET_JSON.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
