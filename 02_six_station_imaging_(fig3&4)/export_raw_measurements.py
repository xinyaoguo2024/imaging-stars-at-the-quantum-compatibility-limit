#!/usr/bin/env python3
"""Export the seeded synthetic measurement records before RML fitting.

The archived RML caches contain the fitted per-band images.  This companion
export records the actual simulated Fourier coordinates, amplitudes, closure
phases, uncertainties, and mixed covariance matrices for every wavelength,
epoch, and receiver.  It therefore provides a compact raw-data layer from
which the likelihood can be reconstructed without rerunning receiver design.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
STRATEGIES = ("edge_uniform", "optimal_singlecopy", "promoted_singlecopy")


def configure(seed: int) -> None:
    os.environ.update(
        {
            "FIG2_RNG_SEED": str(seed),
            "FIG2_EXPOSURE_S": "0.1",
            "FIG2_EXISTING_COUPLING": "1.0",
            "FIG2_REMOTE_DIAMETER_M": "6.0",
            "FIG2_COLLECTION_EFFICIENCY": "0.02",
            "FIG2_COHERENT_BLOCK_SIZE": "3",
            "FIG2_PROMOTION_MODEL": "finite_ns3_povm_q2_per_band_near_transit",
            "FIG2_RECEIVER_CACHE": str(
                ROOT / "receiver_cache_eff002_remote6_finite_perband_100ms_ns3_v1.npz"
            ),
            "FIG2_SKIP_PLOT": "1",
            "MPLCONFIGDIR": str(ROOT / "mplconfig"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "raw_measurements")
    args = parser.parse_args()
    configure(args.seed)
    sys.path.insert(0, str(ROOT / "code"))
    import run_promoted_povm_rml as model  # noqa: E402

    model.configure_good_runtime()
    model.val.closure_basis = model.latest_closure_basis
    case = model.make_six_station_case()
    bands, stats, truth, axis_uas, _prior, _starts = model.simulate_good_bands(case, {})

    arrays: dict[str, np.ndarray] = {
        "truth": np.asarray(truth, dtype=float),
        "axis_uas": np.asarray(axis_uas, dtype=float),
    }
    common = ("u", "v", "amp_true", "closure_reference")
    for band_index, band in enumerate(bands):
        prefix = f"band{band_index:02d}_"
        for key in common:
            arrays[prefix + key] = np.asarray(band[key])
        for strategy in STRATEGIES:
            for key in ("amp", "amp_sigma", "closure", "sigmaqcov", "mixedcov", "vis"):
                arrays[prefix + key + "_" + strategy] = np.asarray(
                    band[f"{key}_{strategy}"]
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"seed_{args.seed}_raw_measurements.npz"
    json_path = args.output_dir / f"seed_{args.seed}_raw_measurements.json"
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "seed": args.seed,
        "definition": "Synthetic pre-RML amplitude and closure-phase records with full mixed covariance",
        "n_bands": len(bands),
        "strategies": list(STRATEGIES),
        "wavelength_bin_centers_nm": model.wavelength_bin_centers_and_weights()[0].tolist(),
        "simulation_stats": stats,
        "npz": npz_path.name,
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(npz_path)
    print(json_path)


if __name__ == "__main__":
    main()
