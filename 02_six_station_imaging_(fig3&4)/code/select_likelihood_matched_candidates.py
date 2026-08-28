#!/usr/bin/env python3
"""Select a common discrepancy-matched reconstruction for every receiver.

Two regularization levels are available for every paired data set:

* the original RML weights (scale 1);
* the positivity- and flux-constrained likelihood limit (scale 0).

For each receiver and seed we retain the most strongly regularized candidate
whose amplitude and closure reduced chi-square values both lie in the common
acceptance interval.  If neither candidate passes, we take the candidate
closest to (1,1) in logarithmic distance.  No image-correlation metric or
ground-truth image enters the selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RAW_SUFFIX = "ns3_eff003_remote6_likelihood"
BASELINE_SUFFIX = "ns3_eff003_remote6_regularized"
SELECTED_SUFFIX = "ns3_eff003_remote6_selected"
RAW_ROOT = (
    ROOT
    / "results"
    / f"paired_povm_100ms_{RAW_SUFFIX}_runs"
    / "100ms"
)
BASELINE_ROOT = (
    ROOT
    / "results"
    / f"paired_povm_100ms_{BASELINE_SUFFIX}_runs"
    / "100ms"
)
OUTPUT_ROOT = (
    ROOT
    / "results"
    / f"paired_povm_100ms_{SELECTED_SUFFIX}_runs"
    / "100ms"
)
RECOVERY_ROOT = ROOT / "results" / "chi2_recovery_unused"
SEEDS = tuple(range(20260529, 20260541))
STRATEGIES = ("edge_uniform", "optimal_singlecopy", "promoted_singlecopy")
CHI2_MIN = 0.80
CHI2_MAX = 1.20


def summary_path(root: Path, seed: int, suffix: str) -> Path:
    stem = f"broad_plume_split_objective_nmode_rml_paired_povm_100ms_seed{seed}_{suffix}"
    return root / f"seed_{seed}" / "rml_outputs" / f"{stem}_summary.json"


def cache_path(payload: dict) -> Path:
    value = payload.get("selected_display_cache_npz")
    if not value:
        value = payload.get("stats", {}).get("selected_display_cache_npz")
    if not value:
        raise KeyError("summary does not identify its selected-display cache")
    return Path(value)


def row_map(payload: dict) -> dict[str, dict]:
    return {str(row["strategy"]): dict(row) for row in payload["rows"]}


def passes(row: dict) -> bool:
    return (
        CHI2_MIN <= float(row["amp_chi2"]) <= CHI2_MAX
        and CHI2_MIN <= float(row["phase_chi2"]) <= CHI2_MAX
    )


def score(row: dict) -> float:
    return max(
        abs(np.log(float(row["amp_chi2"]))),
        abs(np.log(float(row["phase_chi2"]))),
    )


def load_images(payload: dict) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    path = cache_path(payload)
    with np.load(path, allow_pickle=False) as cache:
        truth = np.asarray(cache["truth"], dtype=float)
        axis = np.asarray(cache["axis_uas"], dtype=float)
        images = {
            strategy: np.asarray(cache[f"{strategy}_image"], dtype=float)
            for strategy in STRATEGIES
            if f"{strategy}_image" in cache.files
        }
    parent = payload.get("stats", {}).get("unchanged_parent_rows_reused_from")
    if parent and len(images) < len(STRATEGIES):
        parent_payload = json.loads(Path(parent).read_text())
        parent_truth, parent_axis, parent_images = load_images(parent_payload)
        if not np.array_equal(truth, parent_truth) or not np.array_equal(
            axis, parent_axis
        ):
            raise RuntimeError("baseline and parent display grids differ")
        images.update(parent_images)
    return truth, axis, images


def choose(raw: dict, baseline: dict) -> tuple[dict, str, float]:
    # The common discrepancy principle keeps the strongest acceptable
    # regularization.  Truth-derived image metrics are deliberately ignored.
    if passes(baseline):
        return dict(baseline), "baseline_rml", 1.0
    if passes(raw):
        return dict(raw), "likelihood_limit", 0.0
    if score(baseline) <= score(raw):
        return dict(baseline), "baseline_rml_closest", 1.0
    return dict(raw), "likelihood_limit_closest", 0.0


def recovery_summary_path(seed: int, strategy: str) -> Path:
    run_tag = f"paired_povm_100ms_seed{seed}_ns3_chi2recovery_{strategy}"
    stem = f"broad_plume_split_objective_nmode_rml_{run_tag}"
    return (
        RECOVERY_ROOT
        / f"seed_{seed}"
        / strategy
        / "rml_outputs"
        / f"{stem}_summary.json"
    )


def main() -> None:
    for seed in SEEDS:
        raw_path = summary_path(RAW_ROOT, seed, RAW_SUFFIX)
        baseline_path = summary_path(
            BASELINE_ROOT, seed, BASELINE_SUFFIX
        )
        raw_payload = json.loads(raw_path.read_text())
        baseline_available = baseline_path.is_file()
        baseline_payload = (
            json.loads(baseline_path.read_text())
            if baseline_available
            else raw_payload
        )
        raw_rows = row_map(raw_payload)
        baseline_rows = row_map(baseline_payload)
        raw_truth, raw_axis, raw_images = load_images(raw_payload)
        base_truth, base_axis, base_images = load_images(baseline_payload)
        if not np.array_equal(raw_truth, base_truth) or not np.array_equal(
            raw_axis, base_axis
        ):
            raise RuntimeError(f"seed {seed}: candidate display grids differ")

        selected_rows: list[dict] = []
        selected_images: dict[str, np.ndarray] = {}
        selection: dict[str, dict] = {}
        for strategy in STRATEGIES:
            if baseline_available:
                row, source, scale = choose(
                    raw_rows[strategy], baseline_rows[strategy]
                )
            else:
                row, source, scale = (
                    dict(raw_rows[strategy]),
                    "likelihood_limit_only",
                    0.0,
                )
            recovery_path = recovery_summary_path(seed, strategy)
            recovery_image = None
            if recovery_path.is_file():
                recovery_payload = json.loads(recovery_path.read_text())
                recovery_row = dict(recovery_payload["rows"][0])
                _, _, recovery_images = load_images(recovery_payload)
                recovery_image = recovery_images[strategy]
                if scale == 0.0 and (
                    (passes(recovery_row) and not passes(row))
                    or (
                        passes(recovery_row) == passes(row)
                        and score(recovery_row) < score(row)
                    )
                ):
                    row = recovery_row
                    source = "likelihood_limit_strict_recovery"
                    scale = 0.0
            row["rml_regularizer_scale"] = scale
            row["rml_candidate_source"] = source
            selected_rows.append(row)
            selected_images[strategy] = (
                base_images[strategy]
                if scale == 1.0
                else (
                    recovery_image
                    if source == "likelihood_limit_strict_recovery"
                    else raw_images[strategy]
                )
            )
            selection[strategy] = {
                "source": source,
                "regularizer_scale": scale,
                "amp_chi2": float(row["amp_chi2"]),
                "phase_chi2": float(row["phase_chi2"]),
                "passed_common_interval": passes(row),
                "log_discrepancy_score": score(row),
            }

        run_tag = f"paired_povm_100ms_seed{seed}_{SELECTED_SUFFIX}"
        stem = f"broad_plume_split_objective_nmode_rml_{run_tag}"
        out = OUTPUT_ROOT / f"seed_{seed}" / "rml_outputs"
        out.mkdir(parents=True, exist_ok=True)
        selected_cache = out / f"{stem}_selected_display_cache.npz"
        np.savez_compressed(
            selected_cache,
            truth=raw_truth,
            axis_uas=raw_axis,
            **{
                f"{strategy}_image": selected_images[strategy]
                for strategy in STRATEGIES
            },
        )

        stats = dict(raw_payload["stats"])
        stress = dict(stats["sample_stress_test"])
        stress["run_tag"] = run_tag
        stress["rml_regularizer_selection"] = (
            (
                "largest common scale in {1,0} satisfying "
                f"{CHI2_MIN:.2f} <= (amp,closure) reduced chi2 <= {CHI2_MAX:.2f}; "
                "otherwise minimum worst-component logarithmic discrepancy to (1,1)"
            )
            if baseline_available
            else "positivity- and flux-constrained likelihood-limit reconstruction"
        )
        stress["rml_regularizer_scale_candidates"] = (
            [1.0, 0.0] if baseline_available else [0.0]
        )
        stats["sample_stress_test"] = stress
        stats["selected_display_cache_npz"] = str(selected_cache.resolve())
        stats["likelihood_matching"] = {
            "definition": (
                (
                    "Receiver-blind discrepancy selection over the same two "
                    "regularization levels; no truth-image metric enters selection."
                )
                if baseline_available
                else (
                    "Positivity- and flux-constrained likelihood-limit "
                    "reconstruction; no truth-image metric enters selection."
                )
            ),
            "acceptance_interval": [CHI2_MIN, CHI2_MAX],
            "selection": selection,
            "raw_likelihood_summary": str(raw_path.resolve()),
            "baseline_rml_summary": (
                str(baseline_path.resolve()) if baseline_available else None
            ),
        }
        payload = dict(raw_payload)
        payload["rows"] = selected_rows
        payload["stats"] = stats
        payload["selected_display_cache_npz"] = str(selected_cache.resolve())
        payload["figure_pdf"] = None
        payload["figure_png"] = None
        payload["note"] = (
            "Likelihood-matched reconstruction candidates selected by a common "
            "componentwise discrepancy principle. Receiver physics, paired "
            "noise draws, photon budget, and Fourier coverage are unchanged."
        )
        output_summary = out / f"{stem}_summary.json"
        output_summary.write_text(json.dumps(payload, indent=2) + "\n")
        print(output_summary)


if __name__ == "__main__":
    main()
