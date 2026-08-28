from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as case_lib
import eht_style_weak_prior_closure_rml as rml
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

N_TRIALS = int(os.environ.get("HAWAII4_SCAN_TRIALS", "5000"))
RUN_TOP = int(os.environ.get("HAWAII4_SCAN_RUN_TOP", "3"))
RNG_SEED = int(os.environ.get("HAWAII4_SCAN_SEED", "4151"))
RML_STRATEGY = os.environ.get("HAWAII4_SCAN_STRATEGY", "direct")


def core_telescopes() -> list[aug.Telescope]:
    full = case_lib.load_maunakea_case()
    return [tel for tel in full.telescopes if not tel.is_added]


def base_remote_telescopes() -> list[aug.Telescope]:
    full = case_lib.load_maunakea_case()
    return [tel for tel in full.telescopes if tel.is_added]


def make_case(
    key: str,
    remote_xy: np.ndarray,
    *,
    hub: np.ndarray | None = None,
    score: float = 0.0,
) -> aug.NetworkCase:
    core = core_telescopes()
    telescopes = list(core)
    for idx, (x, y) in enumerate(remote_xy, start=1):
        telescopes.append(aug.Telescope(f"new 5 m opt{idx}", float(x), float(y), 5.0, True))
    stations, diameters, _, _ = aug.station_table_from_case(
        aug.NetworkCase(
            key=key,
            title=key,
            latitude_deg=19.8250,
            center_latlon=(19.8250, -155.4720),
            telescopes=telescopes,
            hub_km=(0.0, 0.0),
            optimization_score=score,
        )
    )
    if hub is None:
        hub, hub_score = aug.optimize_hub(stations, diameters)
        score = score + 0.15 * hub_score
    return aug.NetworkCase(
        key=key,
        title=key.replace("_", " "),
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=(float(hub[0]), float(hub[1])),
        optimization_score=float(score),
    )


def score_case(case: aug.NetworkCase) -> float:
    old_dec = aug.SOURCE_DEC_DEG
    old_n_time = aug.N_TIME_WINDOWS
    old_exp = aug.EXPOSURE_S
    old_gap = aug.EXPOSURE_GAP_S
    try:
        aug.SOURCE_DEC_DEG = ngc.NGC4151.dec_deg
        aug.N_TIME_WINDOWS = rml.N_TIME_WINDOWS
        aug.EXPOSURE_S = rml.EXPOSURE_S
        aug.EXPOSURE_GAP_S = rml.EXPOSURE_GAP_S
        stations, diameters, _, _ = aug.station_table_from_case(case)
        return aug.coverage_score(
            stations,
            diameters,
            latitude_deg=case.latitude_deg,
            hub=np.asarray(case.hub_km, dtype=float),
            max_target_g_lambda=80.0,
        )
    finally:
        aug.SOURCE_DEC_DEG = old_dec
        aug.N_TIME_WINDOWS = old_n_time
        aug.EXPOSURE_S = old_exp
        aug.EXPOSURE_GAP_S = old_gap


def fixed_radii_case(key: str, radii: np.ndarray, angles_deg: np.ndarray) -> aug.NetworkCase:
    core = core_telescopes()
    origin = np.mean(np.array([[tel.x_km, tel.y_km] for tel in core], dtype=float), axis=0)
    angles = np.deg2rad(np.asarray(angles_deg, dtype=float))
    remote = origin + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    case = make_case(key, remote)
    return replace(case, optimization_score=score_case(case))


def existing_subset_case(key: str, keep_indices: list[int]) -> aug.NetworkCase:
    remotes = base_remote_telescopes()
    remote = np.array([[remotes[i].x_km, remotes[i].y_km] for i in keep_indices], dtype=float)
    full = case_lib.load_maunakea_case()
    case = make_case(key, remote, hub=np.asarray(full.hub_km, dtype=float))
    return replace(case, optimization_score=score_case(case))


def random_fixed_radii_candidates() -> list[aug.NetworkCase]:
    rng = np.random.default_rng(RNG_SEED)
    core = core_telescopes()
    origin = np.mean(np.array([[tel.x_km, tel.y_km] for tel in core], dtype=float), axis=0)
    radii = np.array([5.0, 9.0, 14.0, 20.0], dtype=float)
    candidates: list[aug.NetworkCase] = []
    best: list[tuple[float, aug.NetworkCase]] = []
    for trial in range(N_TRIALS):
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=4))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        if np.min(gaps) < np.deg2rad(48.0):
            continue
        remote = origin + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        case = make_case(f"hawaii4_scan_{trial:05d}", remote)
        score = score_case(case)
        best.append((score, replace(case, optimization_score=score)))
    best.sort(key=lambda item: item[0], reverse=True)
    candidates.extend(case for _, case in best[: max(RUN_TOP, 1)])
    return candidates


def candidate_cases() -> list[aug.NetworkCase]:
    cases = [
        existing_subset_case("hawaii4_existing_drop20km", [0, 1, 2, 3]),
        existing_subset_case("hawaii4_existing_drop5km", [1, 2, 3, 4]),
        existing_subset_case("hawaii4_existing_drop875km", [0, 2, 3, 4]),
        fixed_radii_case("hawaii4_cardinal_5_9_14_20", np.array([5.0, 9.0, 14.0, 20.0]), np.array([5.0, 92.0, 188.0, 286.0])),
        fixed_radii_case("hawaii4_diagonal_5_9_14_20", np.array([5.0, 9.0, 14.0, 20.0]), np.array([45.0, 135.0, 225.0, 315.0])),
    ]
    cases.extend(random_fixed_radii_candidates())
    # Deduplicate by key while preserving order.
    unique = {}
    for case in cases:
        unique[case.key] = case
    return sorted(unique.values(), key=lambda c: c.optimization_score, reverse=True)


def summarize_layout(case: aug.NetworkCase) -> dict[str, float | str]:
    stations, diameters, names, is_added = aug.station_table_from_case(case)
    baselines = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            baselines.append(float(np.linalg.norm(stations[j] - stations[i])))
    hub = np.asarray(case.hub_km, dtype=float)
    hub_dist = np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-rml.FIBER_LOSS_DB_PER_KM * aug.FIBER_LENGTH_SCALE * hub_dist / 10.0)
    return {
        "case": case.key,
        "coverage_score": float(case.optimization_score),
        "hub_x_km": float(case.hub_km[0]),
        "hub_y_km": float(case.hub_km[1]),
        "baseline_min_km": float(np.min(baselines)),
        "baseline_median_km": float(np.median(baselines)),
        "baseline_max_km": float(np.max(baselines)),
        "eta_min": float(np.min(eta)),
        "eta_aperture_weighted": float(np.sum(diameters**2 * eta) / np.sum(diameters**2)),
    }


def run_direct_rml(case: aug.NetworkCase) -> dict:
    print(f"[scan-rml] {case.key} score={case.optimization_score:.3f}", flush=True)
    bands, stats, truth, axis_uas = rml.simulate_case(case)
    prior = rml.broad_gaussian_prior(axis_uas)
    image, history = rml.weak_prior_rml(bands, case, RML_STRATEGY, prior, truth, axis_uas)
    metric = rml.metrics_for(image, truth, axis_uas)
    return {
        "case": case,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "image": image,
        "metric": metric,
        "history": history,
        "layout": summarize_layout(case),
    }


def plot_scan(results: list[dict], tag: str) -> tuple[Path, Path]:
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(8.8, max(2.4 * n, 3.0)), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]
    image_axes = []
    for row, result in enumerate(results):
        case = result["case"]
        stations, _, names, is_added = aug.station_table_from_case(case)
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]

        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=54, c="#1f77b4", label="existing")
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=62, c="#d62728", marker="^", label="remote")
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=80, c="#ffb000", marker="*", label="hub")
        for name, x, y in zip(names, stations[:, 0], stations[:, 1]):
            if name.startswith("new"):
                ax.text(x, y, name.replace("new 5 m ", ""), fontsize=5.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{case.key}\nscore={case.optimization_score:.2f}")
        ax.set_xlabel("x east (km)")
        ax.set_ylabel("y north (km)")
        if row == 0:
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[row, 1]
        ax.imshow(opt.normalize_blr_display(result["truth"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title("Input")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

        ax = axes[row, 2]
        metric = result["metric"]
        ax.imshow(opt.normalize_blr_display(result["image"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(
            f"{RML_STRATEGY} RML\n"
            f"global={metric['global_corr']:.2f}, BLR={metric['blr_corr']:.2f}, ring={metric['ring_contrast']:.2f}"
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.015,
        pad=0.01,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.5)
    fig.suptitle(
        f"Hawaii+4 remote-station scan, {ngc.NGC4151.name}, {rml.OBSERVING_DAYS} d, strategy={RML_STRATEGY}",
        fontsize=9.4,
        weight="bold",
    )
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_outputs(cases: list[aug.NetworkCase], results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path, Path]:
    layout_csv = OUTFIG / f"{tag}_layouts.csv"
    metric_csv = OUTFIG / f"{tag}_metrics.csv"
    json_path = OUTFIG / f"{tag}_summary.json"

    layout_rows = [summarize_layout(case) for case in cases]
    with layout_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(layout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layout_rows)

    metric_rows = []
    for result in results:
        metric_rows.append({"case": result["case"].key, "strategy": RML_STRATEGY, **result["metric"], **result["layout"]})
    with metric_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    payload = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "layout_csv": str(layout_csv),
        "metrics_csv": str(metric_csv),
        "strategy": RML_STRATEGY,
        "n_trials": N_TRIALS,
        "run_top": RUN_TOP,
        "rml_n_pix": rml.N_RML,
        "rml_n_iter": rml.N_ITER,
        "observing_days": rml.OBSERVING_DAYS,
        "n_time_windows": rml.N_TIME_WINDOWS,
        "exposure_s": rml.EXPOSURE_S,
        "fiber_loss_db_per_km": rml.FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": rml.MODE_FALSE_POSITIVE,
        "results": [
            {
                "case": result["case"].key,
                "metric": result["metric"],
                "layout": result["layout"],
                "history": result["history"],
                "hub_km": list(result["case"].hub_km),
                "stations": [
                    {
                        "name": tel.name,
                        "x_km": tel.x_km,
                        "y_km": tel.y_km,
                        "diameter_m": tel.diameter_m,
                        "is_added": tel.is_added,
                    }
                    for tel in result["case"].telescopes
                ],
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return layout_csv, metric_csv, json_path


def main() -> None:
    cases = candidate_cases()
    print("[scan] candidate layouts ranked by coverage/loss score:")
    for case in cases[:12]:
        summary = summarize_layout(case)
        print(
            f"  {case.key:32s} score={case.optimization_score:.3f} "
            f"bmax={summary['baseline_max_km']:.1f} eta_min={summary['eta_min']:.3f}",
            flush=True,
        )
    selected = cases[:RUN_TOP]
    # Always include the current manuscript-like Hawaii+4 baseline for comparison.
    baseline = existing_subset_case("hawaii4_current_drop20km", [0, 1, 2, 3])
    selected_map = {baseline.key: baseline}
    selected_map.update({case.key: case for case in selected})
    selected = list(selected_map.values())
    results = [run_direct_rml(case) for case in selected]
    tag = (
        f"hawaii4_remote_scan_{RML_STRATEGY}_{ngc.NGC4151.key}_{rml.OBSERVING_DAYS}d_"
        f"loss{rml.FIBER_LOSS_DB_PER_KM:g}_fp{rml.MODE_FALSE_POSITIVE:g}_n{rml.N_RML}_top{RUN_TOP}"
    ).replace(".", "p")
    pdf, png = plot_scan(results, tag)
    layout_csv, metric_csv, json_path = write_outputs(cases, results, tag, pdf, png)
    print(pdf)
    print(png)
    print(layout_csv)
    print(metric_csv)
    print(json_path)
    for result in results:
        print(result["case"].key, result["metric"])


if __name__ == "__main__":
    main()
