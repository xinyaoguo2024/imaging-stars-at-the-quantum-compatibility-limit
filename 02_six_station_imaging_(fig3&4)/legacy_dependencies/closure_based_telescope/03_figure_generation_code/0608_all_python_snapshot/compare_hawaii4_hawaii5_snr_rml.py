from __future__ import annotations

import csv
import json
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

STRATEGY = "direct"
BOOSTS = tuple(float(x) for x in os.environ.get("HAWAII_SNR_BOOSTS", "1,3,10").split(","))
INCLUDE_BEST_H4 = os.environ.get("HAWAII_INCLUDE_BEST_H4", "1") == "1"
H4_SCAN_SUMMARY = OUTFIG / "hawaii4_remote_scan_direct_ngc4151_30d_loss0p2_fp0p05_n80_top3_summary.json"


def make_subset_case(key: str, keep_remote_indices: list[int]) -> aug.NetworkCase:
    full = case_lib.load_maunakea_case()
    core = [tel for tel in full.telescopes if not tel.is_added]
    remote = [tel for tel in full.telescopes if tel.is_added]
    telescopes = list(core) + [remote[i] for i in keep_remote_indices]
    case = aug.NetworkCase(
        key=key,
        title=key.replace("_", " "),
        latitude_deg=full.latitude_deg,
        center_latlon=full.center_latlon,
        telescopes=telescopes,
        hub_km=full.hub_km,
        optimization_score=0.0,
    )
    return replace(case, optimization_score=coverage_score(case))


def make_full_hawaii5_case() -> aug.NetworkCase:
    full = case_lib.load_maunakea_case()
    return replace(
        full,
        key="hawaii_top4_remote5_ngc4151",
        title="Maunakea top-four core + five remote 5 m stations",
        optimization_score=coverage_score(full),
    )


def make_best_h4_from_scan() -> aug.NetworkCase | None:
    if not H4_SCAN_SUMMARY.exists():
        return None
    data = json.loads(H4_SCAN_SUMMARY.read_text())
    best = None
    best_blr = -np.inf
    for result in data.get("results", []):
        if result["case"] == "hawaii4_current_drop20km":
            continue
        blr = float(result["metric"]["blr_corr"])
        if blr > best_blr:
            best_blr = blr
            best = result
    if best is None:
        return None
    telescopes = [
        aug.Telescope(
            station["name"],
            float(station["x_km"]),
            float(station["y_km"]),
            float(station["diameter_m"]),
            bool(station["is_added"]),
        )
        for station in best["stations"]
    ]
    case = aug.NetworkCase(
        key=f"{best['case']}_best_blr",
        title="Best Hawaii+4 scan candidate by BLR metric",
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=tuple(best["hub_km"]),
        optimization_score=0.0,
    )
    return replace(case, optimization_score=coverage_score(case))


def coverage_score(case: aug.NetworkCase) -> float:
    old_dec = aug.SOURCE_DEC_DEG
    old_n = aug.N_TIME_WINDOWS
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
        aug.N_TIME_WINDOWS = old_n
        aug.EXPOSURE_S = old_exp
        aug.EXPOSURE_GAP_S = old_gap


def build_cases() -> list[aug.NetworkCase]:
    cases = [
        make_subset_case("hawaii4_current_drop20km", [0, 1, 2, 3]),
        make_subset_case("hawaii4_keep20_drop5km", [1, 2, 3, 4]),
        make_full_hawaii5_case(),
    ]
    if INCLUDE_BEST_H4:
        best = make_best_h4_from_scan()
        if best is not None:
            cases.insert(2, best)
    return cases


def phase_sigma_summary(bands: list[dict[str, np.ndarray]]) -> dict[str, float]:
    sigma = np.concatenate([band[f"sigma_{STRATEGY}"] for band in bands])
    amp = np.concatenate([band["amp"] for band in bands])
    good = amp > np.percentile(amp, 50.0)
    return {
        "sigma_median_rad": float(np.median(sigma)),
        "sigma_p10_rad": float(np.percentile(sigma, 10.0)),
        "sigma_p90_rad": float(np.percentile(sigma, 90.0)),
        "sigma_goodamp_median_rad": float(np.median(sigma[good])) if np.any(good) else float("nan"),
        "amp_median": float(np.median(amp)),
        "amp_p90": float(np.percentile(amp, 90.0)),
    }


def layout_summary(case: aug.NetworkCase, stats: dict) -> dict[str, float | str]:
    stations, diameters, _, is_added = aug.station_table_from_case(case)
    baselines = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            baselines.append(float(np.linalg.norm(stations[j] - stations[i])))
    hub = np.asarray(case.hub_km, dtype=float)
    eta = 10.0 ** (-rml.FIBER_LOSS_DB_PER_KM * aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1) / 10.0)
    return {
        "case": case.key,
        "n_station": len(stations),
        "n_remote": int(np.count_nonzero(is_added)),
        "n_baseline": int(len(baselines)),
        "n_closure": int((len(stations) - 1) * (len(stations) - 2) / 2),
        "coverage_score": float(case.optimization_score),
        "baseline_max_km": float(np.max(baselines)),
        "baseline_median_km": float(np.median(baselines)),
        "eta_min": float(np.min(eta)),
        "eta_aperture_weighted": float(np.sum(diameters**2 * eta) / np.sum(diameters**2)),
        "u400_half_g_lambda": float(stats["coverage_400nm_half_range_g_lambda"]["u"]),
        "v400_half_g_lambda": float(stats["coverage_400nm_half_range_g_lambda"]["v"]),
        "u800_half_g_lambda": float(stats["coverage_800nm_half_range_g_lambda"]["u"]),
        "v800_half_g_lambda": float(stats["coverage_800nm_half_range_g_lambda"]["v"]),
    }


def run_condition(case: aug.NetworkCase, boost: float) -> dict:
    old_boost = rml.SNR_BOOST
    try:
        rml.SNR_BOOST = boost
        bands, stats, truth, axis_uas = rml.simulate_case(case)
        prior = rml.broad_gaussian_prior(axis_uas)
        image, history = rml.weak_prior_rml(bands, case, STRATEGY, prior, truth, axis_uas)
        metric = rml.metrics_for(image, truth, axis_uas)
        return {
            "case": case,
            "boost": boost,
            "bands": bands,
            "stats": stats,
            "truth": truth,
            "axis_uas": axis_uas,
            "image": image,
            "history": history,
            "metric": metric,
            "sigma": phase_sigma_summary(bands),
            "layout": layout_summary(case, stats),
        }
    finally:
        rml.SNR_BOOST = old_boost


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(results), 3, figsize=(8.7, 2.35 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = axes[None, :]
    image_axes = []
    for row, result in enumerate(results):
        case = result["case"]
        stations, _, _, is_added = aug.station_table_from_case(case)
        axis = result["axis_uas"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]

        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=44, color="#1f77b4", label="existing")
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=52, marker="^", color="#d62728", label="remote")
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=68, marker="*", color="#ffb000", label="hub")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{case.key}\nB={result['boost']:g}, score={case.optimization_score:.2f}, "
            f"eta_min={result['layout']['eta_min']:.2f}"
        )
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

        metric = result["metric"]
        sigma = result["sigma"]
        ax = axes[row, 2]
        ax.imshow(opt.normalize_blr_display(result["image"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(
            f"direct closure RML\nBLR={metric['blr_corr']:.2f}, global={metric['global_corr']:.2f}, "
            f"sigma50={sigma['sigma_median_rad']:.2f} rad"
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
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.4)
    fig.suptitle(
        f"Hawaii remote-station count vs SNR test, {ngc.NGC4151.name}, {rml.OBSERVING_DAYS} d",
        fontsize=9.5,
        weight="bold",
    )
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_tables(results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        rows.append(
            {
                "case": result["case"].key,
                "boost": result["boost"],
                **result["metric"],
                **result["sigma"],
                **result["layout"],
            }
        )
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
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
                "boost": result["boost"],
                "metric": result["metric"],
                "sigma": result["sigma"],
                "layout": result["layout"],
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
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    cases = build_cases()
    baseline = cases[0]
    full5 = [case for case in cases if case.key == "hawaii_top4_remote5_ngc4151"][0]
    results = []
    # Coverage test at true SNR.
    for case in cases:
        print(f"[coverage] {case.key}", flush=True)
        results.append(run_condition(case, 1.0))
    # SNR test on the current Hawaii+4 geometry and on the full Hawaii+5 geometry.
    for boost in BOOSTS:
        if abs(boost - 1.0) < 1e-12:
            continue
        print(f"[snr] {baseline.key} boost={boost:g}", flush=True)
        results.append(run_condition(baseline, boost))
        print(f"[snr] {full5.key} boost={boost:g}", flush=True)
        results.append(run_condition(full5, boost))

    tag = (
        f"hawaii4_vs_hawaii5_snr_rml_{ngc.NGC4151.key}_{rml.OBSERVING_DAYS}d_"
        f"loss{rml.FIBER_LOSS_DB_PER_KM:g}_fp{rml.MODE_FALSE_POSITIVE:g}_n{rml.N_RML}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_tables(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key, "boost", result["boost"], result["metric"], result["sigma"])


if __name__ == "__main__":
    main()
