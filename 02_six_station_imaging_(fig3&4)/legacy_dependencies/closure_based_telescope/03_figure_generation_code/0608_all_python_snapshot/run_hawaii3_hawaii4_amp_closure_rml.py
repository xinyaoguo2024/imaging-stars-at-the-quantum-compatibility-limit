from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import eht_style_weak_prior_closure_rml as weak_rml
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
import plot_prl_broadband_blr_optimized as opt


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)
DIRECT_MULTISTART = True


def make_hawaii_remote_case(remote_count: int) -> aug.NetworkCase:
    """Use the same Hawaii+3/+4 definition as the weak-prior scan."""
    full = amp_rml.load_maunakea_case()
    core = [tel for tel in full.telescopes if not tel.is_added]
    added = [tel for tel in full.telescopes if tel.is_added]
    if remote_count == 3:
        selected = added[-3:]
    elif remote_count == 4:
        farthest = max(
            added,
            key=lambda tel: np.linalg.norm(np.array([tel.x_km, tel.y_km]) - np.array(full.hub_km, dtype=float)),
        )
        selected = [tel for tel in added if tel.name != farthest.name]
    else:
        raise ValueError("remote_count must be 3 or 4")
    return aug.NetworkCase(
        key=f"hawaii_top4_remote{remote_count}_accurate_amp_ngc4151",
        title=f"Maunakea top-four core + {remote_count} remote 5 m stations; reliable amplitudes",
        latitude_deg=full.latitude_deg,
        center_latlon=full.center_latlon,
        telescopes=core + selected,
        hub_km=full.hub_km,
        optimization_score=full.optimization_score,
    )


def phase_sigma_summary(result: dict, strategy: str) -> dict[str, float]:
    sigma = np.concatenate([band[f"sigma_{strategy}"] for band in result["bands"]])
    amp = np.concatenate([band["amp"] for band in result["bands"]])
    good = amp > np.percentile(amp, 50.0)
    return {
        f"{strategy}_sigma_median_rad": float(np.median(sigma)),
        f"{strategy}_sigma_p90_rad": float(np.percentile(sigma, 90.0)),
        f"{strategy}_sigma_goodamp_median_rad": float(np.median(sigma[good])) if np.any(good) else float("nan"),
    }


def layout_summary(case: aug.NetworkCase) -> dict[str, float | str]:
    stations, diameters, _, is_added = aug.station_table_from_case(case)
    baselines = [
        float(np.linalg.norm(stations[j] - stations[i]))
        for i in range(len(stations))
        for j in range(i + 1, len(stations))
    ]
    hub = np.array(case.hub_km, dtype=float)
    eta = 10.0 ** (-amp_rml.FIBER_LOSS_DB_PER_KM * aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1) / 10.0)
    return {
        "case": case.key,
        "n_station": len(stations),
        "n_remote": int(np.count_nonzero(is_added)),
        "n_baseline": len(baselines),
        "n_closure": int((len(stations) - 1) * (len(stations) - 2) / 2),
        "baseline_max_km": float(np.max(baselines)),
        "baseline_median_km": float(np.median(baselines)),
        "eta_min": float(np.min(eta)),
        "eta_aperture_weighted": float(np.sum(diameters**2 * eta) / np.sum(diameters**2)),
    }


def data_objective(
    image: np.ndarray,
    bands: list[dict[str, np.ndarray]],
    case: aug.NetworkCase,
    strategy: str,
    axis_uas: np.ndarray,
    prior: np.ndarray,
) -> tuple[float, float, float]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    q_basis = base.orthonormal_cycle_basis(base.root_cycle_basis(edges, len(stations)))
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    uv_axis = np.fft.fftshift(np.fft.fftfreq(len(axis_uas), d=fov_rad / len(axis_uas)))
    sigma_key = f"sigmaq_{strategy}" if strategy != "all" and f"sigmaq_{strategy}" in bands[0] else f"sigma_{strategy}"
    sigma_values = np.concatenate([band[sigma_key] for band in bands])
    phase_floor = max(amp_rml.PHASE_FLOOR_RAD, float(np.nanmedian(sigma_values)) * 0.20)
    data_obj, amp_obj, phase_obj = amp_rml.amp_phase_objective(
        image,
        bands,
        strategy,
        q_basis,
        uv_axis,
        phase_floor,
        len(edges),
    )
    # Mild regularization tie-breaker only; the data terms decide the start.
    reg = 0.02 * float(np.mean((image / max(np.sum(image), 1e-30) - prior) ** 2))
    return data_obj + reg, amp_obj, phase_obj


def run_case_multistart(case: aug.NetworkCase) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    prior = amp_rml.broad_gaussian_prior(axis_uas)
    prior_start = amp_rml.project_flux_positive(prior, smooth_pix=0.0)
    dirty_starts = {
        "all": amp_rml.quick_dirty_start(bands, "all", truth),
        "split": amp_rml.quick_dirty_start(bands, "split", truth),
        "direct": amp_rml.quick_dirty_start(bands, "direct", truth),
    }
    images: dict[str, np.ndarray] = {"truth": truth, "prior": prior_start}
    histories: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, float]] = {"prior": amp_rml.metrics_for(prior_start, truth, axis_uas)}
    start_log: dict[str, dict[str, float | str]] = {}
    for strategy in ("all", "split", "direct"):
        if strategy == "direct" and DIRECT_MULTISTART:
            starts = {
                "direct_dirty": dirty_starts["direct"],
                "split_dirty": dirty_starts["split"],
                "all_dirty": dirty_starts["all"],
                "prior": prior_start,
            }
        else:
            starts = {f"{strategy}_dirty": dirty_starts[strategy]}
        best = None
        for start_name, start in starts.items():
            print(f"[rml] {case.key} strategy={strategy} start={start_name}", flush=True)
            image, history = amp_rml.amplitude_closure_rml(
                bands,
                case,
                strategy,
                prior,
                start,
                fov_rad=fov_rad,
            )
            objective, amp_obj, phase_obj = data_objective(image, bands, case, strategy, axis_uas, prior)
            item = (objective, amp_obj, phase_obj, start_name, image, history)
            if best is None or item[0] < best[0]:
                best = item
        assert best is not None
        objective, amp_obj, phase_obj, start_name, image, history = best
        key = f"{strategy}_amp_phase_rml"
        images[key] = image
        histories[key] = history
        metrics[key] = amp_rml.metrics_for(image, truth, axis_uas)
        start_log[key] = {
            "selected_start": start_name,
            "data_objective": objective,
            "amp_objective": amp_obj,
            "phase_objective": phase_obj,
        }
    stats.update(
        {
            "source": amp_rml.SOURCE.name,
            "method": "accurate-amplitude RML with direct-closure multistart",
            "amplitude_assumption": "|V_ij| is treated as calibrated data with small fractional systematic error.",
            "amp_rel_sigma": amp_rml.AMP_REL_SIGMA,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "direct_multistart": DIRECT_MULTISTART,
            "selected_starts": start_log,
            "metrics": metrics,
            "histories": histories,
        }
    )
    return {
        "case": case,
        "stats": stats,
        "bands": bands,
        "truth": truth,
        "axis_uas": axis_uas,
        "images": images,
        "metrics": metrics,
    }


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(results), 4, figsize=(8.8, 2.45 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = axes[None, :]
    cols = [
        ("truth", "Input"),
        ("all_amp_phase_rml", "All vis. + drift"),
        ("split_amp_phase_rml", "Edge-first closure"),
        ("direct_amp_phase_rml", "Direct closure"),
    ]
    image_axes = []
    labels = {
        "hawaii_top4_remote3_accurate_amp_ngc4151": "Hawaii+3",
        "hawaii_top4_remote4_accurate_amp_ngc4151": "Hawaii+4",
    }
    for row, result in enumerate(results):
        axis_uas = result["axis_uas"]
        extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
        case = result["case"]
        for col, (key, title) in enumerate(cols):
            ax = axes[row, col]
            image = result["images"][key]
            ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
            if key == "truth":
                ax.set_title(title)
            else:
                metric = result["metrics"][key]
                ax.set_title(
                    f"{title}\n"
                    f"BLR={metric['blr_corr']:.2f}, global={metric['global_corr']:.2f}"
                )
            ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            if col == 0:
                ax.set_ylabel(f"{labels.get(case.key, case.key)}\n" + r"$\Delta\delta$ ($\mu$as)")
            else:
                ax.set_yticklabels([])
            image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.018,
        pad=0.012,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.6)
    fig.suptitle(
        (
            "Accurate-amplitude RML: per-baseline |V| constraints + closure-protected phases; "
            f"{amp_rml.SOURCE.name}, {amp_rml.OBSERVING_DAYS} d"
        ),
        fontsize=9.5,
        weight="bold",
    )
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_outputs(results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        layout = layout_summary(result["case"])
        sigma_summary = {}
        for strategy in ("all", "split", "direct"):
            sigma_summary.update(phase_sigma_summary(result, strategy))
        for key, metric in result["metrics"].items():
            rows.append({"case": result["case"].key, "image": key, **metric, **layout, **sigma_summary})
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "method": "RML with accurate per-baseline visibility amplitudes and closure-protected phases",
        "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
        "amp_rel_sigma": amp_rml.AMP_REL_SIGMA,
        "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
        "observing_days": amp_rml.OBSERVING_DAYS,
        "n_time_windows": amp_rml.N_TIME_WINDOWS,
        "exposure_s": amp_rml.EXPOSURE_S,
        "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
        "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
        "results": [
            {
                "case": result["case"].key,
                "layout": layout_summary(result["case"]),
                "metrics": result["metrics"],
                "stats": result["stats"],
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
    # Keep the noise model and source patching consistent with the latest weak-RML scripts.
    weak_rml.configure_simulation()
    cases = [make_hawaii_remote_case(3), make_hawaii_remote_case(4)]
    results = [run_case_multistart(case) for case in cases]
    tag = (
        f"hawaii3_hawaii4_accurate_amp_closure_rml_{amp_rml.SOURCE.key}_"
        f"{amp_rml.OBSERVING_DAYS}d_loss{amp_rml.FIBER_LOSS_DB_PER_KM:g}_"
        f"fp{amp_rml.MODE_FALSE_POSITIVE:g}_ampw{amp_rml.AMP_GRAD_WEIGHT:g}_n{amp_rml.N_RML}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key)
        for key in ("all_amp_phase_rml", "split_amp_phase_rml", "direct_amp_phase_rml"):
            print(" ", key, result["metrics"][key])


if __name__ == "__main__":
    main()
