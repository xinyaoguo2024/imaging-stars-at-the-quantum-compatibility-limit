from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_hawaii3_hawaii4_amp_closure_rml as hrun


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260520
N_TRIALS = 3500
STRATEGY = "direct"


def core_telescopes() -> list[aug.Telescope]:
    full = amp_rml.load_maunakea_case()
    return [tel for tel in full.telescopes if not tel.is_added]


def radial_mid_score(stations: np.ndarray, case_lat: float) -> float:
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    hour = aug.realnight_hour_angles(amp_rml.N_TIME_WINDOWS, amp_rml.EXPOSURE_S, amp_rml.EXPOSURE_GAP_S)
    q_all = []
    theta_all = []
    for lam_nm in (400.0, 500.0, 650.0, 800.0):
        uu, vv = aug.project_enu_baselines(
            baselines,
            hour,
            lam_nm * 1e-9,
            latitude_deg=case_lat,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        q = np.sqrt(uu.reshape(-1) ** 2 + vv.reshape(-1) ** 2) / 1e9
        theta = np.mod(np.arctan2(vv.reshape(-1), uu.reshape(-1)), np.pi)
        q_all.append(q)
        theta_all.append(theta)
    q = np.concatenate(q_all)
    theta = np.concatenate(theta_all)
    # Emphasize the first few ring oscillations for R_BLR ~ 60 uas.
    bins_mid = np.array([0.8, 1.3, 2.0, 3.0, 4.7, 6.5, 8.5, 12.0, 18.0])
    h_mid = np.histogram(q, bins=bins_mid)[0].astype(float)
    p_mid = h_mid / max(np.sum(h_mid), 1.0)
    occ_mid = np.mean(h_mid > 0)
    ent_mid = -float(np.sum(p_mid * np.log(p_mid + 1e-12))) / np.log(len(h_mid))
    h_ang = np.histogram(theta[(q > 1.0) & (q < 12.0)], bins=np.linspace(0.0, np.pi, 13))[0].astype(float)
    p_ang = h_ang / max(np.sum(h_ang), 1.0)
    occ_ang = np.mean(h_ang > 0)
    ent_ang = -float(np.sum(p_ang * np.log(p_ang + 1e-12))) / np.log(len(h_ang))
    q_hi = np.mean(q > 25.0)
    return float(2.2 * occ_mid + 1.5 * ent_mid + 1.1 * occ_ang + 0.8 * ent_ang - 0.35 * q_hi)


def make_mid_case(n_remote: int, radii: np.ndarray, *, key: str) -> aug.NetworkCase:
    core = core_telescopes()
    origin = np.mean(np.array([[tel.x_km, tel.y_km] for tel in core], dtype=float), axis=0)
    rng = np.random.default_rng(RNG_SEED + n_remote)
    best_score = -np.inf
    best_remote = None
    for _ in range(N_TRIALS):
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_remote))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        if np.min(gaps) < np.deg2rad(360.0 / n_remote * 0.45):
            continue
        remote = origin + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        stations = np.vstack([np.array([[tel.x_km, tel.y_km] for tel in core]), remote])
        score = radial_mid_score(stations, 19.8250)
        if score > best_score:
            best_score = score
            best_remote = remote
    assert best_remote is not None
    telescopes = list(core)
    for idx, (x, y) in enumerate(best_remote, start=1):
        telescopes.append(aug.Telescope(f"mid 5 m r={radii[idx-1]:g}km", float(x), float(y), 5.0, True))
    tmp = aug.NetworkCase(
        key=key,
        title=key.replace("_", " "),
        latitude_deg=19.8250,
        center_latlon=(19.8250, -155.4720),
        telescopes=telescopes,
        hub_km=(0.0, 0.0),
        optimization_score=best_score,
    )
    stations, diameters, _, _ = aug.station_table_from_case(tmp)
    hub, hub_score = aug.optimize_hub(stations, diameters)
    return replace(tmp, hub_km=(float(hub[0]), float(hub[1])), optimization_score=float(best_score + 0.1 * hub_score))


def q_histogram(case: aug.NetworkCase) -> dict[str, int]:
    stations, _, _, _ = aug.station_table_from_case(case)
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    hour = aug.realnight_hour_angles(amp_rml.N_TIME_WINDOWS, amp_rml.EXPOSURE_S, amp_rml.EXPOSURE_GAP_S)
    q_all = []
    for lam_nm in (400.0, 500.0, 600.0, 700.0, 800.0):
        uu, vv = aug.project_enu_baselines(
            baselines,
            hour,
            lam_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        q_all.append(np.sqrt(uu.reshape(-1) ** 2 + vv.reshape(-1) ** 2) / 1e9)
    q = np.concatenate(q_all)
    bins = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0])
    hist = np.histogram(q, bins=bins)[0]
    return {f"{bins[i]:g}-{bins[i+1]:g}": int(hist[i]) for i in range(len(hist))}


def run_direct_multistart(case: aug.NetworkCase) -> dict:
    print(f"[simulate] {case.key}", flush=True)
    bands, stats, truth, axis = amp_rml.simulate_case(case)
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    prior = amp_rml.broad_gaussian_prior(axis)
    starts = {
        "direct_dirty": amp_rml.quick_dirty_start(bands, "direct", truth),
        "split_dirty": amp_rml.quick_dirty_start(bands, "split", truth),
        "all_dirty": amp_rml.quick_dirty_start(bands, "all", truth),
        "prior": amp_rml.project_flux_positive(prior, smooth_pix=0.0),
    }
    best = None
    for name, start in starts.items():
        print(f"[rml] {case.key} start={name}", flush=True)
        image, history = amp_rml.amplitude_closure_rml(bands, case, STRATEGY, prior, start, fov_rad=fov_rad)
        objective, amp_obj, phase_obj = hrun.data_objective(image, bands, case, STRATEGY, axis, prior)
        item = (objective, name, image, history, amp_obj, phase_obj)
        if best is None or objective < best[0]:
            best = item
    assert best is not None
    objective, start_name, image, history, amp_obj, phase_obj = best
    metric = amp_rml.metrics_for(image, truth, axis)
    return {
        "case": case,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis,
        "image": image,
        "metric": metric,
        "selected_start": start_name,
        "objective": objective,
        "amp_objective": amp_obj,
        "phase_objective": phase_obj,
        "q_hist": q_histogram(case),
    }


def plot_results(results: list[dict], tag: str) -> tuple[Path, Path]:
    fig, axes = plt.subplots(len(results), 4, figsize=(10.0, 2.35 * len(results)), constrained_layout=True)
    if len(results) == 1:
        axes = axes[None, :]
    image_axes = []
    for row, result in enumerate(results):
        case = result["case"]
        stations, _, _, is_added = aug.station_table_from_case(case)
        axis = result["axis_uas"]
        extent = [axis[0], axis[-1], axis[0], axis[-1]]
        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=42, color="#1f77b4", label="existing")
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=48, marker="^", color="#d62728", label="remote")
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=62, marker="*", color="#ffb000", label="hub")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{case.key}\nscore={case.optimization_score:.2f}")
        ax.set_xlabel("x east (km)")
        ax.set_ylabel("y north (km)")
        if row == 0:
            ax.legend(fontsize=6)

        ax = axes[row, 1]
        bins = list(result["q_hist"].keys())
        vals = list(result["q_hist"].values())
        ax.bar(np.arange(len(vals)), vals, color="#669bbc")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(bins, rotation=55, ha="right", fontsize=5.5)
        ax.set_title("radial uv counts")
        ax.set_ylabel("samples")

        ax = axes[row, 2]
        ax.imshow(opt.normalize_blr_display(result["truth"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title("Input")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

        ax = axes[row, 3]
        metric = result["metric"]
        ax.imshow(opt.normalize_blr_display(result["image"]), origin="lower", extent=extent, cmap="inferno")
        ax.set_title(
            f"direct amp+closure RML\n"
            f"BLR={metric['blr_corr']:.2f}, global={metric['global_corr']:.2f}, start={result['selected_start']}"
        )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.012,
        pad=0.01,
    )
    cbar.set_label("normalized BLR-emphasis brightness", fontsize=6.5)
    fig.suptitle("Mid-baseline test for recovering the NGC 4151 BLR ring", fontsize=10.0, weight="bold")
    png = OUTFIG / f"{tag}.png"
    pdf = OUTFIG / f"{tag}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def write_outputs(results: list[dict], tag: str, pdf: Path, png: Path) -> tuple[Path, Path]:
    rows = []
    for result in results:
        row = {
            "case": result["case"].key,
            "selected_start": result["selected_start"],
            "objective": result["objective"],
            "amp_objective": result["amp_objective"],
            "phase_objective": result["phase_objective"],
            **result["metric"],
        }
        for key, value in result["q_hist"].items():
            row[f"uv_{key}_glambda"] = value
        rows.append(row)
    csv_path = OUTFIG / f"{tag}_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "figure_pdf": str(pdf),
        "figure_png": str(png),
        "metrics_csv": str(csv_path),
        "ring_relevant_scales": {
            "blr_radius_uas": ngc.NGC4151.blr_radius_uas,
            "thin_ring_j0_zeros_glambda": [1.32, 3.02, 4.73, 6.45],
        },
        "results": [
            {
                "case": result["case"].key,
                "metric": result["metric"],
                "selected_start": result["selected_start"],
                "q_hist": result["q_hist"],
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
                "hub_km": list(result["case"].hub_km),
            }
            for result in results
        ],
    }
    json_path = OUTFIG / f"{tag}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    current_h4 = hrun.make_hawaii_remote_case(4)
    mid_h4 = make_mid_case(4, np.array([1.6, 2.8, 5.0, 9.0]), key="hawaii_mid4_r1p6_2p8_5_9")
    mid_h5 = make_mid_case(5, np.array([1.4, 2.4, 4.0, 7.0, 11.0]), key="hawaii_mid5_r1p4_2p4_4_7_11")
    results = [run_direct_multistart(case) for case in (current_h4, mid_h4, mid_h5)]
    tag = (
        f"midbaseline_amp_closure_rml_{ngc.NGC4151.key}_{amp_rml.OBSERVING_DAYS}d_"
        f"ampw{amp_rml.AMP_GRAD_WEIGHT:g}_n{amp_rml.N_RML}"
    ).replace(".", "p")
    pdf, png = plot_results(results, tag)
    csv_path, json_path = write_outputs(results, tag, pdf, png)
    print(pdf)
    print(png)
    print(csv_path)
    print(json_path)
    for result in results:
        print(result["case"].key, result["metric"], result["q_hist"])


if __name__ == "__main__":
    main()
