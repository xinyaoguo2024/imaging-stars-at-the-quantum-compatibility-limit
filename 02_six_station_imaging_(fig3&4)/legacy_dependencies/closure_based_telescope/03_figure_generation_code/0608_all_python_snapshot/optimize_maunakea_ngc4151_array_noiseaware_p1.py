from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_augmented_existing_telescope_ngc_sources_noiseaware_p1 as noiseaware
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_clean as base
from make_hawaii_optical_overview_figure import CLUSTER_CENTER, VISIBLE_400_800
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

N_ADDED = int(__import__("os").environ.get("MAUNAKEA_OPT_N_ADDED", "4"))
N_TRIALS = int(__import__("os").environ.get("MAUNAKEA_OPT_N_TRIALS", "24000"))
RADIUS_RANGE_KM = (6.0, 19.0)
MIN_SEPARATION_KM = 5.0
MAX_BASELINE_SOFT_KM = 35.0
RNG_SEED = 20260516 + 4151


def maunakea_existing_core() -> list[aug.Telescope]:
    center = CLUSTER_CENTER["Maunakea"]
    apertures = {
        "Keck I": 10.0,
        "Keck II": 10.0,
        "Subaru": 8.2,
        "Gemini North": 8.1,
        "CFHT": 3.6,
    }
    existing = []
    for name, lat, lon, _, cluster in VISIBLE_400_800:
        if cluster != "Maunakea" or name not in apertures:
            continue
        x, y = aug.xy_km(lat, lon, center)
        existing.append(aug.Telescope(name, x, y, apertures[name], False))
    return existing


def fixed_core_hub(existing: list[aug.Telescope]) -> np.ndarray:
    pos = np.array([[t.x_km, t.y_km] for t in existing], dtype=float)
    diam = np.array([t.diameter_m for t in existing], dtype=float)
    return np.average(pos, axis=0, weights=diam**2)


def entropy_score(hist: np.ndarray) -> float:
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    p = hist.astype(float).ravel() / total
    occupied = p > 0.0
    if not np.any(occupied):
        return 0.0
    return float(-np.sum(p[occupied] * np.log(p[occupied])) / math.log(len(p)))


def coverage_objective(
    stations: np.ndarray,
    diameters: np.ndarray,
    hub: np.ndarray,
    *,
    latitude_deg: float,
    declination_deg: float,
) -> tuple[float, dict[str, float]]:
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    lengths = np.linalg.norm(baselines, axis=1)
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)

    uv_parts = []
    for lam_nm in (400.0, 500.0, 650.0, 800.0):
        uu, vv = project_enu_baselines(
            baselines,
            hour_angles,
            lam_nm * 1e-9,
            latitude_deg=latitude_deg,
            declination_deg=declination_deg,
        )
        uv_parts.append(np.column_stack([uu.reshape(-1) / 1e9, vv.reshape(-1) / 1e9]))
        uv_parts.append(np.column_stack([-uu.reshape(-1) / 1e9, -vv.reshape(-1) / 1e9]))
    uv = np.vstack(uv_parts)
    u = uv[:, 0]
    v = uv[:, 1]
    r = np.sqrt(u**2 + v**2)
    theta = np.mod(np.arctan2(v, u), np.pi)

    # A fixed reconstruction box makes candidates comparable.  The 90 Glambda
    # range comfortably includes 400 nm tracks from ~35 km baselines.
    max_g = 90.0
    in_box = (np.abs(u) <= max_g) & (np.abs(v) <= max_g)
    hist2d = np.histogram2d(
        u[in_box],
        v[in_box],
        bins=(np.linspace(-max_g, max_g, 25), np.linspace(-max_g, max_g, 25)),
    )[0]
    radial_hist = np.histogram(r[(r > 1.5) & (r < max_g)], bins=np.linspace(1.5, max_g, 20))[0]
    angular_hist = np.histogram(theta[(r > 8.0) & (r < max_g)], bins=np.linspace(0.0, np.pi, 25))[0]
    occ2d = float(np.mean(hist2d > 0.0))
    radial_occ = float(np.mean(radial_hist > 0.0))
    angular_occ = float(np.mean(angular_hist > 0.0))
    ent2d = entropy_score(hist2d)
    entr = entropy_score(radial_hist)
    enta = entropy_score(angular_hist)
    umax = float(np.max(np.abs(u)))
    vmax = float(np.max(np.abs(v)))
    isotropy = min(umax, vmax) / max(umax, vmax, 1e-9)

    # Baseline diversity in physical space helps prevent many tracks being
    # copies of the same projected ellipse.
    long = lengths > 6.0
    btheta = np.mod(np.arctan2(baselines[:, 1], baselines[:, 0]), np.pi)
    bhist = np.histogram2d(
        lengths[long],
        btheta[long],
        bins=(np.linspace(6.0, 38.0, 12), np.linspace(0.0, np.pi, 13)),
    )[0]
    baseline_diversity = entropy_score(bhist)
    baseline_occ = float(np.mean(bhist > 0.0))
    mid_count = float(np.sum((lengths > 5.0) & (lengths < 14.0)))
    long_count = float(np.sum((lengths >= 14.0) & (lengths < 32.0)))

    effective_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_dist / 10.0)
    aperture_link = float(np.sum(diameters**2 * eta) / np.sum(diameters**2))
    min_link = float(np.min(eta))

    max_baseline = float(np.max(lengths))
    score = (
        2.35 * ent2d
        + 1.45 * entr
        + 1.30 * enta
        + 1.15 * occ2d
        + 0.70 * radial_occ
        + 0.65 * angular_occ
        + 1.00 * isotropy
        + 0.80 * baseline_diversity
        + 0.45 * baseline_occ
        + 0.035 * min(mid_count, 12.0)
        + 0.025 * min(long_count, 16.0)
        + 0.80 * aperture_link
        + 0.30 * min_link
        - 0.020 * max(max_baseline - MAX_BASELINE_SOFT_KM, 0.0) ** 2
    )
    metrics = {
        "score": float(score),
        "uv_entropy_2d": ent2d,
        "uv_occupancy_2d": occ2d,
        "radial_entropy": entr,
        "radial_occupancy": radial_occ,
        "angular_entropy": enta,
        "angular_occupancy": angular_occ,
        "uv_isotropy": isotropy,
        "baseline_diversity": baseline_diversity,
        "baseline_occupancy": baseline_occ,
        "baseline_median_km": float(np.median(lengths)),
        "baseline_max_km": max_baseline,
        "station_link_eff_min": min_link,
        "station_link_eff_aperture_weighted": aperture_link,
        "coverage_400_800_umax_g_lambda": umax,
        "coverage_400_800_vmax_g_lambda": vmax,
    }
    return float(score), metrics


def sample_layout(rng: np.random.Generator, origin: np.ndarray) -> np.ndarray | None:
    # A deliberately asymmetric pattern: sample broad angles, but reject nearly
    # repeated spokes and close outstation pairs.
    angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, N_ADDED))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    min_gap_deg = 38.0 if N_ADDED <= 4 else 18.0
    if np.min(gaps) < np.deg2rad(min_gap_deg):
        return None
    radii = rng.uniform(RADIUS_RANGE_KM[0], RADIUS_RANGE_KM[1], size=N_ADDED)
    # Encourage a mix of intermediate and long arms instead of all stations on
    # one annulus.
    min_inner = 1 if N_ADDED <= 4 else 2
    min_outer = 1 if N_ADDED <= 4 else 2
    if np.sum(radii < 10.5) < min_inner or np.sum(radii > 14.0) < min_outer:
        return None
    pos = origin + np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    d = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1))
    d += np.eye(N_ADDED) * 1e9
    if float(np.min(d)) < MIN_SEPARATION_KM:
        return None
    return pos


def optimize_case() -> tuple[aug.NetworkCase, dict]:
    rng = np.random.default_rng(RNG_SEED)
    existing = maunakea_existing_core()
    existing_pos = np.array([[t.x_km, t.y_km] for t in existing], dtype=float)
    existing_diam = np.array([t.diameter_m for t in existing], dtype=float)
    origin = np.mean(existing_pos, axis=0)
    hub = fixed_core_hub(existing)
    center = CLUSTER_CENTER["Maunakea"]

    best_score = -np.inf
    best_pos: np.ndarray | None = None
    best_metrics: dict[str, float] = {}
    accepted = 0
    for _ in range(N_TRIALS):
        added = sample_layout(rng, origin)
        if added is None:
            continue
        accepted += 1
        stations = np.vstack([existing_pos, added])
        diameters = np.concatenate([existing_diam, np.full(N_ADDED, 5.0)])
        score, metrics = coverage_objective(
            stations,
            diameters,
            hub,
            latitude_deg=center[0],
            declination_deg=ngc.NGC4151.dec_deg,
        )
        if score > best_score:
            best_score = score
            best_pos = added
            best_metrics = metrics

    assert best_pos is not None
    telescopes = list(existing)
    for idx, pos in enumerate(best_pos, start=1):
        telescopes.append(aug.Telescope(f"optimized 5 m {idx}", float(pos[0]), float(pos[1]), 5.0, True))
    case = aug.NetworkCase(
        key=f"maunakea_plus{N_ADDED}_ngc4151_opt",
        title=f"Maunakea optical core + {N_ADDED} optimized 5 m outstations",
        latitude_deg=center[0],
        center_latlon=center,
        telescopes=telescopes,
        hub_km=(float(hub[0]), float(hub[1])),
        optimization_score=float(best_score),
    )
    payload = {
        "n_trials": N_TRIALS,
        "accepted_trials": accepted,
        "radius_range_km": RADIUS_RANGE_KM,
        "min_separation_km": MIN_SEPARATION_KM,
        "hub_km": list(case.hub_km),
        "metrics": best_metrics,
        "stations": [
            {
                "name": t.name,
                "x_km": t.x_km,
                "y_km": t.y_km,
                "diameter_m": t.diameter_m,
                "is_added": t.is_added,
            }
            for t in telescopes
        ],
    }
    return case, payload


def plot_layout_comparison(optimized: aug.NetworkCase, reference: aug.NetworkCase) -> tuple[Path, Path]:
    cases = [reference, optimized]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), constrained_layout=True)
    for row, case in enumerate(cases):
        stations, diameters, _, is_added = aug.station_table_from_case(case)
        ax = axes[row, 0]
        ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=34, color="#005f73", edgecolor="white", linewidth=0.4, label="existing", zorder=3)
        ax.scatter(stations[is_added, 0], stations[is_added, 1], s=44, marker="^", color="#ae2012", edgecolor="white", linewidth=0.4, label="new 5 m", zorder=3)
        ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=80, marker="*", color="#ca6702", label="hub", zorder=4)
        for i, (x, y) in enumerate(stations):
            ax.text(x + 0.25, y + 0.25, f"S{i+1}", fontsize=6.4)
        for i, j in base.edge_list(len(stations)):
            ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.84", lw=0.45, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(case.title)
        ax.set_xlabel("east (km)")
        ax.set_ylabel("north (km)")
        ax.legend(frameon=False, fontsize=6.4)

        ax = axes[row, 1]
        baselines = np.array([stations[j] - stations[i] for i, j in base.edge_list(len(stations))])
        hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
        for lam_nm, color, alpha in [(400.0, "#005f73", 0.45), (800.0, "#ee9b00", 0.42)]:
            uu, vv = project_enu_baselines(
                baselines,
                hour_angles,
                lam_nm * 1e-9,
                latitude_deg=case.latitude_deg,
                declination_deg=ngc.NGC4151.dec_deg,
            )
            ax.scatter(uu.reshape(-1) / 1e9, vv.reshape(-1) / 1e9, s=1.2, color=color, alpha=alpha, label=f"{lam_nm:.0f} nm")
            ax.scatter(-uu.reshape(-1) / 1e9, -vv.reshape(-1) / 1e9, s=1.2, color=color, alpha=0.5 * alpha)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("NGC 4151 uv coverage")
        ax.set_xlabel(r"$u$ (G$\lambda$)")
        ax.set_ylabel(r"$v$ (G$\lambda$)")
        ax.legend(frameon=False, fontsize=6.4)
    png = OUTFIG / f"maunakea_ngc4151_optimized_plus{N_ADDED}_layout_comparison.png"
    pdf = OUTFIG / f"maunakea_ngc4151_optimized_plus{N_ADDED}_layout_comparison.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    optimized, payload = optimize_case()
    layout_json = OUTFIG / f"maunakea_ngc4151_optimized_plus{N_ADDED}_layout.json"
    layout_json.write_text(json.dumps(payload, indent=2) + "\n")
    reference = noiseaware.maunakea_plus4_case_from_plus5(
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json"
    )
    comp_pdf, comp_png = plot_layout_comparison(optimized, reference)
    print(layout_json)
    print(comp_pdf)
    print(comp_png)
    print(json.dumps(payload["metrics"], indent=2))

    stats = noiseaware.run_case(optimized, ngc.NGC4151, q_values=[0.25, 0.50, 0.75])
    summary_json = OUTFIG / f"maunakea_ngc4151_optimized_plus{N_ADDED}_noiseaware_p1_summary.json"
    summary_json.write_text(json.dumps(stats, indent=2) + "\n")
    print(stats["figure_pdf"])
    print(stats["figure_png"])
    print(summary_json)
    print(json.dumps(stats["metrics_by_q"], indent=2))


if __name__ == "__main__":
    main()
