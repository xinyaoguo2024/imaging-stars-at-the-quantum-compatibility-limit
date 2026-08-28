from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_clean as base
from make_hawaii_optical_overview_figure import CLUSTER_CENTER, VISIBLE_400_800
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)

RADII_KM = np.array([5.0, 8.75, 12.5, 16.25, 20.0])
N_TRIALS = int(__import__("os").environ.get("TOP4_PLUS5_TRIALS", "20000"))
RNG_SEED = 20260516 + 951


def top4_maunakea_core() -> list[aug.Telescope]:
    center = CLUSTER_CENTER["Maunakea"]
    apertures = {
        "Keck I": 10.0,
        "Keck II": 10.0,
        "Subaru": 8.2,
        "Gemini North": 8.1,
        "CFHT": 3.6,
    }
    rows = []
    for name, lat, lon, _, cluster in VISIBLE_400_800:
        if cluster != "Maunakea" or name not in apertures:
            continue
        x, y = aug.xy_km(lat, lon, center)
        rows.append(aug.Telescope(name, x, y, apertures[name], False))
    rows.sort(key=lambda t: t.diameter_m, reverse=True)
    return rows[:4]


def fixed_core_hub(core: list[aug.Telescope]) -> np.ndarray:
    pos = np.array([[t.x_km, t.y_km] for t in core], dtype=float)
    diam = np.array([t.diameter_m for t in core], dtype=float)
    return np.average(pos, axis=0, weights=diam**2)


def entropy_score(hist: np.ndarray) -> float:
    p = hist.astype(float).ravel()
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(hist.size))


def coverage_metrics(stations: np.ndarray, diameters: np.ndarray, hub: np.ndarray) -> dict[str, float]:
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    lengths = np.linalg.norm(baselines, axis=1)
    hour_angles = realnight_hour_angles(24, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    parts = []
    for lam_nm in (400.0, 650.0, 800.0):
        uu, vv = project_enu_baselines(
            baselines,
            hour_angles,
            lam_nm * 1e-9,
            latitude_deg=CLUSTER_CENTER["Maunakea"][0],
            declination_deg=ngc.NGC4151.dec_deg,
        )
        uv = np.column_stack([uu.reshape(-1) / 1e9, vv.reshape(-1) / 1e9])
        parts.append(uv)
        parts.append(-uv)
    uv = np.vstack(parts)
    u, v = uv[:, 0], uv[:, 1]
    r = np.hypot(u, v)
    th = np.mod(np.arctan2(v, u), np.pi)

    max_g = 90.0
    in_box = (np.abs(u) < max_g) & (np.abs(v) < max_g)
    h2 = np.histogram2d(u[in_box], v[in_box], bins=(np.linspace(-max_g, max_g, 21), np.linspace(-max_g, max_g, 21)))[0]
    hr = np.histogram(r[(r > 1.5) & (r < max_g)], bins=np.linspace(1.5, max_g, 20))[0]
    ha = np.histogram(th[(r > 8.0) & (r < max_g)], bins=np.linspace(0, np.pi, 25))[0]
    btheta = np.mod(np.arctan2(baselines[:, 1], baselines[:, 0]), np.pi)
    hbr = np.histogram(lengths[lengths > 1.0], bins=np.linspace(1.0, 38.0, 16))[0]
    hb = np.histogram2d(
        lengths[lengths > 1.0],
        btheta[lengths > 1.0],
        bins=(np.linspace(1.0, 38.0, 14), np.linspace(0, np.pi, 13)),
    )[0]
    eff_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * eff_dist / 10.0)
    metrics = {
        "uv_occ2d": float(np.mean(h2 > 0)),
        "uv_ent2d": entropy_score(h2),
        "radial_occ": float(np.mean(hr > 0)),
        "radial_ent": entropy_score(hr),
        "angular_occ": float(np.mean(ha > 0)),
        "angular_ent": entropy_score(ha),
        "baseline_occ": float(np.mean(hb > 0)),
        "baseline_ent": entropy_score(hb),
        "baseline_radial_occ": float(np.mean(hbr > 0)),
        "baseline_radial_ent": entropy_score(hbr),
        "uv_isotropy": float(min(np.max(np.abs(u)), np.max(np.abs(v))) / max(np.max(np.abs(u)), np.max(np.abs(v)))),
        "umax_g_lambda": float(np.max(np.abs(u))),
        "vmax_g_lambda": float(np.max(np.abs(v))),
        "baseline_min_km": float(np.min(lengths[lengths > 0.1])),
        "baseline_median_km": float(np.median(lengths)),
        "baseline_max_km": float(np.max(lengths)),
        "eta_min": float(np.min(eta)),
        "eta_aperture_weighted": float(np.sum(diameters**2 * eta) / np.sum(diameters**2)),
    }
    score = (
        1.7 * metrics["uv_ent2d"]
        + 3.0 * metrics["radial_ent"]
        + 1.0 * metrics["angular_ent"]
        + 0.8 * metrics["uv_occ2d"]
        + 1.4 * metrics["radial_occ"]
        + 0.65 * metrics["uv_isotropy"]
        + 1.4 * metrics["baseline_radial_ent"]
        + 0.8 * metrics["baseline_radial_occ"]
        + 0.45 * metrics["baseline_ent"]
        + 0.25 * metrics["baseline_occ"]
        + 0.7 * metrics["eta_aperture_weighted"]
        + 0.25 * metrics["eta_min"]
        - 0.03 * max(metrics["baseline_max_km"] - 35.0, 0.0) ** 2
    )
    metrics["score"] = float(score)
    return metrics


def optimize_angles() -> tuple[aug.NetworkCase, dict]:
    rng = np.random.default_rng(RNG_SEED)
    core = top4_maunakea_core()
    core_pos = np.array([[t.x_km, t.y_km] for t in core])
    core_diam = np.array([t.diameter_m for t in core])
    origin = np.average(core_pos, axis=0, weights=core_diam**2)
    hub = fixed_core_hub(core)
    best: tuple[float, np.ndarray, dict[str, float]] | None = None
    accepted = 0
    for _ in range(N_TRIALS):
        # Roll back to the earlier radius-stratified design: fixed outstation
        # radii from 5 to 20 km, with only the angular placement optimized.
        base_angles = np.linspace(0.0, 2.0 * np.pi, len(RADII_KM), endpoint=False)
        angles = base_angles + rng.uniform(-0.46, 0.46, size=len(RADII_KM)) + rng.uniform(0.0, 2.0 * np.pi)
        rng.shuffle(angles)
        positions = origin + np.column_stack([RADII_KM * np.cos(angles), RADII_KM * np.sin(angles)])
        d = np.sqrt(np.sum((positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1)) + np.eye(len(RADII_KM)) * 1e9
        if np.min(d) < 3.5:
            continue
        accepted += 1
        stations = np.vstack([core_pos, positions])
        diameters = np.concatenate([core_diam, np.full(len(RADII_KM), 5.0)])
        metrics = coverage_metrics(stations, diameters, hub)
        if best is None or metrics["score"] > best[0]:
            best = (metrics["score"], positions, metrics)
    assert best is not None
    _, positions, metrics = best
    telescopes = list(core)
    for idx, (radius, pos) in enumerate(zip(RADII_KM, positions), start=1):
        telescopes.append(aug.Telescope(f"new 5 m r={radius:g}km", float(pos[0]), float(pos[1]), 5.0, True))
    case = aug.NetworkCase(
        key="maunakea_top4_plus5_ngc4151",
        title="Maunakea top-four core + five 5 m outstations (5--20 km)",
        latitude_deg=CLUSTER_CENTER["Maunakea"][0],
        center_latlon=CLUSTER_CENTER["Maunakea"],
        telescopes=telescopes,
        hub_km=(float(hub[0]), float(hub[1])),
        optimization_score=float(metrics["score"]),
    )
    payload = {
        "description": "Maunakea four largest optical telescopes plus five 5 m stations with fixed radii from 5 to 20 km and optimized angles for NGC 4151 uv coverage.",
        "case_key": case.key,
        "case_title": case.title,
        "n_trials": N_TRIALS,
        "accepted_trials": accepted,
        "radii_km": RADII_KM.tolist(),
        "hub_km": list(case.hub_km),
        "metrics": metrics,
        "stations": [
            {
                "name": t.name,
                "x_km": t.x_km,
                "y_km": t.y_km,
                "diameter_m": t.diameter_m,
                "is_added": t.is_added,
            }
            for t in case.telescopes
        ],
    }
    return case, payload


def plot_case(case: aug.NetworkCase, payload: dict) -> tuple[Path, Path]:
    stations = np.array([[t.x_km, t.y_km] for t in case.telescopes])
    diameters = np.array([t.diameter_m for t in case.telescopes])
    is_added = np.array([t.is_added for t in case.telescopes])
    edges = base.edge_list(len(stations))
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    ax = axes[0]
    ax.scatter(stations[~is_added, 0], stations[~is_added, 1], s=44, color="#005f73", edgecolor="white", label="top-four core")
    ax.scatter(stations[is_added, 0], stations[is_added, 1], s=48, marker="^", color="#ae2012", edgecolor="white", label="new 5 m")
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=78, marker="*", color="#ca6702", label="hub")
    for i, (x, y) in enumerate(stations):
        ax.text(x + 0.18, y + 0.18, f"S{i+1}\n{diameters[i]:g}m", fontsize=6.0)
    for i, j in edges:
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.84", lw=0.45, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("station topology")
    ax.legend(frameon=False, fontsize=6.4)

    ax = axes[1]
    for lam_nm, color, alpha in [(400.0, "#005f73", 0.45), (800.0, "#ee9b00", 0.42)]:
        uu, vv = project_enu_baselines(
            baselines,
            hour_angles,
            lam_nm * 1e-9,
            latitude_deg=case.latitude_deg,
            declination_deg=ngc.NGC4151.dec_deg,
        )
        ax.scatter(uu.reshape(-1) / 1e9, vv.reshape(-1) / 1e9, s=1.15, color=color, alpha=alpha, label=f"{lam_nm:.0f} nm")
        ax.scatter(-uu.reshape(-1) / 1e9, -vv.reshape(-1) / 1e9, s=1.15, color=color, alpha=0.55 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("NGC 4151 uv coverage")
    ax.legend(frameon=False, fontsize=6.4)
    fig.suptitle("Maunakea top-four + five radius-stratified outstations (5--20 km)", fontsize=11.0, weight="bold")
    png = OUTFIG / "maunakea_top4_plus5_ngc4151_coverage.png"
    pdf = OUTFIG / "maunakea_top4_plus5_ngc4151_coverage.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    case, payload = optimize_angles()
    pdf, png = plot_case(case, payload)
    layout = OUTFIG / "maunakea_top4_plus5_ngc4151_layout.json"
    payload["figure_pdf"] = str(pdf)
    payload["figure_png"] = str(png)
    layout.write_text(json.dumps(payload, indent=2) + "\n")
    print(layout)
    print(pdf)
    print(png)
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
