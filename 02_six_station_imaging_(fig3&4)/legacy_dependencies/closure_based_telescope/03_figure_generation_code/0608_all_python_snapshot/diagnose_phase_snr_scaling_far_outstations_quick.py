from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"

WAVELENGTHS_NM = np.array([430.0, 500.0, 600.0, 720.0])
TIME_INDICES = np.array([0, 10, 20, 35, 50, 71])
FLUX_FACTORS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])


def case_from_stats(path: Path) -> aug.NetworkCase:
    stats = json.loads(path.read_text())
    telescopes = [
        aug.Telescope(
            station["name"],
            station["x_km"],
            station["y_km"],
            station["diameter_m"],
            station["is_added"],
        )
        for station in stats["stations"]
    ]
    return aug.NetworkCase(
        key=stats["case"],
        title=stats["title"],
        latitude_deg=stats["latitude_deg"],
        center_latlon=tuple(stats["center_latlon"]),
        telescopes=telescopes,
        hub_km=tuple(stats["hub_km"]),
        optimization_score=stats["optimization_score"],
    )


def psd_inverse(matrix: np.ndarray, rel_floor: float = 1e-12) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(matrix)
    vmax = max(float(np.max(vals)), 0.0)
    floor = max(vmax * rel_floor, 1e-300)
    inv_vals = 1.0 / np.maximum(vals, floor)
    return (vecs * inv_vals) @ vecs.T


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "p10": float(np.nanpercentile(arr, 10)),
        "median": float(np.nanpercentile(arr, 50)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def median_sigma_for_flux(case: aug.NetworkCase, flux_factor: float, observing_days: float = 30.0) -> dict[str, float]:
    stations, diameters, _, _ = aug.station_table_from_case(case)
    n_station = len(stations)
    edges = base.edge_list(n_station)
    baselines = np.array([stations[j] - stations[i] for i, j in edges])
    w_basis = base.root_cycle_basis(edges, n_station)
    q_basis = base.orthonormal_cycle_basis(w_basis)
    closure_rank_share = min(1.0, (n_station - 1.0) / w_basis.shape[1])
    split_fraction = 1.0 / (n_station - 1.0)

    truth, _ = base.make_source(aug.N_PIX, aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * aug.HALF_WIDTH_UAS * base.UAS_TO_RAD
    vgrid, uv_axis = base.visibility_grid(truth, fov_rad)
    hub = np.array(case.hub_km)
    eta = 10.0 ** (
        -aug.FIBER_LOSS_DB_PER_KM
        * aug.FIBER_LENGTH_SCALE
        * np.linalg.norm(stations - hub, axis=1)
        / 10.0
    )
    # Fibre propagation is pure attenuation; false positives/backgrounds are independent.
    eps = np.full_like(eta, getattr(aug, "MODE_FALSE_POSITIVE", getattr(base, "MODE_FALSE_POSITIVE", 0.05)))
    pair_false_positive = getattr(aug, "PAIR_FALSE_POSITIVE", getattr(base, "PAIR_FALSE_POSITIVE", 0.0))
    hours = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    selected_hours = hours[TIME_INDICES]

    edge_first_edge = []
    direct_edge = []
    edge_first_loop = []
    direct_loop = []

    for wavelength_nm in WAVELENGTHS_NM:
        lam = wavelength_nm * 1e-9
        freq = base.C_LIGHT / lam
        # Use the same 10-nm channel width as the imaging simulation.
        lo_nm = wavelength_nm - 5.0
        hi_nm = wavelength_nm + 5.0
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        total_modes = aug.EXPOSURE_S * observing_days * (freq_hi - freq_lo)
        u_station = flux_factor * aug.station_u_modes(freq, diameters)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            selected_hours,
            lam,
            latitude_deg=case.latitude_deg,
            declination_deg=aug.SOURCE_DEC_DEG,
        )
        for uu, vv in zip(uu_rows, vv_rows):
            vtrue = base.interp_vis(vgrid, uv_axis, uu, vv)
            amp = np.abs(vtrue)
            nu_eff = np.clip(amp, 1e-4, 0.98)
            fisher_split = np.zeros(len(edges), dtype=float)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * math.sqrt(eta[i] * eta[j] * u_station[i] * u_station[j])
                load = (
                    split_fraction * (eta[i] * u_station[i] + eta[j] * u_station[j] + eps[i] + eps[j])
                    + pair_false_positive
                )
                fisher_split[edge_index] = (
                    total_modes * 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-300)
                )
            sigma_split = 1.0 / np.sqrt(np.maximum(fisher_split, 1e-300))
            edge_first_edge.extend(sigma_split.tolist())
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            edge_first_loop.extend(np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0)).tolist())

            fisher_direct = (
                total_modes
                * aug.noisy_closure_fisher_station_u(vtrue, eta, eps, u_station, q_basis, edges)
                * closure_rank_share
            )
            cov_direct_edge = q_basis @ psd_inverse(fisher_direct) @ q_basis.T
            direct_edge.extend(np.sqrt(np.maximum(np.diag(cov_direct_edge), 0.0)).tolist())
            # In the imaging code, direct closure noise is represented as
            # projected edge-phase uncertainty; report the same diagnostic.
            direct_loop.extend(np.sqrt(np.maximum(np.diag(cov_direct_edge), 0.0)).tolist())

    return {
        "edge_first_raw_edge": float(np.median(edge_first_edge)),
        "edge_first_projected_edge": float(np.median(edge_first_loop)),
        "direct_projected_edge": float(np.median(direct_edge)),
    }


def analyze_case(stats_path: Path) -> dict:
    case = case_from_stats(stats_path)
    rows = []
    for flux_factor in FLUX_FACTORS:
        med = median_sigma_for_flux(case, flux_factor)
        med["flux_factor"] = float(flux_factor)
        rows.append(med)
    base_row = rows[0]
    requirements = {}
    for key in ("edge_first_raw_edge", "edge_first_projected_edge", "direct_projected_edge"):
        sigma = base_row[key]
        phase_snr_factor = sigma / 0.1
        requirements[key] = {
            "sigma_30d_rad": sigma,
            "phase_snr_factor_to_0p1": phase_snr_factor,
            "time_or_mode_factor_to_0p1": phase_snr_factor**2,
        }
        reached = [row for row in rows if row[key] <= 0.1]
        requirements[key]["flux_factor_grid_to_0p1"] = reached[0]["flux_factor"] if reached else None
    return {
        "case": case.key,
        "title": case.title,
        "flux_scan": rows,
        "requirements": requirements,
    }


def main() -> None:
    paths = [
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus3_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json",
    ]
    summary = {path.stem: analyze_case(path) for path in paths}
    out_path = OUTFIG / "phase_snr_scaling_far_outstations_quick.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
