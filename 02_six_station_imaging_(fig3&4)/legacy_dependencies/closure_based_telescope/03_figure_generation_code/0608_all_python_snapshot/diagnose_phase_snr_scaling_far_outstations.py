from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_prl_broadband_clean as base
from plot_prl_broadband_blr_realnight import project_enu_baselines, realnight_hour_angles


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"


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


def psd_inverse(matrix: np.ndarray, rel_floor: float = 1e-14) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(matrix)
    vmax = max(float(np.max(vals)), 0.0)
    floor = max(vmax * rel_floor, 1e-300)
    inv_vals = np.where(vals > floor, 1.0 / vals, np.inf)
    if np.any(~np.isfinite(inv_vals)):
        inv_vals = np.where(np.isfinite(inv_vals), inv_vals, 1.0 / floor)
    return (vecs * inv_vals) @ vecs.T


def triangle_vector(edges: list[tuple[int, int]], tri: tuple[int, int, int]) -> np.ndarray:
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    w = np.zeros(len(edges))
    i, j, k = tri
    for a, b, sign in ((i, j, 1.0), (j, k, 1.0), (k, i, 1.0)):
        if a < b:
            w[edge_index[(a, b)]] += sign
        else:
            w[edge_index[(b, a)]] -= sign
    return w


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "p10": float(np.nanpercentile(values, 10)),
        "median": float(np.nanpercentile(values, 50)),
        "p90": float(np.nanpercentile(values, 90)),
    }


def analyze_case(stats_path: Path, observing_days: int) -> dict:
    case = case_from_stats(stats_path)
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
    effective_hub_dist = aug.FIBER_LENGTH_SCALE * np.linalg.norm(stations - hub, axis=1)
    eta = 10.0 ** (-aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    # Fibre propagation is pure attenuation; false positives/backgrounds are independent.
    eps = np.full_like(eta, getattr(aug, "MODE_FALSE_POSITIVE", getattr(base, "MODE_FALSE_POSITIVE", 0.05)))
    pair_false_positive = getattr(aug, "PAIR_FALSE_POSITIVE", getattr(base, "PAIR_FALSE_POSITIVE", 0.0))

    hour_angles = realnight_hour_angles(aug.N_TIME_WINDOWS, aug.EXPOSURE_S, aug.EXPOSURE_GAP_S)
    lam_edges_nm = np.arange(aug.LAMBDA_MIN_NM, aug.LAMBDA_MAX_NM + 0.5 * aug.LAMBDA_STEP_NM, aug.LAMBDA_STEP_NM)
    lam_edges_nm[-1] = aug.LAMBDA_MAX_NM

    split_edge_sigmas = []
    split_cycle_sigmas = []
    direct_edge_sigmas = []
    direct_triangle_sigmas = []
    split_triangle_sigmas = []

    triangles = list(itertools.combinations(range(n_station), 3))
    triangle_ws = [triangle_vector(edges, tri) for tri in triangles]

    for lo_nm, hi_nm in zip(lam_edges_nm[:-1], lam_edges_nm[1:]):
        lam = math.sqrt(lo_nm * hi_nm) * 1e-9
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / (hi_nm * 1e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1e-9)
        df = freq_hi - freq_lo
        total_modes = aug.EXPOSURE_S * observing_days * df
        u_station = aug.station_u_modes(freq, diameters)
        uu_rows, vv_rows = project_enu_baselines(
            baselines,
            hour_angles,
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
                fisher_split[edge_index] = total_modes * 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-300)
            sigma_split_edge = 1.0 / np.sqrt(np.maximum(fisher_split, 1e-300))
            split_edge_sigmas.extend(sigma_split_edge.tolist())
            cov_split_cycle = q_basis.T @ ((sigma_split_edge**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            split_cycle_sigmas.extend(np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0)).tolist())

            fisher_direct = (
                total_modes
                * aug.noisy_closure_fisher_station_u(vtrue, eta, eps, u_station, q_basis, edges)
                * closure_rank_share
            )
            inv_direct = psd_inverse(fisher_direct)
            cov_direct_edge = q_basis @ inv_direct @ q_basis.T
            direct_edge_sigmas.extend(np.sqrt(np.maximum(np.diag(cov_direct_edge), 0.0)).tolist())

            for w in triangle_ws:
                a = q_basis.T @ w
                split_triangle_sigmas.append(math.sqrt(float(np.sum(w * w * sigma_split_edge**2))))
                direct_triangle_sigmas.append(math.sqrt(max(float(a @ inv_direct @ a), 0.0)))

    out = {
        "case": case.key,
        "title": case.title,
        "observing_days": observing_days,
        "n_station": n_station,
        "n_baseline": len(edges),
        "n_closure": int(w_basis.shape[1]),
        "closure_rank_share": float(closure_rank_share),
        "uncapped_sigma_rad": {
            "edge_first_raw_baseline": summarize(np.array(split_edge_sigmas)),
            "edge_first_projected_edge": summarize(np.array(split_cycle_sigmas)),
            "direct_projected_edge": summarize(np.array(direct_edge_sigmas)),
            "edge_first_triangle_closure": summarize(np.array(split_triangle_sigmas)),
            "direct_triangle_closure": summarize(np.array(direct_triangle_sigmas)),
        },
    }
    for key, summary in out["uncapped_sigma_rad"].items():
        out.setdefault("snr_factor_for_sigma_0p1", {})[key] = {
            name: value / 0.1 for name, value in summary.items()
        }
        out.setdefault("photon_budget_factor_for_sigma_0p1", {})[key] = {
            name: (value / 0.1) ** 2 for name, value in summary.items()
        }
    return out


def main() -> None:
    cases = [
        OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus3_far_stats.json",
        OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json",
    ]
    summary = {}
    for observing_days in (30, 90):
        for stats_path in cases:
            result = analyze_case(stats_path, observing_days)
            summary[f"{result['case']}_{observing_days}d"] = result
            print(json.dumps(result, indent=2))
    out_path = OUTFIG / "phase_snr_scaling_far_outstations.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
