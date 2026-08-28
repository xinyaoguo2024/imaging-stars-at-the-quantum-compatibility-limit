from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

import make_eight_station_cfi_qfi_note_tables as cfi
import hawaii3_compact_case
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_clean as base


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
TABLE_DIR = ROOT / "output" / "tables"
PDF_DIR = ROOT / "output" / "pdf"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FALSE_POSITIVE = float(os.environ.get("BASELINE_FALSE_POSITIVE", str(base.PAIR_FALSE_POSITIVE)))
DIRECT_STATION_FALSE_POSITIVE = float(os.environ.get("DIRECT_STATION_FALSE_POSITIVE", "0.0"))
LOCK_MONTE_CARLO_DRAWS = int(os.environ.get("LOCK_MONTE_CARLO_DRAWS", "32"))
RESOLVABLE_SIGMA_RAD = float(os.environ.get("CLOSURE_RESOLVABLE_SIGMA_RAD", str(np.pi / 5.0)))

JSON_OUT = TABLE_DIR / "closure_loop_rms_snr1_table.json"
CSV_OUT = TABLE_DIR / "closure_loop_rms_snr1_table.csv"
HAWAII_JSON_OUT = TABLE_DIR / "closure_loop_rms_snr1_hawaii_top4_plus3_ngc4151.json"
HAWAII_CSV_OUT = TABLE_DIR / "closure_loop_rms_snr1_hawaii_top4_plus3_ngc4151.csv"
TEX_OUT = PDF_DIR / "closure_loop_rms_snr1_note.tex"
PDF_OUT = PDF_DIR / "closure_loop_rms_snr1_note.pdf"


def add_resolvability_flags(row: dict) -> dict:
    row = dict(row)
    row["resolvable_edge"] = bool(row["sigma_edge_rad"] < RESOLVABLE_SIGMA_RAD)
    row["resolvable_ideal_qfi"] = bool(row["sigma_ideal_closure_first_rad"] < RESOLVABLE_SIGMA_RAD)
    row["resolvable_workpoint_cfi"] = bool(row["sigma_workpoint_limited_closure_first_rad"] < RESOLVABLE_SIGMA_RAD)
    return row


def resolvable_label(row: dict) -> str:
    labels = []
    if row.get("resolvable_edge", False):
        labels.append("E")
    if row.get("resolvable_ideal_qfi", False):
        labels.append("Q")
    if row.get("resolvable_workpoint_cfi", False):
        labels.append("C")
    return ",".join(labels) if labels else "--"


def resolvable_count(rows: list[dict], key: str) -> int:
    return int(sum(1 for row in rows if row.get(key, False)))


def latex_number(value: float, *, sig: int = 3) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 100.0 or abs_value < 1e-2:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / 10.0**exponent
        return rf"${mantissa:.{sig}g}\times 10^{{{exponent}}}$"
    if abs_value >= 10.0:
        return f"{value:.{sig}g}"
    return f"{value:.{sig}g}"


def compute_fast_lock_covariances_with_false_positive(context: dict, split_fraction: float) -> list[np.ndarray]:
    """Fast-loop gauge covariance per ten-minute sample.

    The lock uses the same science photon flux and simultaneous edge-first splitting as the
    edge-first benchmark.  The covariance is built per time window by summing over wavelength
    channels, but not over observing nights.
    """
    stats = context["stats"]
    edges = context["edges"]
    n_station = len(context["station_link_eff"])
    n_time = int(stats["n_time_windows"])
    edge_lock_fisher = np.zeros((n_time, len(edges)), dtype=float)
    station_eff = context["station_link_eff"]
    station_noise = context["station_channel_noise"]

    for lam, lam_lo, lam_hi in zip(
        context["lambda_centers_m"],
        context["lambda_edges_m"][:-1],
        context["lambda_edges_m"][1:],
    ):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = base.source_mode_occupation(freq, diameter_m=base.TELESCOPE_DIAMETER_M)
        lock_modes = float(stats["exposure_s"]) * df
        uu_rows, vv_rows = cfi.project_enu_baselines(
            context["baseline_vectors_km"],
            context["hour_angles"],
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )
        for time_index, (u, v) in enumerate(zip(uu_rows, vv_rows)):
            vis = base.interp_vis(context["vgrid"], context["uv_axis"], u, v)
            nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * u_mode * np.sqrt(station_eff[i] * station_eff[j])
                load = split_fraction * (
                    station_eff[i] * u_mode
                    + station_eff[j] * u_mode
                    + station_noise[i]
                    + station_noise[j]
                ) + base.PAIR_FALSE_POSITIVE
                density = 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-300)
                edge_lock_fisher[time_index, edge_index] += lock_modes * density

    inc = cfi.incidence_matrix(edges, n_station)
    covariances: list[np.ndarray] = []
    for time_index in range(n_time):
        laplacian = inc.T @ (edge_lock_fisher[time_index, :, None] * inc)
        evals, evecs = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
        inv_evals = np.zeros_like(evals)
        positive = evals > 1e-12
        inv_evals[positive] = 1.0 / evals[positive]
        covariance = (evecs * inv_evals) @ evecs.T
        covariances.append(0.5 * (covariance + covariance.T))
    return covariances


def compute_loop_rows() -> dict:
    rng = np.random.default_rng(20260519)
    context = cfi.setup_from_stats()
    stats = context["stats"]
    edges = context["edges"]
    w_basis = context["w_basis"]
    station_eta = context["station_link_eff"]
    station_noise = context["station_channel_noise"]
    n_station = len(station_eta)
    n_closure = int(w_basis.shape[1])
    split_fraction = 1.0 / (n_station - 1.0)

    edge_index = {edge: index for index, edge in enumerate(edges)}
    loop_defs = []
    for i in range(1, n_station):
        for j in range(i + 1, n_station):
            loop_defs.append(
                {
                    "label": (0, i, j),
                    "edge_indices": (edge_index[(0, i)], edge_index[(i, j)], edge_index[(0, j)]),
                }
            )

    loop_fedge_total = np.zeros(n_closure)
    loop_fideal_total = np.zeros(n_closure)
    loop_fcfi_total = np.zeros(n_closure)
    loop_qfi_weight_total = np.zeros(n_closure)
    loop_edge_fi_total = np.zeros((n_closure, 3))

    lock_covariances = compute_fast_lock_covariances_with_false_positive(context, split_fraction)

    for _band_index, time_index, vis, u_mode, total_modes in cfi.accumulated_samples(context):
        nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
        station_load_edge = station_eta * u_mode + station_noise
        station_load_direct = station_eta * u_mode + station_noise + DIRECT_STATION_FALSE_POSITIVE

        for loop_index, loop_def in enumerate(loop_defs):
            a, b, c_station = loop_def["label"]
            e_ab, e_bc, e_ac = loop_def["edge_indices"]
            loop_edges = ((a, b, e_ab), (b, c_station, e_bc), (c_station, a, e_ac))

            f_edges = []
            for i, j, edge_id in loop_edges:
                signal = split_fraction * u_mode * np.sqrt(station_eta[i] * station_eta[j])
                load = split_fraction * (station_load_edge[i] + station_load_edge[j]) + base.PAIR_FALSE_POSITIVE
                f_edges.append(4.0 * signal**2 * nu_eff[edge_id] ** 2 / max(load, 1e-300))
            f_edges = np.maximum(np.array(f_edges, dtype=float), 1e-300)

            g_ab = u_mode * np.sqrt(station_eta[a] * station_eta[b]) * nu_eff[e_ab]
            g_bc = u_mode * np.sqrt(station_eta[b] * station_eta[c_station]) * nu_eff[e_bc]
            g_ca = u_mode * np.sqrt(station_eta[c_station] * station_eta[a]) * nu_eff[e_ac]

            fedge_density = 1.0 / np.sum(1.0 / f_edges)
            fideal_density = cfi.triangle_direct_fisher(
                g_ab,
                g_bc,
                g_ca,
                station_load_direct[a],
                station_load_direct[b],
                station_load_direct[c_station],
            )
            fcfi_density, qfi_density, _retention = cfi.triangle_cfi_ratio_from_covariance(
                rng,
                g_ab,
                g_bc,
                g_ca,
                station_load_direct[a],
                station_load_direct[b],
                station_load_direct[c_station],
                lock_covariances[time_index][np.ix_([a, b, c_station], [a, b, c_station])],
                draws=LOCK_MONTE_CARLO_DRAWS,
            )

            loop_fedge_total[loop_index] += total_modes * fedge_density
            loop_fideal_total[loop_index] += total_modes * fideal_density
            loop_fcfi_total[loop_index] += total_modes * fcfi_density
            loop_qfi_weight_total[loop_index] += total_modes * qfi_density
            loop_edge_fi_total[loop_index] += total_modes * f_edges

    rows = []
    for index, (loop_def, w_loop) in enumerate(zip(loop_defs, w_basis.T), start=1):
        label = loop_def["label"]
        fedge = max(float(loop_fedge_total[index - 1]), 1e-300)
        fideal = max(float(loop_fideal_total[index - 1]), 1e-300)
        fcfi = max(float(loop_fcfi_total[index - 1]), 1e-300)
        qfi_weight = max(float(loop_qfi_weight_total[index - 1]), 1e-300)
        sorted_edge_fi = np.sort(loop_edge_fi_total[index - 1])[::-1]
        chi2 = float(np.sqrt(sorted_edge_fi[1] / sorted_edge_fi[0]))
        chi3 = float(np.sqrt(sorted_edge_fi[2] / sorted_edge_fi[0]))
        sigma_edge = 1.0 / np.sqrt(fedge)
        sigma_ideal = 1.0 / np.sqrt(fideal)
        sigma_cfi = 1.0 / np.sqrt(fcfi)
        rows.append(
            add_resolvability_flags(
                {
                "index": index,
                "loop": f"S{label[0] + 1}-S{label[1] + 1}-S{label[2] + 1}",
                "chi2": chi2,
                "chi3": chi3,
                "sigma_edge_rad": float(sigma_edge),
                "sigma_ideal_closure_first_rad": float(sigma_ideal),
                "sigma_workpoint_limited_closure_first_rad": float(sigma_cfi),
                "gain_ideal_over_edge": float(sigma_edge / sigma_ideal),
                "gain_workpoint_limited_over_edge": float(sigma_edge / sigma_cfi),
                "cfi_over_qfi": float(fcfi / qfi_weight),
                }
            )
        )

    payload = {
        "definition": "per-loop RMS closure-phase errors at global SNR gain 1",
        "stats_path": str(cfi.STATS_PATH.relative_to(ROOT)),
        "station_layout_file": stats["station_layout_file"],
        "hub_km": stats["hub_km"],
        "global_snr_gain": 1.0,
        "fiber_loss_db_per_km": float(stats["fiber_loss_db_per_km"]),
        "fiber_length_scale": float(stats["fiber_length_scale"]),
        "baseline_false_positive": base.PAIR_FALSE_POSITIVE,
        "mode_false_positive": base.MODE_FALSE_POSITIVE,
        "direct_station_false_positive_equivalent": DIRECT_STATION_FALSE_POSITIVE,
        "resolvable_sigma_rad": RESOLVABLE_SIGMA_RAD,
        "resolvable_sigma_definition": "A loop is tagged resolvable if its one-sigma closure-phase RMS is below pi/5 rad by default.",
        "observing_days": int(stats["observing_days"]),
        "n_time_windows": int(stats["n_time_windows"]),
        "exposure_s": float(stats["exposure_s"]),
        "exposure_gap_s": float(stats["exposure_gap_s"]),
        "lambda_min_nm": float(stats["lambda_min_nm"]),
        "lambda_max_nm": float(stats["lambda_max_nm"]),
        "lambda_step_nm": float(stats["lambda_step_nm"]),
        "n_stations": n_station,
        "n_baselines": len(edges),
        "n_independent_closures": n_closure,
        "split_fraction": split_fraction,
        "lock_model": (
            "The workpoint-limited closure-first column propagates a fast-loop station-gauge "
            "covariance built from science-flux-equivalent split-baseline locking Fisher "
            "information within each ten-minute sample, summed over 400-800 nm but not over nights."
        ),
        "rows": rows,
    }
    return payload


def make_hawaii_top4_plus3_case() -> dict:
    """Use the same compact Hawaii+3 layout as the current PRL Fig. 3."""
    network = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    selected = [
        {
            "name": tel.name,
            "x_km": float(tel.x_km),
            "y_km": float(tel.y_km),
            "diameter_m": float(tel.diameter_m),
            "is_added": bool(tel.is_added),
        }
        for tel in network.telescopes
    ]
    return {
        "layout_path": "hawaii3_compact_case.py:make_hawaii3_compact_remote_case",
        "subset_rule": "current Fig. 3 compact 2/4/9 km outstation layout",
        "stations": selected,
        "hub_km": [float(network.hub_km[0]), float(network.hub_km[1])],
        "latitude_deg": float(network.latitude_deg),
        "source": "NGC 4151",
    }


def context_from_hawaii_case(case: dict) -> dict:
    stats = json.loads(cfi.STATS_PATH.read_text())
    stats = dict(stats)
    stats["station_layout_file"] = case["layout_path"]
    stats["hub_km"] = case["hub_km"]
    stats["array_latitude_deg"] = case["latitude_deg"]
    stats["source_declination_deg"] = ngc.NGC4151.dec_deg
    stats["fiber_loss_db_per_km"] = 0.2
    stats["fiber_length_scale"] = 0.75

    stations_km = np.array([[row["x_km"], row["y_km"]] for row in case["stations"]], dtype=float)
    diameters_m = np.array([row["diameter_m"] for row in case["stations"]], dtype=float)
    hub_km = np.array(case["hub_km"], dtype=float)
    edges = base.edge_list(len(stations_km))
    w_basis = base.root_cycle_basis(edges, len(stations_km))

    hub_distances_km = np.linalg.norm(stations_km - hub_km, axis=1)
    effective_hub_distances_km = float(stats["fiber_length_scale"]) * hub_distances_km
    station_link_eff = 10.0 ** (-float(stats["fiber_loss_db_per_km"]) * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, base.MODE_FALSE_POSITIVE)

    n_pix = int(stats["n_pix"])
    half_width_uas = float(stats["half_width_uas"])
    truth, _axis = base.make_source(n_pix, half_width_uas)
    vgrid, uv_axis = base.visibility_grid(truth, 2.0 * half_width_uas * base.UAS_TO_RAD)
    baseline_vectors_km = np.array([stations_km[j] - stations_km[i] for i, j in edges])

    lambda_edges_nm = np.arange(
        float(stats["lambda_min_nm"]),
        float(stats["lambda_max_nm"]) + 0.5 * float(stats["lambda_step_nm"]),
        float(stats["lambda_step_nm"]),
    )
    lambda_edges_nm[-1] = float(stats["lambda_max_nm"])
    lambda_edges_m = lambda_edges_nm * 1e-9
    lambda_centers_m = np.sqrt(lambda_edges_m[:-1] * lambda_edges_m[1:])
    hour_angles = cfi.realnight_hour_angles(
        int(stats["n_time_windows"]),
        float(stats["exposure_s"]),
        float(stats["exposure_gap_s"]),
    )

    return {
        "stats": stats,
        "case": case,
        "stations_km": stations_km,
        "diameters_m": diameters_m,
        "hub_km": hub_km,
        "edges": edges,
        "w_basis": w_basis,
        "station_link_eff": station_link_eff,
        "station_channel_noise": station_channel_noise,
        "baseline_vectors_km": baseline_vectors_km,
        "vgrid": vgrid,
        "uv_axis": uv_axis,
        "lambda_edges_m": lambda_edges_m,
        "lambda_centers_m": lambda_centers_m,
        "hour_angles": hour_angles,
    }


def accumulated_samples_station_u(context: dict):
    stats = context["stats"]
    diameters_m = context["diameters_m"]
    for band_index, (lam, lam_lo, lam_hi) in enumerate(
        zip(context["lambda_centers_m"], context["lambda_edges_m"][:-1], context["lambda_edges_m"][1:])
    ):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_station = np.array(
            [base.source_mode_occupation(freq, diameter_m=float(diameter)) for diameter in diameters_m],
            dtype=float,
        )
        total_modes = float(stats["exposure_s"]) * int(stats["observing_days"]) * df
        uu_rows, vv_rows = cfi.project_enu_baselines(
            context["baseline_vectors_km"],
            context["hour_angles"],
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )
        for time_index, (u, v) in enumerate(zip(uu_rows, vv_rows)):
            vis = base.interp_vis(context["vgrid"], context["uv_axis"], u, v)
            yield band_index, time_index, vis, u_station, total_modes


def compute_fast_lock_covariances_station_u(context: dict, split_fraction: float) -> list[np.ndarray]:
    stats = context["stats"]
    edges = context["edges"]
    n_station = len(context["station_link_eff"])
    n_time = int(stats["n_time_windows"])
    edge_lock_fisher = np.zeros((n_time, len(edges)), dtype=float)
    station_eff = context["station_link_eff"]
    station_noise = context["station_channel_noise"]
    diameters_m = context["diameters_m"]

    for lam, lam_lo, lam_hi in zip(
        context["lambda_centers_m"],
        context["lambda_edges_m"][:-1],
        context["lambda_edges_m"][1:],
    ):
        freq = base.C_LIGHT / lam
        freq_lo = base.C_LIGHT / lam_hi
        freq_hi = base.C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_station = np.array(
            [base.source_mode_occupation(freq, diameter_m=float(diameter)) for diameter in diameters_m],
            dtype=float,
        )
        lock_modes = float(stats["exposure_s"]) * df
        uu_rows, vv_rows = cfi.project_enu_baselines(
            context["baseline_vectors_km"],
            context["hour_angles"],
            lam,
            latitude_deg=float(stats["array_latitude_deg"]),
            declination_deg=float(stats["source_declination_deg"]),
        )
        for time_index, (u, v) in enumerate(zip(uu_rows, vv_rows)):
            vis = base.interp_vis(context["vgrid"], context["uv_axis"], u, v)
            nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
            for edge_index, (i, j) in enumerate(edges):
                signal = split_fraction * np.sqrt(station_eff[i] * station_eff[j] * u_station[i] * u_station[j])
                load = split_fraction * (
                    station_eff[i] * u_station[i]
                    + station_eff[j] * u_station[j]
                    + station_noise[i]
                    + station_noise[j]
                ) + base.PAIR_FALSE_POSITIVE
                density = 4.0 * signal**2 * nu_eff[edge_index] ** 2 / max(load, 1e-300)
                edge_lock_fisher[time_index, edge_index] += lock_modes * density

    inc = cfi.incidence_matrix(edges, n_station)
    covariances: list[np.ndarray] = []
    for time_index in range(n_time):
        laplacian = inc.T @ (edge_lock_fisher[time_index, :, None] * inc)
        evals, evecs = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
        inv_evals = np.zeros_like(evals)
        positive = evals > 1e-12
        inv_evals[positive] = 1.0 / evals[positive]
        covariances.append(0.5 * ((evecs * inv_evals) @ evecs.T + ((evecs * inv_evals) @ evecs.T).T))
    return covariances


def compute_loop_rows_station_u(context: dict, *, source_name: str) -> dict:
    rng = np.random.default_rng(20260520)
    stats = context["stats"]
    edges = context["edges"]
    w_basis = context["w_basis"]
    station_eta = context["station_link_eff"]
    station_noise = context["station_channel_noise"]
    n_station = len(station_eta)
    n_closure = int(w_basis.shape[1])
    split_fraction = 1.0 / (n_station - 1.0)

    edge_index = {edge: index for index, edge in enumerate(edges)}
    loop_defs = []
    for i in range(1, n_station):
        for j in range(i + 1, n_station):
            loop_defs.append(
                {
                    "label": (0, i, j),
                    "edge_indices": (edge_index[(0, i)], edge_index[(i, j)], edge_index[(0, j)]),
                }
            )

    loop_fedge_total = np.zeros(n_closure)
    loop_fideal_total = np.zeros(n_closure)
    loop_fcfi_total = np.zeros(n_closure)
    loop_qfi_weight_total = np.zeros(n_closure)
    loop_edge_fi_total = np.zeros((n_closure, 3))
    lock_covariances = compute_fast_lock_covariances_station_u(context, split_fraction)

    for _band_index, time_index, vis, u_station, total_modes in accumulated_samples_station_u(context):
        nu_eff = np.clip(np.abs(vis), 1e-4, 0.98)
        station_load_edge = station_eta * u_station + station_noise
        station_load_direct = station_eta * u_station + station_noise + DIRECT_STATION_FALSE_POSITIVE

        for loop_index, loop_def in enumerate(loop_defs):
            a, b, c_station = loop_def["label"]
            e_ab, e_bc, e_ac = loop_def["edge_indices"]
            loop_edges = ((a, b, e_ab), (b, c_station, e_bc), (c_station, a, e_ac))
            f_edges = []
            for i, j, edge_id in loop_edges:
                signal = split_fraction * np.sqrt(station_eta[i] * station_eta[j] * u_station[i] * u_station[j])
                load = split_fraction * (station_load_edge[i] + station_load_edge[j]) + base.PAIR_FALSE_POSITIVE
                f_edges.append(4.0 * signal**2 * nu_eff[edge_id] ** 2 / max(load, 1e-300))
            f_edges = np.maximum(np.array(f_edges, dtype=float), 1e-300)

            g_ab = np.sqrt(station_eta[a] * station_eta[b] * u_station[a] * u_station[b]) * nu_eff[e_ab]
            g_bc = (
                np.sqrt(station_eta[b] * station_eta[c_station] * u_station[b] * u_station[c_station])
                * nu_eff[e_bc]
            )
            g_ca = (
                np.sqrt(station_eta[c_station] * station_eta[a] * u_station[c_station] * u_station[a])
                * nu_eff[e_ac]
            )

            fedge_density = 1.0 / np.sum(1.0 / f_edges)
            fideal_density = cfi.triangle_direct_fisher(
                g_ab,
                g_bc,
                g_ca,
                station_load_direct[a],
                station_load_direct[b],
                station_load_direct[c_station],
            )
            fcfi_density, qfi_density, _retention = cfi.triangle_cfi_ratio_from_covariance(
                rng,
                g_ab,
                g_bc,
                g_ca,
                station_load_direct[a],
                station_load_direct[b],
                station_load_direct[c_station],
                lock_covariances[time_index][np.ix_([a, b, c_station], [a, b, c_station])],
                draws=LOCK_MONTE_CARLO_DRAWS,
            )
            loop_fedge_total[loop_index] += total_modes * fedge_density
            loop_fideal_total[loop_index] += total_modes * fideal_density
            loop_fcfi_total[loop_index] += total_modes * fcfi_density
            loop_qfi_weight_total[loop_index] += total_modes * qfi_density
            loop_edge_fi_total[loop_index] += total_modes * f_edges

    rows = []
    for index, loop_def in enumerate(loop_defs, start=1):
        label = loop_def["label"]
        fedge = max(float(loop_fedge_total[index - 1]), 1e-300)
        fideal = max(float(loop_fideal_total[index - 1]), 1e-300)
        fcfi = max(float(loop_fcfi_total[index - 1]), 1e-300)
        qfi_weight = max(float(loop_qfi_weight_total[index - 1]), 1e-300)
        sorted_edge_fi = np.sort(loop_edge_fi_total[index - 1])[::-1]
        sigma_edge = 1.0 / np.sqrt(fedge)
        sigma_ideal = 1.0 / np.sqrt(fideal)
        sigma_cfi = 1.0 / np.sqrt(fcfi)
        rows.append(
            add_resolvability_flags(
                {
                "index": index,
                "loop": f"S{label[0] + 1}-S{label[1] + 1}-S{label[2] + 1}",
                "chi2": float(np.sqrt(sorted_edge_fi[1] / sorted_edge_fi[0])),
                "chi3": float(np.sqrt(sorted_edge_fi[2] / sorted_edge_fi[0])),
                "sigma_edge_rad": float(sigma_edge),
                "sigma_ideal_closure_first_rad": float(sigma_ideal),
                "sigma_workpoint_limited_closure_first_rad": float(sigma_cfi),
                "gain_ideal_over_edge": float(sigma_edge / sigma_ideal),
                "gain_workpoint_limited_over_edge": float(sigma_edge / sigma_cfi),
                "cfi_over_qfi": float(fcfi / qfi_weight),
                }
            )
        )

    case = context["case"]
    return {
        "definition": "per-loop RMS closure-phase errors at global SNR gain 1",
        "case": "Hawaii top-four optical telescopes plus three 5 m remote stations",
        "source": source_name,
        "layout_path": case["layout_path"],
        "subset_rule": case["subset_rule"],
        "stations": case["stations"],
        "hub_km": case["hub_km"],
        "global_snr_gain": 1.0,
        "fiber_loss_db_per_km": float(stats["fiber_loss_db_per_km"]),
        "fiber_length_scale": float(stats["fiber_length_scale"]),
        "baseline_false_positive": base.PAIR_FALSE_POSITIVE,
        "mode_false_positive": base.MODE_FALSE_POSITIVE,
        "direct_station_false_positive_equivalent": DIRECT_STATION_FALSE_POSITIVE,
        "resolvable_sigma_rad": RESOLVABLE_SIGMA_RAD,
        "resolvable_sigma_definition": "A loop is tagged resolvable if its one-sigma closure-phase RMS is below pi/5 rad by default.",
        "source_morphology": getattr(ngc, "SOURCE_MORPHOLOGY", "default"),
        "observing_days": int(stats["observing_days"]),
        "n_time_windows": int(stats["n_time_windows"]),
        "exposure_s": float(stats["exposure_s"]),
        "exposure_gap_s": float(stats["exposure_gap_s"]),
        "lambda_min_nm": float(stats["lambda_min_nm"]),
        "lambda_max_nm": float(stats["lambda_max_nm"]),
        "lambda_step_nm": float(stats["lambda_step_nm"]),
        "n_stations": n_station,
        "n_baselines": len(edges),
        "n_independent_closures": n_closure,
        "split_fraction": split_fraction,
        "rows": rows,
    }


def compute_hawaii_payload() -> dict:
    case = make_hawaii_top4_plus3_case()
    with ngc.patched_source(ngc.NGC4151):
        context = context_from_hawaii_case(case)
        return compute_loop_rows_station_u(context, source_name=ngc.NGC4151.name)


def write_tables(payload: dict) -> None:
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    fieldnames = list(payload["rows"][0].keys())
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])


def write_hawaii_tables(payload: dict) -> None:
    HAWAII_JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    fieldnames = list(payload["rows"][0].keys())
    with HAWAII_CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])


def table_lines(rows: list[dict]) -> list[str]:
    lines = [
        r"\begin{longtable}{r l r r r r r r r c}",
        r"\caption{Per-loop closure phase RMS and gains at global SNR gain 1. RMS values are in radians. The Res. column marks strategies with $\sigma_\Phi<\pi/5$: E=edge, Q=direct QFI, C=direct CFI.}\\",
        r"\toprule",
        r"Idx & Loop & $\chi_2$ & $\chi_3$ & $\sigma_{\rm edge}$ & $\sigma_{\rm dir}^{\rm QFI}$ & $\sigma_{\rm dir}^{\rm CFI}$ & $G_{\rm QFI}$ & $G_{\rm CFI}$ & Res. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Idx & Loop & $\chi_2$ & $\chi_3$ & $\sigma_{\rm edge}$ & $\sigma_{\rm dir}^{\rm QFI}$ & $\sigma_{\rm dir}^{\rm CFI}$ & $G_{\rm QFI}$ & $G_{\rm CFI}$ & Res. \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            rf"{row['index']} & {row['loop']} & "
            rf"{latex_number(row['chi2'])} & {latex_number(row['chi3'])} & "
            rf"{latex_number(row['sigma_edge_rad'])} & "
            rf"{latex_number(row['sigma_ideal_closure_first_rad'])} & "
            rf"{latex_number(row['sigma_workpoint_limited_closure_first_rad'])} & "
            rf"{latex_number(row['gain_ideal_over_edge'])} & "
            rf"{latex_number(row['gain_workpoint_limited_over_edge'])} & "
            rf"{resolvable_label(row)} \\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return lines


def write_tex(payload: dict, hawaii_payload: dict) -> None:
    rows = payload["rows"]
    hawaii_rows = hawaii_payload["rows"]
    med_gain_ideal = float(np.median([row["gain_ideal_over_edge"] for row in rows]))
    med_gain_cfi = float(np.median([row["gain_workpoint_limited_over_edge"] for row in rows]))
    med_retention = float(np.median([row["cfi_over_qfi"] for row in rows]))
    hawaii_med_gain_ideal = float(np.median([row["gain_ideal_over_edge"] for row in hawaii_rows]))
    hawaii_med_gain_cfi = float(np.median([row["gain_workpoint_limited_over_edge"] for row in hawaii_rows]))
    hawaii_med_retention = float(np.median([row["cfi_over_qfi"] for row in hawaii_rows]))
    threshold = float(payload["resolvable_sigma_rad"])
    n_rows = len(rows)
    n_hawaii_rows = len(hawaii_rows)
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{amsmath}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large Closure-loop phase RMS at unit SNR gain}\\[3pt]",
        r"{\normalsize Two array benchmarks, per-loop independent closure readout}",
        r"\end{center}",
        r"\section*{Optimized eight-station benchmark}",
        r"\paragraph{Setup.}",
        (
            "The table reports one-sigma RMS errors for the 21 root-loop closure phases "
            r"$\Gamma_{1ij}=\phi_{1i}+\phi_{ij}-\phi_{1j}$ at global SNR gain 1. "
            rf"The array layout is \texttt{{\detokenize{{{payload['station_layout_file']}}}}}, with hub "
            rf"$({payload['hub_km'][0]:.1f},{payload['hub_km'][1]:.1f})\,\mathrm{{km}}$, "
            rf"{payload['observing_days']} observing nights, {payload['n_time_windows']} "
            rf"ten-minute samples per night, and "
            rf"${payload['lambda_min_nm']:.0f}-{payload['lambda_max_nm']:.0f}\,\mathrm{{nm}}$ in "
            rf"${payload['lambda_step_nm']:.0f}\,\mathrm{{nm}}$ bins. "
            rf"The fibre attenuation is ${payload['fiber_loss_db_per_km']:.1f}\,\mathrm{{dB/km}}$ "
            rf"with fibre lengths scaled by {payload['fiber_length_scale']:.2f}. "
            rf"Pure fibre loss is treated only as attenuation.  The independent mode-local "
            rf"false-positive/background occupancy is {payload['mode_false_positive']:.2f}; "
            rf"the additional pair-combiner false-positive load is "
            rf"{payload['baseline_false_positive']:.2f}."
        ),
        r"\paragraph{Definitions.}",
        (
            r"$\sigma_{\rm edge}$ is the post-processed edge-first RMS obtained from the "
            r"harmonic-sum Fisher information of the three split baseline phase estimates. "
            r"$\sigma_{\rm dir}^{\rm QFI}$ is the ideal closure-first RMS from the local "
            r"three-mode SLD/QFI for that loop. "
            r"$\sigma_{\rm dir}^{\rm CFI}$ uses the same closure-first sorter but averages over "
            r"residual station-gauge errors from the fast-loop covariance, so it is the "
            r"working-point-limited RMS. Gains are RMS gains relative to edge-first: "
            r"$G_{\rm QFI}=\sigma_{\rm edge}/\sigma_{\rm dir}^{\rm QFI}$ and "
            r"$G_{\rm CFI}=\sigma_{\rm edge}/\sigma_{\rm dir}^{\rm CFI}$. "
            r"No full-basis scheduling or rank-sharing penalty is applied here; this is a per-loop benchmark. "
            rf"We tag a loop as practically resolvable when $\sigma_\Phi < {threshold:.3f}\,\mathrm{{rad}}=\pi/5$."
        ),
        r"\paragraph{Summary.}",
        (
            rf"The median ideal closure-first gain is $G_{{\rm QFI}}={med_gain_ideal:.2f}$, "
            rf"while finite working-point sensitivity gives median "
            rf"$G_{{\rm CFI}}={med_gain_cfi:.2f}$. "
            rf"The median CFI/QFI retention of the programmed sorter is {med_retention:.2f}. "
            rf"Using the $\pi/5$ criterion, the resolvable-loop counts are "
            rf"{resolvable_count(rows, 'resolvable_edge')}/{n_rows} for edge-first, "
            rf"{resolvable_count(rows, 'resolvable_ideal_qfi')}/{n_rows} for ideal direct QFI, and "
            rf"{resolvable_count(rows, 'resolvable_workpoint_cfi')}/{n_rows} for workpoint-limited direct CFI. "
            "After station-gauge marginalization, weak-edge loops no longer acquire unbounded "
            "direct Fisher information; their absolute RMS must be read from the sigma columns."
        ),
        r"\scriptsize",
    ]
    lines.extend(table_lines(rows))
    lines.extend(
        [
            r"\normalsize",
            r"\section*{Hawaii top-four plus three remote stations}",
            r"\paragraph{Setup.}",
            (
                "This second benchmark uses the four largest 400-800 nm Maunakea optical "
                "telescopes (Keck I, Keck II, Subaru, and Gemini North) plus three new "
                "5 m remote stations in the same compact 2/4/9 km layout used in the current "
                "Fig.~3 RML benchmark.  The source is NGC 4151, using its frequency-dependent "
                "optical SED and BLR image model from the local "
                rf"NGC source module with morphology \texttt{{\detokenize{{{hawaii_payload.get('source_morphology', 'default')}}}}}.  "
                "Mixed apertures are included station by station in the photon occupations."
            ),
            r"\paragraph{Hawaii summary.}",
            (
                rf"For this 7-station Hawaii network there are {hawaii_payload['n_independent_closures']} "
                rf"root-loop closure phases.  The median ideal closure-first gain is "
                rf"$G_{{\rm QFI}}={hawaii_med_gain_ideal:.2f}$; finite workpoint sensitivity gives "
                rf"median $G_{{\rm CFI}}={hawaii_med_gain_cfi:.2f}$.  The median CFI/QFI retention "
                rf"is {hawaii_med_retention:.2f}.  With the $\pi/5$ RMS criterion, the resolvable-loop "
                rf"counts are {resolvable_count(hawaii_rows, 'resolvable_edge')}/{n_hawaii_rows} "
                rf"(edge-first), {resolvable_count(hawaii_rows, 'resolvable_ideal_qfi')}/{n_hawaii_rows} "
                rf"(direct QFI), and {resolvable_count(hawaii_rows, 'resolvable_workpoint_cfi')}/{n_hawaii_rows} "
                rf"(direct CFI)."
            ),
            r"\scriptsize",
        ]
    )
    lines.extend(table_lines(hawaii_rows))
    lines.extend(
        [
            r"\normalsize",
            r"\paragraph{Caveat.}",
            (
                "The working-point-limited CFI column assumes that the fast loop uses only the "
                "science flux in one ten-minute sample to estimate the station gauge frame. "
                "A brighter guide channel, a metrology-assisted phase reference, or averaging of "
                "station-frame information across nearby samples would reduce this penalty. "
                "Conversely, unmodelled instrumental drifts would increase it."
            ),
            r"\end{document}",
        ]
    )
    TEX_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    payload = compute_loop_rows()
    hawaii_payload = compute_hawaii_payload()
    write_tables(payload)
    write_hawaii_tables(hawaii_payload)
    write_tex(payload, hawaii_payload)
    print(JSON_OUT)
    print(CSV_OUT)
    print(HAWAII_JSON_OUT)
    print(HAWAII_CSV_OUT)
    print(TEX_OUT)


if __name__ == "__main__":
    main()
