#!/usr/bin/env python3
"""Finite-copy calibration helpers for each Fig. 3 wavelength band.

This is deliberately a one-working-point saturation proxy.  It compares
per-copy-normalized n_s=2 and n_s=3 copy-permutation-invariant joint PVMs with
the same working point's score-lift QLAN covariance.  The optimized finite
PVMs are not strict score-preserving lifts of the one-copy POVM; their only
role is to calibrate the global finite-depth factor requested for the Fig. 3
simulation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FIG2_CODE = HERE / "code"
PVM_CODE = HERE / "n4_n5_ns2_ns3_phase_pvm_scan_0715"
for path in (FIG2_CODE, PVM_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Match the existing twelve-seed 100-ms cache exactly.
os.environ["FIG2_EXPOSURE_S"] = "0.100"
os.environ["FIG2_N_TIME_WINDOWS"] = "36"
os.environ["FIG2_SAMPLE_CADENCE_S"] = "900.0"
os.environ["FIG2_EXPOSURE_GAP_S"] = "899.9"
os.environ["FIG2_AMPLITUDE_BRANCH_FRACTION"] = "0.5"
os.environ["FIG2_PHASE_BRANCH_FRACTION"] = "0.5"
os.environ["FIG2_COHERENT_BLOCK_SIZE"] = "3"
os.environ["FIG2_EXISTING_COUPLING"] = "1.0"
os.environ["FIG2_REMOTE_DIAMETER_M"] = "6.0"
os.environ["FIG2_COLLECTION_EFFICIENCY"] = "0.02"

import run_promoted_povm_rml as fig2  # noqa: E402
from optimize_phase_pvm_scan import (  # noqa: E402
    ReducedBlock,
    local_copy_tangents,
    make_blocks,
    tensor_power,
    unitary_retraction,
)


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(a) + np.asarray(a).T)


def qfi_matrix(rho: np.ndarray, tangents: list[np.ndarray]) -> np.ndarray:
    values, vectors = np.linalg.eigh(rho)
    transformed = [vectors.conj().T @ dot @ vectors for dot in tangents]
    result = np.zeros((len(tangents), len(tangents)))
    for e, left in enumerate(transformed):
        for f, right in enumerate(transformed):
            denom = values[:, None] + values[None, :]
            result[e, f] = float(
                np.sum(2.0 * left * right.T / denom).real
            )
    return sym(result)


def schur_target(
    fisher: np.ndarray, science: np.ndarray, nuisance: np.ndarray
) -> np.ndarray:
    if nuisance.size == 0:
        return sym(fisher[np.ix_(science, science)])
    jss = fisher[np.ix_(science, science)]
    jsn = fisher[np.ix_(science, nuisance)]
    jnn = fisher[np.ix_(nuisance, nuisance)]
    return sym(jss - jsn @ np.linalg.pinv(jnn, rcond=1.0e-11) @ jsn.T)


@dataclass
class WorkingPoint:
    rho: np.ndarray
    full_tangents: list[np.ndarray]
    science_qfi: np.ndarray
    single_covariance: np.ndarray
    qlan_covariance: np.ndarray
    science_transform_to_display: np.ndarray
    edges: list[tuple[int, int]]
    loop_labels: list[str]
    metadata: dict[str, object]


def make_working_point(band_index_override: int | None = None) -> WorkingPoint:
    fig2.configure_good_runtime()
    fig2.apply_sample_stress_runtime()
    fig2.val.closure_basis = fig2.latest_closure_basis
    case = fig2.make_six_station_case()
    stations, diameters, _, _ = fig2.aug.station_table_from_case(case)
    hub = np.asarray(case.hub_km, dtype=float)
    n_station = len(stations)
    edges = fig2.opt.base.edge_list(n_station)
    baselines = np.asarray([stations[j] - stations[i] for i, j in edges], dtype=float)
    base_q = fig2.opt.base.orthonormal_cycle_basis(
        fig2.opt.base.root_cycle_basis(edges, n_station)
    )
    selected_basis = fig2.latest_loop_basis(edges)
    piston_basis = fig2.receiver_design.incidence_basis(edges, n_station)
    parameter_map = fig2.receiver_design.mixed_parameter_map(base_q, piston_basis)

    truth, _ = fig2.opt.base.make_source(fig2.aug.N_PIX, fig2.aug.HALF_WIDTH_UAS)
    fov_rad = 2.0 * fig2.aug.HALF_WIDTH_UAS * fig2.opt.base.UAS_TO_RAD
    vgrid, uv_axis = fig2.opt.base.visibility_grid(truth, fov_rad)
    wavelength_source = getattr(fig2.opt.base, "make_source_at_wavelength_nm", None)

    effective_hub_dist = fig2.aug.FIBER_LENGTH_SCALE * np.linalg.norm(
        stations - hub, axis=1
    )
    eta = 10.0 ** (-fig2.aug.FIBER_LOSS_DB_PER_KM * effective_hub_dist / 10.0)
    station_noise = np.full(n_station, fig2.EPS_STATION_RUN, dtype=float)
    hour_angles = fig2.realnight_hour_angles(
        fig2.aug.N_TIME_WINDOWS,
        fig2.aug.EXPOSURE_S,
        fig2.aug.EXPOSURE_GAP_S,
    )
    wavelength_edges = fig2.wavelength_bin_edges_nm()
    centers = np.sqrt(wavelength_edges[:-1] * wavelength_edges[1:])
    if band_index_override is None:
        band_index = int(np.argmin(np.abs(centers - 650.0)))
    else:
        band_index = int(band_index_override)
        if not 0 <= band_index < len(centers):
            raise ValueError(
                f"band_index_override must lie in [0,{len(centers) - 1}]"
            )
    time_index = int(np.argmin(np.abs(hour_angles)))
    center_nm = float(centers[band_index])
    wavelength_m = center_nm * 1.0e-9
    frequency = fig2.opt.base.C_LIGHT / wavelength_m
    u_station = fig2.detected_station_mode_occupations(frequency, diameters)
    if callable(wavelength_source):
        band_truth, _ = wavelength_source(
            fig2.aug.N_PIX, fig2.aug.HALF_WIDTH_UAS, center_nm
        )
        band_vgrid, band_uv_axis = fig2.opt.base.visibility_grid(band_truth, fov_rad)
    else:
        band_vgrid, band_uv_axis = vgrid, uv_axis
    uu_rows, vv_rows = fig2.project_enu_baselines(
        baselines,
        hour_angles,
        wavelength_m,
        latitude_deg=case.latitude_deg,
        declination_deg=fig2.GOOD_SOURCE.dec_deg,
    )
    vtrue = fig2.opt.base.interp_vis(
        band_vgrid, band_uv_axis, uu_rows[time_index], vv_rows[time_index]
    )
    rho, raw_derivatives, occupation = (
        fig2.receiver_design.conditional_state_and_raw_derivatives(
            vtrue, eta, u_station, station_noise, edges
        )
    )
    full_tangents_array = np.einsum(
        "ip,ijk->pjk", parameter_map, raw_derivatives, optimize=True
    )
    full_tangents = [np.asarray(dot) for dot in full_tangents_array]
    n_edge = len(edges)
    n_closure = base_q.shape[1]
    science = np.arange(n_edge + n_closure)
    nuisance = np.arange(n_edge + n_closure, len(full_tangents))
    full_qfi = qfi_matrix(rho, full_tangents)
    science_qfi = schur_target(full_qfi, science, nuisance)

    optimal_effects, optimal_meta = fig2.receiver_design.locally_optimized_effects(
        rho,
        raw_derivatives,
        parameter_map,
        vtrue,
        edges,
        amplitude_fraction=0.5,
        phase_fraction=0.5,
    )
    _, single_fisher, lift = fig2.receiver_design.evaluate_effects_in_mixed_space(
        rho,
        raw_derivatives,
        parameter_map,
        optimal_effects,
        n_edge=n_edge,
        n_closure=n_closure,
    )
    science_transform = np.eye(n_edge + n_closure)
    science_transform[n_edge:, n_edge:] = selected_basis.T @ base_q

    loop_labels = [
        "-".join(str(index + 1) for index in triangle)
        for triangle in fig2.BALANCED_INDEPENDENT_TRIANGLES
    ]
    metadata = {
        "band_index": band_index,
        "time_index": time_index,
        "lambda_center_nm": center_nm,
        "hour_angle_rad": float(hour_angles[time_index]),
        "one_photon_occupation": float(occupation),
        "rho_eigenvalues": np.linalg.eigvalsh(rho).tolist(),
        "n_station": n_station,
        "n_edge_amplitude": n_edge,
        "n_closure_phase": n_closure,
        "n_piston_nuisance": int(nuisance.size),
        "singlecopy_povm_hash": optimal_meta["povm_hash"],
        "singlecopy_effect_count": len(optimal_effects),
        "singlecopy_lift_kappa": float(lift.kappa_operator),
        "existing_coupled_area_fraction": float(fig2.EXISTING_COUPLED_AREA_FRACTION),
        "remote_diameter_m": float(fig2.REMOTE_DIAMETER_M),
        "photon_collection_efficiency": float(fig2.PHOTON_COLLECTION_EFFICIENCY),
    }
    return WorkingPoint(
        rho=rho,
        full_tangents=full_tangents,
        science_qfi=science_qfi,
        single_covariance=np.linalg.inv(single_fisher),
        qlan_covariance=lift.covariance_v,
        science_transform_to_display=science_transform,
        edges=edges,
        loop_labels=loop_labels,
        metadata=metadata,
    )


@dataclass
class Evaluation:
    objective: float
    fisher: np.ndarray
    covariance: np.ndarray
    science_covariance: np.ndarray
    gradients: list[np.ndarray]


def evaluate(
    blocks: list[ReducedBlock],
    unitaries: list[np.ndarray],
    full_weight: np.ndarray,
    science_dimension: int,
) -> Evaluation:
    fisher = np.zeros_like(full_weight)
    probabilities: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for block, unitary in zip(blocks, unitaries, strict=True):
        p0 = np.einsum("iy,ij,jy->y", unitary.conj(), block.sigma, unitary).real
        d0 = np.asarray(
            [
                np.einsum("iy,ij,jy->y", unitary.conj(), dot, unitary).real
                for dot in block.tangents
            ]
        ).T
        fisher += block.multiplicity * np.einsum(
            "ye,yf,y->ef", d0, d0, 1.0 / p0
        )
        probabilities.append(p0)
        derivatives.append(d0)
    fisher = sym(fisher)
    covariance = np.linalg.inv(fisher)
    science_covariance = sym(covariance[:science_dimension, :science_dimension])
    objective = float(np.trace(full_weight @ covariance))
    sensitivity = covariance @ full_weight @ covariance
    gradients: list[np.ndarray] = []
    for block, unitary, p0, d0 in zip(
        blocks, unitaries, probabilities, derivatives, strict=True
    ):
        h = d0 @ sensitivity.T
        quadratic = np.einsum("ye,ye->y", d0, h)
        tangent_action = np.einsum(
            "ye,eij,jy->iy", h, block.tangents, unitary, optimize=True
        )
        euclidean = block.multiplicity * (
            -2.0 * tangent_action / p0[None, :]
            + (block.sigma @ unitary) * quadratic[None, :] / p0[None, :] ** 2
        )
        overlap = unitary.conj().T @ euclidean
        gradients.append(euclidean - unitary @ (overlap + overlap.conj().T) / 2.0)
    return Evaluation(objective, fisher, covariance, science_covariance, gradients)


def optimize(
    blocks: list[ReducedBlock],
    full_weight: np.ndarray,
    science_dimension: int,
    *,
    restarts: int,
    steps: int,
    seed: int,
) -> tuple[list[np.ndarray], Evaluation, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    best: tuple[list[np.ndarray], Evaluation] | None = None
    records: list[dict[str, float | int]] = []
    for restart in range(restarts):
        unitaries = [
            unitary_retraction(
                rng.normal(size=(block.dimension, block.dimension))
                + 1j * rng.normal(size=(block.dimension, block.dimension))
            )
            for block in blocks
        ]
        current = evaluate(blocks, unitaries, full_weight, science_dimension)
        rate = 0.05
        for iteration in range(steps):
            gradient_norm = float(
                np.sqrt(sum(np.linalg.norm(g) ** 2 for g in current.gradients))
            )
            trial_rate = rate
            accepted = False
            for _ in range(24):
                trial_unitaries = [
                    unitary_retraction(unitary - trial_rate * gradient)
                    for unitary, gradient in zip(
                        unitaries, current.gradients, strict=True
                    )
                ]
                try:
                    trial = evaluate(
                        blocks, trial_unitaries, full_weight, science_dimension
                    )
                except np.linalg.LinAlgError:
                    trial_rate *= 0.5
                    continue
                if trial.objective < current.objective - 1.0e-4 * trial_rate * gradient_norm**2:
                    unitaries = trial_unitaries
                    current = trial
                    rate = min(1.35 * trial_rate, 0.25)
                    accepted = True
                    break
                trial_rate *= 0.5
            if not accepted:
                break
        gradient_norm = float(
            np.sqrt(sum(np.linalg.norm(g) ** 2 for g in current.gradients))
        )
        records.append(
            {
                "restart": restart,
                "objective": current.objective,
                "gradient_norm": gradient_norm,
                "steps": iteration + 1,
            }
        )
        print(
            f"[optimize] restart={restart + 1}/{restarts} risk={current.objective:.8g} "
            f"grad={gradient_norm:.3e}",
            flush=True,
        )
        if best is None or current.objective < best[1].objective:
            best = ([unitary.copy() for unitary in unitaries], current)
    if best is None:
        raise RuntimeError("optimization failed")
    return best[0], best[1], records


def coordinate_rows(
    point: WorkingPoint,
    covariances: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    transformed = {
        key: sym(
            point.science_transform_to_display
            @ covariance
            @ point.science_transform_to_display.T
        )
        for key, covariance in covariances.items()
    }
    rows: list[dict[str, object]] = []
    labels = [f"A{i + 1}{j + 1}" for i, j in point.edges] + [
        f"phi_{label}" for label in point.loop_labels
    ]
    for index, label in enumerate(labels):
        row: dict[str, object] = {
            "index": index,
            "kind": "amplitude" if index < len(point.edges) else "closure_phase",
            "label": label,
        }
        for key, covariance in transformed.items():
            row[f"variance_{key}"] = float(covariance[index, index])
            row[f"marginal_fisher_{key}"] = float(1.0 / covariance[index, index])
        row["fisher_ratio_ns2_to_qlan"] = (
            row["marginal_fisher_ns2"] / row["marginal_fisher_qlan"]
        )
        row["fisher_ratio_ns3_to_qlan"] = (
            row["marginal_fisher_ns3"] / row["marginal_fisher_qlan"]
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts-ns2", type=int, default=8)
    parser.add_argument("--restarts-ns3", type=int, default=8)
    parser.add_argument("--steps-ns2", type=int, default=2500)
    parser.add_argument("--steps-ns3", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    point = make_working_point()
    science_dimension = point.science_qfi.shape[0]
    full_dimension = len(point.full_tangents)
    full_weight = np.zeros((full_dimension, full_dimension))
    full_weight[:science_dimension, :science_dimension] = point.science_qfi

    results: dict[int, tuple[list[np.ndarray], Evaluation, list[dict[str, float | int]]]] = {}
    for copies, restarts, steps in (
        (2, args.restarts_ns2, args.steps_ns2),
        (3, args.restarts_ns3, args.steps_ns3),
    ):
        sigma = tensor_power(point.rho, copies)
        tangents = local_copy_tangents(point.rho, point.full_tangents, copies)
        blocks = make_blocks(6, copies, sigma, tangents)
        print(
            f"[setup] n_s={copies} raw={6**copies} blocks="
            f"{[(block.label, block.dimension, block.multiplicity) for block in blocks]}",
            flush=True,
        )
        results[copies] = optimize(
            blocks,
            full_weight,
            science_dimension,
            restarts=restarts,
            steps=steps,
            seed=args.seed + copies * 1000,
        )

    covariances = {
        "single": point.single_covariance,
        "ns2": results[2][1].science_covariance,
        "ns3": results[3][1].science_covariance,
        "qlan": point.qlan_covariance,
    }
    rows = coordinate_rows(point, covariances)
    qfi_risks = {
        key: float(np.trace(point.science_qfi @ covariance))
        for key, covariance in covariances.items()
    }
    factor = min(1.0, qfi_risks["qlan"] / qfi_risks["ns3"])
    payload = {
        "definition": (
            "One-working-point universal-saturation proxy. Finite PVMs are locally "
            "A-optimal in QFI-whitened full-amplitude plus closure-phase space and are "
            "not strict score-preserving lifts of the one-copy POVM."
        ),
        "working_point": point.metadata,
        "local_coordinate": "h=sqrt(n_s)*(theta-theta0)",
        "parameter_space": {
            "amplitude": "complete E=15 edge space",
            "phase": "C=10 closure space",
            "nuisance": "N-1=5 station-piston directions",
            "branch_budget": "50:50 amplitude/phase in the source one-copy POVM",
        },
        "qfi_weighted_risks": qfi_risks,
        "global_fisher_factor_ns3_over_qlan": factor,
        "global_covariance_inflation_qlan_to_ns3": 1.0 / factor,
        "factor_rule": (
            "alpha=min(1, Tr[JQ V_QLAN]/Tr[JQ V_ns3]); all Fig.2 bins use "
            "F_ns3=alpha F_QLAN, equivalently V_ns3=V_QLAN/alpha"
        ),
        "ns2_optimizer": results[2][2],
        "ns3_optimizer": results[3][2],
        "coordinate_rows": rows,
    }
    output_json = HERE / "finite_ns_calibration_650nm.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    for copies in (2, 3):
        unitaries, evaluation, _ = results[copies]
        np.savez_compressed(
            HERE / f"finite_ns{copies}_pvm_650nm.npz",
            fisher=evaluation.fisher,
            covariance=evaluation.covariance,
            science_covariance=evaluation.science_covariance,
            **{
                f"unitary_{index}": unitary
                for index, unitary in enumerate(unitaries)
            },
        )
    print(json.dumps({
        "working_point": point.metadata,
        "qfi_weighted_risks": qfi_risks,
        "global_fisher_factor_ns3_over_qlan": factor,
    }, indent=2))
    print(output_json)


if __name__ == "__main__":
    main()
