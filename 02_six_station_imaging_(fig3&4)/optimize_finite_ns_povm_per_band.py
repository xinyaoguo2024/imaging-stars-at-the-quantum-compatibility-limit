#!/usr/bin/env python3
"""Overcomplete finite-copy POVM optimization at the Fig. 2 650-nm point.

The earlier calibration restricted every copy-permutation Schur block to a
rank-one PVM.  Here a d-dimensional block is measured by an m=q*d outcome
rank-one POVM

    E_y = |a_y><a_y|,        sum_y E_y = I_d,

represented by a d x m row-isometry A=[a_1 ... a_m], A A^dagger=I_d.
The q-fold split of any PVM is an exactly equivalent feasible POVM, so this
search can never improve merely by changing normalizations or benchmarks.

The local A-optimal objective, working point, nuisance treatment, and QFI
weight are imported unchanged from calibrate_finite_ns_at_650nm.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from calibrate_finite_ns_per_band import (
    HERE,
    Evaluation as PvmEvaluation,
    WorkingPoint,
    evaluate as evaluate_pvm,
    local_copy_tangents,
    make_blocks,
    make_working_point,
    sym,
    tensor_power,
)


def row_retraction(matrix: np.ndarray) -> np.ndarray:
    """QR retraction onto A A^dagger=I for a wide d x m matrix."""
    q, r = np.linalg.qr(np.asarray(matrix).conj().T, mode="reduced")
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
    q = q @ np.diag(phases.conj())
    return q.conj().T


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix) + np.asarray(matrix).conj().T)


def split_pvm(unitary: np.ndarray, redundancy: int) -> np.ndarray:
    """Repeat every PVM outcome q times without changing its statistics."""
    if redundancy < 1:
        raise ValueError("redundancy must be positive")
    return np.concatenate([unitary / np.sqrt(redundancy)] * redundancy, axis=1)


@dataclass
class PovmEvaluation:
    objective: float
    fisher: np.ndarray
    covariance: np.ndarray
    science_covariance: np.ndarray
    gradients: list[np.ndarray]
    probabilities: list[np.ndarray]


def evaluate_povm(
    blocks,
    frames: list[np.ndarray],
    full_weight: np.ndarray,
    science_dimension: int,
) -> PovmEvaluation:
    fisher = np.zeros_like(full_weight)
    probabilities: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for block, frame in zip(blocks, frames, strict=True):
        p0 = np.einsum("iy,ij,jy->y", frame.conj(), block.sigma, frame).real
        if p0.min() <= 1.0e-15:
            raise np.linalg.LinAlgError("encountered an effectively zero outcome")
        d0 = np.asarray(
            [
                np.einsum("iy,ij,jy->y", frame.conj(), dot, frame).real
                for dot in block.tangents
            ]
        ).T
        fisher += block.multiplicity * np.einsum(
            "ye,yf,y->ef", d0, d0, 1.0 / p0, optimize=True
        )
        probabilities.append(p0)
        derivatives.append(d0)

    fisher = sym(fisher)
    covariance = np.linalg.inv(fisher)
    science_covariance = sym(covariance[:science_dimension, :science_dimension])
    objective = float(np.trace(full_weight @ covariance))
    sensitivity = covariance @ full_weight @ covariance

    gradients: list[np.ndarray] = []
    for block, frame, p0, d0 in zip(
        blocks, frames, probabilities, derivatives, strict=True
    ):
        h = d0 @ sensitivity.T
        quadratic = np.einsum("ye,ye->y", d0, h)
        tangent_action = np.einsum(
            "ye,eij,jy->iy", h, block.tangents, frame, optimize=True
        )
        euclidean = block.multiplicity * (
            -2.0 * tangent_action / p0[None, :]
            + (block.sigma @ frame) * quadratic[None, :] / p0[None, :] ** 2
        )
        # Orthogonal projection onto Z A^dagger + A Z^dagger = 0.
        normal = hermitian(euclidean @ frame.conj().T)
        gradients.append(euclidean - normal @ frame)
    return PovmEvaluation(
        objective,
        fisher,
        covariance,
        science_covariance,
        gradients,
        probabilities,
    )


def completeness_error(frames: list[np.ndarray]) -> float:
    return float(
        max(
            np.linalg.norm(frame @ frame.conj().T - np.eye(frame.shape[0]), ord="fro")
            for frame in frames
        )
    )


def optimize_from_pvm(
    blocks,
    pvm_unitaries: list[np.ndarray],
    full_weight: np.ndarray,
    science_dimension: int,
    *,
    redundancy: int,
    restarts: int,
    steps: int,
    seed: int,
    perturbation: float,
) -> tuple[list[np.ndarray], PovmEvaluation, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    split = [split_pvm(unitary, redundancy) for unitary in pvm_unitaries]
    exact_split = evaluate_povm(blocks, split, full_weight, science_dimension)
    best_frames = [frame.copy() for frame in split]
    best_evaluation = exact_split
    records: list[dict[str, float | int]] = [
        {
            "restart": -1,
            "objective": exact_split.objective,
            "gradient_norm": float(
                np.sqrt(sum(np.linalg.norm(g) ** 2 for g in exact_split.gradients))
            ),
            "steps": 0,
            "initialization": "exact_split_pvm",
        }
    ]
    print(
        f"[POVM] exact split-PVM risk={exact_split.objective:.10g} "
        f"complete={completeness_error(split):.3e}",
        flush=True,
    )

    for restart in range(restarts):
        # The exact split is stationary within the repeated-outcome submanifold.
        # A small generic perturbation exposes directions available only to an
        # overcomplete frame.  Include one broader random-frame restart too.
        if restart == restarts - 1 and restarts > 1:
            frames = [
                row_retraction(
                    rng.normal(size=frame.shape) + 1j * rng.normal(size=frame.shape)
                )
                for frame in split
            ]
            initialization = "random_tight_frame"
        else:
            scale = perturbation * (1.0 + restart / max(1, restarts - 1))
            frames = [
                row_retraction(
                    frame
                    + scale
                    * (
                        rng.normal(size=frame.shape)
                        + 1j * rng.normal(size=frame.shape)
                    )
                    / np.sqrt(frame.shape[1])
                )
                for frame in split
            ]
            initialization = f"perturbed_split_{scale:.5g}"

        current = evaluate_povm(blocks, frames, full_weight, science_dimension)
        rate = 0.025
        iterations = 0
        for iteration in range(steps):
            gradient_norm = float(
                np.sqrt(sum(np.linalg.norm(g) ** 2 for g in current.gradients))
            )
            if gradient_norm < 1.0e-9:
                break
            trial_rate = rate
            accepted = False
            for _ in range(28):
                trial_frames = [
                    row_retraction(frame - trial_rate * gradient)
                    for frame, gradient in zip(
                        frames, current.gradients, strict=True
                    )
                ]
                try:
                    trial = evaluate_povm(
                        blocks, trial_frames, full_weight, science_dimension
                    )
                except np.linalg.LinAlgError:
                    trial_rate *= 0.5
                    continue
                if trial.objective < (
                    current.objective
                    - 1.0e-4 * trial_rate * gradient_norm**2
                ):
                    frames = trial_frames
                    current = trial
                    rate = min(1.3 * trial_rate, 0.15)
                    accepted = True
                    break
                trial_rate *= 0.5
            iterations = iteration + 1
            if not accepted:
                break
            if iterations % 250 == 0:
                print(
                    f"[POVM] restart={restart + 1}/{restarts} step={iterations} "
                    f"risk={current.objective:.9g} grad={gradient_norm:.3e}",
                    flush=True,
                )

        gradient_norm = float(
            np.sqrt(sum(np.linalg.norm(g) ** 2 for g in current.gradients))
        )
        record: dict[str, float | int | str] = {
            "restart": restart,
            "objective": current.objective,
            "gradient_norm": gradient_norm,
            "steps": iterations,
            "initialization": initialization,
        }
        records.append(record)
        print(
            f"[POVM] restart={restart + 1}/{restarts} risk={current.objective:.10g} "
            f"grad={gradient_norm:.3e} steps={iterations} "
            f"complete={completeness_error(frames):.3e}",
            flush=True,
        )
        if current.objective < best_evaluation.objective:
            best_frames = [frame.copy() for frame in frames]
            best_evaluation = current

    return best_frames, best_evaluation, records


def finite_difference_check(
    blocks,
    frames: list[np.ndarray],
    full_weight: np.ndarray,
    science_dimension: int,
    seed: int,
) -> dict[str, float]:
    """Check the projected analytic gradient along one tangent direction."""
    rng = np.random.default_rng(seed)
    directions: list[np.ndarray] = []
    for frame in frames:
        raw = rng.normal(size=frame.shape) + 1j * rng.normal(size=frame.shape)
        directions.append(raw - hermitian(raw @ frame.conj().T) @ frame)
    base = evaluate_povm(blocks, frames, full_weight, science_dimension)
    analytic = float(
        2.0
        * sum(
            np.vdot(gradient, direction).real
            for gradient, direction in zip(base.gradients, directions, strict=True)
        )
    )
    epsilon = 1.0e-6
    plus = evaluate_povm(
        blocks,
        [
            row_retraction(frame + epsilon * direction)
            for frame, direction in zip(frames, directions, strict=True)
        ],
        full_weight,
        science_dimension,
    ).objective
    minus = evaluate_povm(
        blocks,
        [
            row_retraction(frame - epsilon * direction)
            for frame, direction in zip(frames, directions, strict=True)
        ],
        full_weight,
        science_dimension,
    ).objective
    numerical = float((plus - minus) / (2.0 * epsilon))
    return {
        "analytic_directional_derivative": analytic,
        "numerical_directional_derivative": numerical,
        "relative_error": float(
            abs(analytic - numerical) / max(1.0, abs(analytic), abs(numerical))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, choices=(2, 3), default=3)
    parser.add_argument("--band-index", type=int, required=True)
    parser.add_argument("--redundancy", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--perturbation", type=float, default=0.035)
    args = parser.parse_args()

    point: WorkingPoint = make_working_point(args.band_index)
    science_dimension = point.science_qfi.shape[0]
    full_dimension = len(point.full_tangents)
    full_weight = np.zeros((full_dimension, full_dimension))
    full_weight[:science_dimension, :science_dimension] = point.science_qfi

    sigma = tensor_power(point.rho, args.copies)
    tangents = local_copy_tangents(point.rho, point.full_tangents, args.copies)
    blocks = make_blocks(6, args.copies, sigma, tangents)
    pvm_path = HERE / f"finite_ns{args.copies}_pvm_650nm_warmstart.npz"
    saved = np.load(pvm_path)
    pvm_unitaries = [saved[f"unitary_{i}"] for i in range(len(blocks))]
    pvm_evaluation: PvmEvaluation = evaluate_pvm(
        blocks,
        pvm_unitaries,
        full_weight,
        science_dimension,
    )
    print(
        f"[setup] ns={args.copies} dimensions={[b.dimension for b in blocks]} "
        f"multiplicities={[b.multiplicity for b in blocks]} "
        f"outcomes={[args.redundancy * b.dimension for b in blocks]}",
        flush=True,
    )
    check_frames = [
        row_retraction(
            split_pvm(unitary, args.redundancy)
            + 0.01
            * (
                np.random.default_rng(args.seed + i).normal(
                    size=(unitary.shape[0], args.redundancy * unitary.shape[1])
                )
                + 1j
                * np.random.default_rng(args.seed + 100 + i).normal(
                    size=(unitary.shape[0], args.redundancy * unitary.shape[1])
                )
            )
            / np.sqrt(args.redundancy * unitary.shape[1])
        )
        for i, unitary in enumerate(pvm_unitaries)
    ]
    derivative_check = finite_difference_check(
        blocks, check_frames, full_weight, science_dimension, args.seed + 999
    )
    print(f"[gradient-check] {json.dumps(derivative_check)}", flush=True)
    if derivative_check["relative_error"] > 5.0e-5:
        raise RuntimeError("POVM gradient failed finite-difference validation")

    frames, evaluation, records = optimize_from_pvm(
        blocks,
        pvm_unitaries,
        full_weight,
        science_dimension,
        redundancy=args.redundancy,
        restarts=args.restarts,
        steps=args.steps,
        seed=args.seed + 1000 * args.copies,
        perturbation=args.perturbation,
    )

    qlan_risk = float(np.trace(point.science_qfi @ point.qlan_covariance))
    single_risk = float(np.trace(point.science_qfi @ point.single_covariance))
    payload = {
        "definition": (
            "Locally A-optimal overcomplete rank-one POVM search, with every "
            "d-dimensional Schur block represented by a d x (q d) row-isometry."
        ),
        "working_point": point.metadata,
        "copies": args.copies,
        "redundancy": args.redundancy,
        "block_dimensions": [block.dimension for block in blocks],
        "block_multiplicities": [block.multiplicity for block in blocks],
        "block_outcomes": [frame.shape[1] for frame in frames],
        "pvm_risk": pvm_evaluation.objective,
        "povm_risk": evaluation.objective,
        "singlecopy_risk": single_risk,
        "qlan_risk": qlan_risk,
        "povm_over_pvm_risk": evaluation.objective / pvm_evaluation.objective,
        "fisher_gain_povm_over_pvm": pvm_evaluation.objective / evaluation.objective,
        "aggregate_fisher_gain_povm_over_single": single_risk / evaluation.objective,
        "aggregate_snr_gain_povm_over_single": np.sqrt(single_risk / evaluation.objective),
        "global_fisher_factor_povm_over_qlan": qlan_risk / evaluation.objective,
        "completeness_error": completeness_error(frames),
        "probability_sum": float(
            sum(
                block.multiplicity * np.sum(probability)
                for block, probability in zip(
                    blocks, evaluation.probabilities, strict=True
                )
            )
        ),
        "gradient_check": derivative_check,
        "optimizer_records": records,
        "caveat": (
            "This is a local nonconvex rank-one POVM optimum in a fixed "
            "copy-permutation block-diagonal class, not a global finite-copy "
            "optimality certificate and not yet a strict score-preserving lift."
        ),
    }
    wavelength_tag = f"{float(point.metadata['lambda_center_nm']):.3f}".replace(".", "p")
    stem = (
        f"finite_ns{args.copies}_povm_q{args.redundancy}_"
        f"band{args.band_index:02d}_{wavelength_tag}nm"
    )
    output_json = HERE / f"{stem}.json"
    output_npz = HERE / f"{stem}.npz"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        output_npz,
        fisher=evaluation.fisher,
        covariance=evaluation.covariance,
        science_covariance=evaluation.science_covariance,
        **{f"frame_{i}": frame for i, frame in enumerate(frames)},
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(output_json)
    print(output_npz)


if __name__ == "__main__":
    main()
