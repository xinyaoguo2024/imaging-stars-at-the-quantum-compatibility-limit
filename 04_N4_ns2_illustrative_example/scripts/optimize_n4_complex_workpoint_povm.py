#!/usr/bin/env python3
r"""Optimize q=2 Schur POVMs at the fixed complex N=4 working point.

The state, parameter ordering, and repetitive benchmark are identical to
``optimize_n4_complex_workpoint_pvm.py``.  In every Schur block the PVM is
enlarged to a two-times-overcomplete rank-one Parseval frame,

    M_{lambda,y} = |a_{lambda,y}><a_{lambda,y}| \otimes I_{m_lambda},
    A_lambda A_lambda^dagger = I.

The A-optimal objective is Tr[J^{-1}] for the six edge phases, in the local
coordinates h=sqrt(n_s)(phi-phi_0).  Since the frame optimization is
nonconvex, the result is a validated best-of-restarts local optimum rather
than a certificate for the unrestricted optimal POVM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import optimize_n4_complex_workpoint_pvm as complex_pvm
import optimize_n4_overcomplete_povm as frame_core
import optimize_phase_pvm_scan as schur_core


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = Path(os.environ.get("N4_DATA_DIR", str(PACKAGE / "data")))


def build_problem(copies: int):
    rho = complex_pvm.complex_working_point()
    edges, tangents = schur_core.phase_tangents(rho)
    if edges != complex_pvm.EDGES:
        raise RuntimeError("unexpected edge ordering")
    sigma = schur_core.tensor_power(rho, copies)
    dots = schur_core.local_copy_tangents(rho, tangents, copies)
    blocks = schur_core.make_blocks(complex_pvm.N, copies, sigma, dots)
    return rho, tangents, sigma, dots, blocks


def saved_pvm_frames(copies: int, blocks, redundancy: int) -> list[np.ndarray]:
    saved = np.load(DATA / f"n4_ns{copies}_phase_pvm_complexwp.npz")
    return [
        frame_core.split_pvm(saved[f"unitary_{block.label}"], redundancy)
        for block in blocks
    ]


def optimize_case(
    copies: int,
    *,
    redundancy: int,
    restarts: int,
    steps: int,
    refine_steps: int,
    seed: int,
    perturbation: float,
):
    rho, tangents, sigma, dots, blocks = build_problem(copies)
    split = saved_pvm_frames(copies, blocks, redundancy)
    best_frames = [frame.copy() for frame in split]
    best = frame_core.evaluate(blocks, best_frames)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = [
        {
            "restart": -1,
            "initialization": "exact_split_complex_workpoint_pvm",
            "risk": best.risk,
            "gradient_norm": frame_core.gradient_norm(best),
            "steps": 0,
        }
    ]
    print(
        f"n_s={copies}: split-PVM risk={best.risk:.10f}, "
        f"grad={frame_core.gradient_norm(best):.3e}",
        flush=True,
    )

    for restart in range(restarts):
        # Most starts probe the neighbourhood of the optimized PVM.  The last
        # start is an unrelated random tight frame, which checks that the
        # result is not solely an artefact of the PVM basin.
        if restart == restarts - 1:
            initial = [
                frame_core.row_retraction(
                    rng.normal(size=frame.shape)
                    + 1j * rng.normal(size=frame.shape)
                )
                for frame in split
            ]
            initialization = "random_tight_frame"
        else:
            scale = perturbation * (1 + restart / max(1, restarts - 1))
            initial = [
                frame_core.row_retraction(
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
            initialization = f"perturbed_split_{scale:.6g}"
        frames, evaluation, steps_taken = frame_core.descend(
            blocks, initial, steps=steps
        )
        record = {
            "restart": restart,
            "initialization": initialization,
            "risk": evaluation.risk,
            "gradient_norm": frame_core.gradient_norm(evaluation),
            "steps": steps_taken,
        }
        records.append(record)
        print(
            f"n_s={copies} restart={restart + 1}/{restarts}: "
            f"risk={evaluation.risk:.10f}, "
            f"grad={record['gradient_norm']:.3e}, steps={steps_taken}",
            flush=True,
        )
        if evaluation.risk < best.risk:
            best_frames = [frame.copy() for frame in frames]
            best = evaluation

    best_frames, best, steps_taken = frame_core.descend(
        blocks, best_frames, steps=refine_steps
    )
    records.append(
        {
            "restart": "refinement",
            "initialization": "saved_best",
            "risk": best.risk,
            "gradient_norm": frame_core.gradient_norm(best),
            "steps": steps_taken,
        }
    )
    return rho, tangents, sigma, dots, blocks, best_frames, best, records


def refine_saved_case(copies: int, *, steps: int):
    rho, tangents, sigma, dots, blocks = build_problem(copies)
    path = DATA / f"n4_ns{copies}_phase_povm_q2_complexwp.npz"
    saved = np.load(path)
    frames = [saved[f"frame_{block.label}"] for block in blocks]
    initial = frame_core.evaluate(blocks, frames)
    frames, evaluation, steps_taken = frame_core.descend(
        blocks, frames, steps=steps
    )
    records = [
        {
            "restart": "refine_existing",
            "initialization": "saved_complexwp_q2_povm",
            "initial_risk": initial.risk,
            "initial_gradient_norm": frame_core.gradient_norm(initial),
            "risk": evaluation.risk,
            "gradient_norm": frame_core.gradient_norm(evaluation),
            "steps": steps_taken,
        }
    ]
    return rho, tangents, sigma, dots, blocks, frames, evaluation, records


def full_space_reconstruction(
    blocks,
    frames: list[np.ndarray],
    sigma: np.ndarray,
    dots: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Independently reconstruct effects and the CFIM in the full space."""
    dimension = sigma.shape[0]
    effects: list[np.ndarray] = []
    for block, frame in zip(blocks, frames, strict=True):
        for column in range(frame.shape[1]):
            ket = frame[:, column]
            reduced_effect = np.outer(ket, ket.conj())
            effect = np.zeros((dimension, dimension), dtype=complex)
            for basis in block.bases:
                effect += basis @ reduced_effect @ basis.conj().T
            effects.append(schur_core.hermitian(effect))
    probabilities = np.asarray(
        [np.trace(sigma @ effect).real for effect in effects]
    )
    derivatives = np.asarray(
        [[np.trace(dot @ effect).real for dot in dots] for effect in effects]
    )
    fisher = np.einsum(
        "ye,yf,y->ef", derivatives, derivatives, 1 / probabilities, optimize=True
    )
    completeness = np.linalg.norm(
        sum(effects, start=np.zeros_like(sigma)) - np.eye(dimension), ord="fro"
    )
    return probabilities, derivatives, schur_core.hermitian(fisher).real, float(completeness)


def validate_and_summarize(
    copies: int,
    rho: np.ndarray,
    tangents: list[np.ndarray],
    sigma: np.ndarray,
    dots: list[np.ndarray],
    blocks,
    frames: list[np.ndarray],
    evaluation: frame_core.Evaluation,
    records: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    probabilities, derivatives, independent_fisher, full_completeness = (
        full_space_reconstruction(blocks, frames, sigma, dots)
    )
    fisher = np.asarray(evaluation.fisher, dtype=float)
    covariance = np.linalg.inv(fisher)
    edge_gains = np.diag(np.linalg.inv(complex_pvm.J_REP)) / np.diag(covariance)
    mode_gains = 24 * np.linalg.eigvalsh(fisher)
    qfi = schur_core.quantum_fisher_matrix(rho, tangents)
    qvalues, qvectors = np.linalg.eigh(qfi)
    qinvhalf = qvectors @ np.diag(qvalues ** -0.5) @ qvectors.T
    qnormalized = schur_core.hermitian(qinvhalf @ fisher @ qinvhalf).real
    qratios = np.linalg.eigvalsh(qnormalized)
    risk = float(np.trace(covariance))
    repetitive_risk = float(np.trace(np.linalg.inv(complex_pvm.J_REP)))

    weighted_probabilities = np.concatenate(
        [
            block.multiplicity * p
            for block, p in zip(
                blocks, evaluation.probabilities, strict=True
            )
        ]
    )
    weighted_derivatives = np.vstack(
        [
            block.multiplicity * d
            for block, d in zip(blocks, evaluation.derivatives, strict=True)
        ]
    )
    checks = {
        "rho_trace": abs(np.trace(rho) - 1),
        "rho_hermiticity": np.linalg.norm(rho - rho.conj().T, ord="fro"),
        "reduced_frame_completeness": frame_core.completeness_error(frames),
        "full_space_completeness": full_completeness,
        "probability_normalization": abs(probabilities.sum() - 1),
        "probability_agreement": np.linalg.norm(
            probabilities - weighted_probabilities
        ),
        "derivative_agreement": np.linalg.norm(
            derivatives - weighted_derivatives, ord="fro"
        ),
        "independent_fisher_agreement": np.linalg.norm(
            independent_fisher - fisher, ord="fro"
        ),
        "qfi_domination": max(0.0, qratios.max() - 1),
    }
    if probabilities.min() <= 0:
        raise RuntimeError("nonpositive outcome probability")
    if max(checks.values()) > 5e-9:
        raise RuntimeError(f"validation failed: {checks}")

    summary: dict[str, object] = {
        "N": complex_pvm.N,
        "n_s": copies,
        "visibility_magnitude": complex_pvm.G_ABS,
        "edge_order_one_indexed": [
            [i + 1, j + 1] for i, j in complex_pvm.EDGES
        ],
        "edge_phases_radians": complex_pvm.PHASES.tolist(),
        "edge_phases_pi_units": (complex_pvm.PHASES / np.pi).tolist(),
        "rho_real": rho.real.tolist(),
        "rho_imag": rho.imag.tolist(),
        "rho_eigenvalues": np.linalg.eigvalsh(rho).tolist(),
        "local_coordinate": "h=sqrt(n_s)*(phi-phi0)",
        "repetitive_benchmark": "J_rep=I_6/24",
        "receiver_class": (
            "q=2 overcomplete rank-one Parseval-frame POVM in each "
            "copy-permutation Schur block"
        ),
        "schur_blocks": [
            {
                "label": block.label,
                "reduced_dimension": block.dimension,
                "multiplicity": block.multiplicity,
                "number_of_effects": frame.shape[1],
            }
            for block, frame in zip(blocks, frames, strict=True)
        ],
        "number_of_reported_outcomes": int(
            sum(frame.shape[1] for frame in frames)
        ),
        "joint_fisher_matrix": fisher.tolist(),
        "joint_fisher_eigenvalues": np.linalg.eigvalsh(fisher).tolist(),
        "joint_covariance_matrix": covariance.tolist(),
        "nuisance_aware_edge_fisher": (1 / np.diag(covariance)).tolist(),
        "nuisance_aware_edge_gains_over_repetitive": edge_gains.tolist(),
        "mean_nuisance_aware_edge_gain": float(edge_gains.mean()),
        "minimum_nuisance_aware_edge_gain": float(edge_gains.min()),
        "maximum_nuisance_aware_edge_gain": float(edge_gains.max()),
        "generalized_mode_gains_over_repetitive": mode_gains.tolist(),
        "minimum_mode_gain": float(mode_gains.min()),
        "arithmetic_mean_mode_gain": float(mode_gains.mean()),
        "maximum_mode_gain": float(mode_gains.max()),
        "repetitive_A_risk": repetitive_risk,
        "joint_A_risk": risk,
        "A_risk_gain": repetitive_risk / risk,
        "A_risk_SNR_gain": float(np.sqrt(repetitive_risk / risk)),
        "qfi_matrix": qfi.tolist(),
        "qfi_eigenvalues": qvalues.tolist(),
        "qfi_A_risk": float(np.trace(np.linalg.inv(qfi))),
        "joint_to_qfi_generalized_eigenvalues": qratios.tolist(),
        "maximum_joint_to_qfi_ratio": float(qratios.max()),
        "minimum_outcome_probability": float(probabilities.min()),
        "maximum_outcome_probability": float(probabilities.max()),
        "probability_sum": float(probabilities.sum()),
        "riemannian_gradient_norm": frame_core.gradient_norm(evaluation),
        "optimizer_records": records,
        "validation": {key: float(value) for key, value in checks.items()},
        "local_nonconvex_optimum_only": True,
        "caveat": (
            "Best validated local optimum in the q=2 Schur-Parseval-frame "
            "class; not a global unrestricted-POVM certificate."
        ),
    }
    arrays = {
        "rho": rho,
        "edges_zero_indexed": np.asarray(complex_pvm.EDGES),
        "edge_phases_radians": complex_pvm.PHASES,
        "J_repetitive": complex_pvm.J_REP,
        "fisher": fisher,
        "covariance": covariance,
        "probabilities": probabilities,
        "derivatives": derivatives,
        "nuisance_aware_edge_gains": edge_gains,
        "mode_gains": mode_gains,
        "qfi": qfi,
        "qfi_normalized_eigenvalues": qratios,
    }
    return summary, arrays


def save_case(
    copies: int,
    blocks,
    frames: list[np.ndarray],
    summary: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    stem = f"n4_ns{copies}_phase_povm_q2_complexwp"
    (DATA / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for block, frame in zip(blocks, frames, strict=True):
        arrays[f"frame_{block.label}"] = frame
        arrays[f"sigma_{block.label}"] = block.sigma
        arrays[f"tangents_{block.label}"] = block.tangents
        for index, basis in enumerate(block.bases):
            arrays[f"schur_basis_{block.label}_{index}"] = basis
    np.savez_compressed(DATA / f"{stem}.npz", **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redundancy", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps-ns2", type=int, default=3000)
    parser.add_argument("--steps-ns3", type=int, default=8000)
    parser.add_argument("--refine-steps", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--perturbation", type=float, default=0.035)
    parser.add_argument(
        "--refine-only",
        choices=("2", "3", "both"),
        help="continue from saved complex-workpoint q=2 POVMs",
    )
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    copies_to_run = (
        [2, 3]
        if args.refine_only in (None, "both")
        else [int(args.refine_only)]
    )
    for copies in copies_to_run:
        if args.refine_only:
            result = refine_saved_case(copies, steps=args.refine_steps)
        else:
            result = optimize_case(
                copies,
                redundancy=args.redundancy,
                restarts=args.restarts,
                steps=args.steps_ns2 if copies == 2 else args.steps_ns3,
                refine_steps=args.refine_steps,
                seed=args.seed + copies,
                perturbation=args.perturbation,
            )
        (
            rho,
            tangents,
            sigma,
            dots,
            blocks,
            frames,
            evaluation,
            records,
        ) = result
        summary, arrays = validate_and_summarize(
            copies,
            rho,
            tangents,
            sigma,
            dots,
            blocks,
            frames,
            evaluation,
            records,
        )
        save_case(copies, blocks, frames, summary, arrays)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
