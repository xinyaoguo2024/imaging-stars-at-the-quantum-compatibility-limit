#!/usr/bin/env python3
"""Optimize N=4 phase PVMs at a complex, full-rank working point.

The six edge visibilities have |g_ij|=1/2 and phases, in the fixed order
(12,13,14,23,24,34),

    (0, pi/12, pi/6, pi/6, pi/12, -pi/6).

The local parameters are h=sqrt(n_s)(phi-phi_0).  Results are normalized to
the repetitive uniform-edge-first benchmark J_rep=I_6/24.  The optimization
uses the permutation-invariant Schur-block PVM parameterization implemented
in optimize_phase_pvm_scan.py.  Its A-optimal objective is nonconvex, so the
saved receiver is a best-of-restarts local optimum rather than a certificate
of global optimality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import optimize_phase_pvm_scan as core


N = 4
G_ABS = 0.5
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
PHASES = np.array(
    [0.0, np.pi / 12, np.pi / 6, np.pi / 6, np.pi / 12, -np.pi / 6]
)
J_REP = np.eye(len(EDGES)) / 24


def complex_working_point() -> np.ndarray:
    rho = np.eye(N, dtype=complex) / N
    for (i, j), phase in zip(EDGES, PHASES, strict=True):
        rho[i, j] = (G_ABS / N) * np.exp(1j * phase)
        rho[j, i] = rho[i, j].conj()
    values = np.linalg.eigvalsh(rho)
    if values.min() <= 1e-12:
        raise ValueError(f"working point is not full rank: eig={values}")
    return rho


def validate_and_summarize(
    copies: int,
    rho: np.ndarray,
    tangents: list[np.ndarray],
    blocks: list[core.ReducedBlock],
    unitaries: list[np.ndarray],
    evaluation: core.Evaluation,
    records: list[core.RestartRecord],
) -> dict[str, object]:
    probabilities = np.concatenate(evaluation.probabilities)
    derivatives = np.vstack(evaluation.derivatives)
    reconstructed = np.einsum(
        "ye,yf,y->ef", derivatives, derivatives, 1 / probabilities
    )

    completeness = 0.0
    orthogonality = 0.0
    for block, unitary in zip(blocks, unitaries, strict=True):
        completeness = max(
            completeness,
            float(
                np.linalg.norm(
                    unitary @ unitary.conj().T - np.eye(block.dimension), ord="fro"
                )
            ),
        )
        orthogonality = max(
            orthogonality,
            float(
                np.linalg.norm(
                    unitary.conj().T @ unitary - np.eye(block.dimension), ord="fro"
                )
            ),
        )

    fisher = np.asarray(evaluation.fisher, dtype=float)
    covariance = np.linalg.inv(fisher)
    edge_gains = np.diag(np.linalg.inv(J_REP)) / np.diag(covariance)
    mode_gains = np.linalg.eigvalsh(
        np.linalg.solve(np.linalg.cholesky(J_REP), fisher)
        @ np.linalg.inv(np.linalg.cholesky(J_REP).T)
    )
    # J_rep is I/24, but retain the explicit expression above as a check.
    direct_mode_gains = 24 * np.linalg.eigvalsh(fisher)
    if np.linalg.norm(mode_gains - direct_mode_gains) > 1e-10:
        raise RuntimeError("inconsistent generalized Fisher eigenvalues")

    qfi = core.quantum_fisher_matrix(rho, tangents)
    q_values, q_vectors = np.linalg.eigh(qfi)
    q_inv_sqrt = q_vectors @ np.diag(q_values ** -0.5) @ q_vectors.T
    q_normalized = core.hermitian(q_inv_sqrt @ fisher @ q_inv_sqrt).real
    q_ratios = np.linalg.eigvalsh(q_normalized)

    gradient_norm = float(
        np.sqrt(
            sum(np.linalg.norm(gradient, ord="fro") ** 2 for gradient in evaluation.gradients)
        )
    )
    risk = float(np.trace(covariance))
    repetitive_risk = float(np.trace(np.linalg.inv(J_REP)))
    summary: dict[str, object] = {
        "N": N,
        "n_s": copies,
        "visibility_magnitude": G_ABS,
        "edge_order_one_indexed": [[i + 1, j + 1] for i, j in EDGES],
        "edge_phases_radians": PHASES.tolist(),
        "edge_phases_pi_units": (PHASES / np.pi).tolist(),
        "rho_real": rho.real.tolist(),
        "rho_imag": rho.imag.tolist(),
        "rho_eigenvalues": np.linalg.eigvalsh(rho).tolist(),
        "local_coordinate": "h=sqrt(n_s)*(phi-phi0)",
        "repetitive_benchmark": "J_rep=I_6/24",
        "number_of_joint_outcomes": int(sum(block.dimension for block in blocks)),
        "schur_blocks": [
            {
                "label": block.label,
                "reduced_dimension": block.dimension,
                "multiplicity": block.multiplicity,
                "full_effect_rank": block.multiplicity,
            }
            for block in blocks
        ],
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
        "qfi_eigenvalues": q_values.tolist(),
        "qfi_A_risk": float(np.trace(np.linalg.inv(qfi))),
        "joint_to_qfi_generalized_eigenvalues": q_ratios.tolist(),
        "maximum_joint_to_qfi_ratio": float(q_ratios.max()),
        "minimum_outcome_probability": float(probabilities.min()),
        "maximum_outcome_probability": float(probabilities.max()),
        "probability_sum": float(probabilities.sum()),
        "fisher_reconstruction_fro": float(
            np.linalg.norm(reconstructed - fisher, ord="fro")
        ),
        "reduced_completeness_fro": completeness,
        "reduced_projector_orthogonality_fro": orthogonality,
        "riemannian_gradient_norm": gradient_norm,
        "restart_objective_min": float(min(r.objective for r in records)),
        "restart_objective_median": float(
            np.median([r.objective for r in records])
        ),
        "restart_objective_max": float(max(r.objective for r in records)),
        "restart_gradient_median": float(
            np.median([r.gradient_norm for r in records])
        ),
        "local_nonconvex_optimum_only": True,
    }

    # Hard numerical validation before an artifact can be written.
    checks = {
        "rho_trace": abs(np.trace(rho) - 1),
        "rho_hermiticity": np.linalg.norm(rho - rho.conj().T, ord="fro"),
        "probability_normalization": abs(probabilities.sum() - 1),
        "fisher_reconstruction": np.linalg.norm(reconstructed - fisher, ord="fro"),
        "completeness": completeness,
        "orthogonality": orthogonality,
        "qfi_domination": max(0.0, q_ratios.max() - 1),
    }
    summary["validation"] = {key: float(value) for key, value in checks.items()}
    if probabilities.min() <= 0:
        raise RuntimeError("nonpositive outcome probability")
    if max(checks.values()) > 5e-9:
        raise RuntimeError(f"validation failed: {checks}")
    return summary


def save_case(
    output: Path,
    copies: int,
    rho: np.ndarray,
    blocks: list[core.ReducedBlock],
    unitaries: list[np.ndarray],
    evaluation: core.Evaluation,
    records: list[core.RestartRecord],
    summary: dict[str, object],
) -> None:
    stem = f"n4_ns{copies}_phase_pvm_complexwp"
    (output / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    arrays: dict[str, np.ndarray] = {
        "rho": rho,
        "rho_real": rho.real,
        "rho_imag": rho.imag,
        "edges_zero_indexed": np.asarray(EDGES),
        "edge_phases_radians": PHASES,
        "edge_phases_pi_units": PHASES / np.pi,
        "J_repetitive": J_REP,
        "fisher": evaluation.fisher,
        "covariance": evaluation.covariance,
        "probabilities": np.concatenate(evaluation.probabilities),
        "derivatives": np.vstack(evaluation.derivatives),
        "nuisance_aware_edge_gains": np.asarray(
            summary["nuisance_aware_edge_gains_over_repetitive"]
        ),
        "mode_gains": np.asarray(summary["generalized_mode_gains_over_repetitive"]),
        "qfi": np.asarray(summary["qfi_matrix"]),
        "restart_objectives": np.asarray([r.objective for r in records]),
        "restart_gradient_norms": np.asarray([r.gradient_norm for r in records]),
        "restart_steps": np.asarray([r.steps_taken for r in records]),
    }
    offset = 0
    block_slices = []
    for block, unitary in zip(blocks, unitaries, strict=True):
        arrays[f"unitary_{block.label}"] = unitary
        arrays[f"sigma_{block.label}"] = block.sigma
        arrays[f"tangents_{block.label}"] = block.tangents
        for index, basis in enumerate(block.bases):
            arrays[f"schur_basis_{block.label}_{index}"] = basis
        block_slices.append((offset, offset + block.dimension))
        offset += block.dimension
    arrays["outcome_block_slices"] = np.asarray(block_slices)
    np.savez_compressed(output / f"{stem}.npz", **arrays)


def optimize_one(
    copies: int,
    restarts: int,
    steps: int,
    refine_steps: int,
    seed: int,
    output: Path,
) -> dict[str, object]:
    rho = complex_working_point()
    edges, tangents = core.phase_tangents(rho)
    if edges != EDGES:
        raise RuntimeError("unexpected edge ordering")
    sigma = core.tensor_power(rho, copies)
    local_tangents = core.local_copy_tangents(rho, tangents, copies)
    blocks = core.make_blocks(N, copies, sigma, local_tangents)
    print(
        f"\n=== complex workpoint: N=4, n_s={copies}; "
        f"blocks={[(b.label,b.dimension,b.multiplicity) for b in blocks]} ===",
        flush=True,
    )
    unitaries, evaluation, records = core.optimize_blocks(
        blocks, restarts=restarts, steps=steps, seed=seed
    )
    if refine_steps:
        unitaries, evaluation, refinement = core.refine_from_initial(
            blocks, unitaries, refine_steps
        )
        records.append(refinement)
    summary = validate_and_summarize(
        copies, rho, tangents, blocks, unitaries, evaluation, records
    )
    save_case(
        output, copies, rho, blocks, unitaries, evaluation, records, summary
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def refine_saved_case(
    copies: int, refine_steps: int, output: Path
) -> dict[str, object]:
    """Continue descent from a previously saved complex-workpoint receiver."""
    rho = complex_working_point()
    edges, tangents = core.phase_tangents(rho)
    if edges != EDGES:
        raise RuntimeError("unexpected edge ordering")
    sigma = core.tensor_power(rho, copies)
    local_tangents = core.local_copy_tangents(rho, tangents, copies)
    blocks = core.make_blocks(N, copies, sigma, local_tangents)
    path = output / f"n4_ns{copies}_phase_pvm_complexwp.npz"
    saved = np.load(path)
    unitaries = [np.asarray(saved[f"unitary_{block.label}"]) for block in blocks]
    records = [
        core.RestartRecord(
            seed=int(index),
            objective=float(objective),
            gradient_norm=float(gradient),
            steps_taken=int(steps),
        )
        for index, (objective, gradient, steps) in enumerate(
            zip(
                saved["restart_objectives"],
                saved["restart_gradient_norms"],
                saved["restart_steps"],
                strict=True,
            )
        )
    ]
    unitaries, evaluation, refinement = core.refine_from_initial(
        blocks, unitaries, refine_steps
    )
    records.append(refinement)
    summary = validate_and_summarize(
        copies, rho, tangents, blocks, unitaries, evaluation, records
    )
    save_case(
        output, copies, rho, blocks, unitaries, evaluation, records, summary
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns2-restarts", type=int, default=20)
    parser.add_argument("--ns2-steps", type=int, default=2500)
    parser.add_argument("--ns3-restarts", type=int, default=12)
    parser.add_argument("--ns3-steps", type=int, default=5000)
    parser.add_argument("--refine-steps", type=int, default=5000)
    parser.add_argument(
        "--refine-only",
        choices=("2", "3", "both"),
        help="continue from saved complex-workpoint PVMs instead of new restarts",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.refine_only:
        copies_to_run = [2, 3] if args.refine_only == "both" else [int(args.refine_only)]
        summaries = [
            refine_saved_case(copies, args.refine_steps, args.output)
            for copies in copies_to_run
        ]
    else:
        summaries = [
            optimize_one(
                copies=2,
                restarts=args.ns2_restarts,
                steps=args.ns2_steps,
                refine_steps=args.refine_steps,
                seed=args.seed,
                output=args.output,
            ),
            optimize_one(
                copies=3,
                restarts=args.ns3_restarts,
                steps=args.ns3_steps,
                refine_steps=args.refine_steps,
                seed=args.seed + 1000,
                output=args.output,
            ),
        ]
    by_copies = {int(item["n_s"]): item for item in summaries}
    for copies in (2, 3):
        summary_path = args.output / f"n4_ns{copies}_phase_pvm_complexwp_summary.json"
        if copies not in by_copies and summary_path.exists():
            by_copies[copies] = json.loads(summary_path.read_text())
    summaries = [by_copies[copies] for copies in sorted(by_copies)]
    comparison = {
        "working_point": {
            "edge_order_one_indexed": [[i + 1, j + 1] for i, j in EDGES],
            "edge_phases_pi_units": (PHASES / np.pi).tolist(),
            "rho_real": complex_working_point().real.tolist(),
            "rho_imag": complex_working_point().imag.tolist(),
        },
        "receivers": [
            {
                key: item[key]
                for key in (
                    "n_s",
                    "number_of_joint_outcomes",
                    "joint_A_risk",
                    "A_risk_gain",
                    "A_risk_SNR_gain",
                    "nuisance_aware_edge_gains_over_repetitive",
                    "mean_nuisance_aware_edge_gain",
                    "minimum_nuisance_aware_edge_gain",
                    "maximum_nuisance_aware_edge_gain",
                    "generalized_mode_gains_over_repetitive",
                    "riemannian_gradient_norm",
                )
            }
            for item in summaries
        ],
    }
    (args.output / "n4_complexwp_pvm_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
