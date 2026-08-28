#!/usr/bin/env python3
"""Unit-weight Holevo benchmark for the N=4 complex working point.

The six parameters are the oriented edge phases in the order
(12, 13, 14, 23, 24, 34).  This script performs two independent numerical
calculations:

1. a convex semidefinite formulation of the finite-dimensional Holevo bound;
2. a multistart minimization of the equivalent, unsmoothed Holevo functional.

The second calculation makes the reported covariance explicit, while the SDP
provides a global convex cross-check.  All Fisher matrices and covariances use
the local per-copy coordinate h = sqrt(n_s) (phi - phi_0).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "n4_complexwp_holevo_summary.json"
N = 4
P = 6
EDGES = list(itertools.combinations(range(N), 2))
EDGE_LABELS = [f"{i + 1}{j + 1}" for i, j in EDGES]
PHASES = np.array(
    [0.0, np.pi / 12, np.pi / 6, np.pi / 6, np.pi / 12, -np.pi / 6]
)


def hermitian_basis(n: int) -> np.ndarray:
    """Return a trace-orthonormal Hermitian basis, including the diagonals."""
    basis: list[np.ndarray] = []
    for i in range(n):
        element = np.zeros((n, n), dtype=complex)
        element[i, i] = 1.0
        basis.append(element)
    for i, j in itertools.combinations(range(n), 2):
        symmetric = np.zeros((n, n), dtype=complex)
        symmetric[i, j] = symmetric[j, i] = 1 / np.sqrt(2)
        basis.append(symmetric)
        antisymmetric = np.zeros((n, n), dtype=complex)
        antisymmetric[i, j] = -1j / np.sqrt(2)
        antisymmetric[j, i] = 1j / np.sqrt(2)
        basis.append(antisymmetric)
    return np.asarray(basis)


def working_state() -> np.ndarray:
    rho = np.eye(N, dtype=complex) / N
    for (i, j), phase in zip(EDGES, PHASES, strict=True):
        rho[i, j] = 0.5 * np.exp(1j * phase) / N
        rho[j, i] = rho[i, j].conjugate()
    return rho


def phase_tangents(rho: np.ndarray) -> list[np.ndarray]:
    tangents: list[np.ndarray] = []
    for i, j in EDGES:
        dot = np.zeros_like(rho)
        dot[i, j] = 1j * rho[i, j]
        dot[j, i] = -1j * rho[j, i]
        tangents.append(dot)
    return tangents


def sld_data(
    rho: np.ndarray, tangents: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(rho)
    denominator = values[:, None] + values[None, :]
    slds: list[np.ndarray] = []
    for tangent in tangents:
        transformed = vectors.conjugate().T @ tangent @ vectors
        slds.append(vectors @ (2 * transformed / denominator) @ vectors.conjugate().T)
    gram = np.asarray(
        [[np.trace(rho @ left @ right) for right in slds] for left in slds]
    )
    qfi = 0.5 * (gram.real + gram.real.T)
    weak = 0.5 * (gram.imag - gram.imag.T)
    return slds, qfi, weak


def matrix_absolute_i_antisymmetric(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(1j * (0.5 * (matrix - matrix.T)))
    result = (vectors * np.abs(values)) @ vectors.conjugate().T
    return 0.5 * (result.real + result.real.T)


def operator_coordinates(
    rho: np.ndarray, tangents: list[np.ndarray], basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gram = np.asarray(
        [
            [np.trace(rho @ left @ right) for right in basis]
            for left in basis
        ]
    )
    gram = 0.5 * (gram + gram.conjugate().T)
    mean = np.asarray([np.trace(rho @ element).real for element in basis])
    derivatives = np.asarray(
        [
            [np.trace(tangent @ element).real for element in basis]
            for tangent in tangents
        ]
    )
    return gram, mean, derivatives


def solve_holevo_sdp(
    gram: np.ndarray,
    mean: np.ndarray,
    derivatives: np.ndarray,
) -> dict[str, object]:
    """Solve min Tr(V) with V real and V >= Z[X] by a Schur LMI."""
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    factor = np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.conjugate().T
    dimension = gram.shape[0]
    coefficients = cp.Variable((dimension, P))
    covariance = cp.Variable((P, P), symmetric=True)
    factored = factor @ coefficients
    lmi = cp.bmat(
        [
            [covariance, cp.conj(factored).T],
            [factored, np.eye(dimension)],
        ]
    )
    constraints = [
        coefficients.T @ mean == 0,
        coefficients.T @ derivatives.T == np.eye(P),
        lmi >> 0,
    ]
    problem = cp.Problem(cp.Minimize(cp.trace(covariance)), constraints)

    solver_records: list[dict[str, object]] = []
    solutions: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    settings = [
        (
            "CLARABEL",
            dict(
                tol_gap_abs=1e-9,
                tol_gap_rel=1e-9,
                tol_feas=1e-9,
                max_iter=2000,
            ),
        ),
        ("SCS", dict(eps=1e-7, max_iters=250_000)),
    ]
    for solver, options in settings:
        value = problem.solve(solver=solver, verbose=False, **options)
        if coefficients.value is None or covariance.value is None:
            solver_records.append(
                {"solver": solver, "status": problem.status, "objective": None}
            )
            continue
        coeff = np.asarray(coefficients.value)
        cov = np.asarray(covariance.value)
        z_matrix = coeff.T @ gram @ coeff
        solver_records.append(
            {
                "solver": solver,
                "status": problem.status,
                "reported_objective": float(value),
                "trace_covariance": float(np.trace(cov)),
                "unsmoothed_functional_from_X": float(
                    np.trace(z_matrix.real)
                    + np.sum(np.abs(np.linalg.eigvalsh(1j * z_matrix.imag)))
                ),
                "minimum_eigenvalue_V_minus_Z": float(
                    np.linalg.eigvalsh(cov - z_matrix).min()
                ),
                "unbiasedness_residual": float(
                    np.max(np.abs(coeff.T @ derivatives.T - np.eye(P)))
                ),
                "zero_mean_residual": float(np.max(np.abs(coeff.T @ mean))),
            }
        )
        solutions.append((float(value), coeff, cov, solver))
    if not solutions:
        raise RuntimeError("Both Holevo SDP solvers failed")
    best = min(solutions, key=lambda item: item[0])
    return {
        "reported_objective": best[0],
        "coefficients": best[1],
        "covariance": best[2],
        "selected_solver": best[3],
        "solver_records": solver_records,
        "gram_factorization_residual": float(
            np.max(np.abs(factor.conjugate().T @ factor - gram))
        ),
    }


def affine_parameterization(
    mean: np.ndarray,
    derivatives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    constraints = np.vstack([mean, derivatives])
    target = np.zeros((P, P + 1))
    target[:, 1:] = np.eye(P)
    particular = target @ np.linalg.pinv(constraints.T)
    null = null_space(constraints)
    return particular, null


def nonlinear_multistart(
    gram: np.ndarray,
    mean: np.ndarray,
    derivatives: np.ndarray,
    slds: list[np.ndarray],
    qfi: np.ndarray,
    basis: np.ndarray,
    *,
    restarts: int,
    seed: int,
    maxiter: int,
) -> dict[str, object]:
    """Minimize the equivalent Holevo functional with smooth continuation."""
    particular, null = affine_parameterization(mean, derivatives)
    inverse_qfi = np.linalg.inv(qfi)
    sld_dual = []
    for index in range(P):
        operator = sum(inverse_qfi[index, j] * slds[j] for j in range(P))
        sld_dual.append([np.trace(element @ operator).real for element in basis])
    sld_dual = np.asarray(sld_dual)
    initial = (sld_dual - particular) @ null

    real_gram = gram.real
    imaginary_gram = gram.imag

    def value_gradient(flat: np.ndarray, epsilon: float) -> tuple[float, np.ndarray]:
        free = flat.reshape(P, -1)
        coefficients_by_parameter = particular + free @ null.T
        real_z = coefficients_by_parameter @ real_gram @ coefficients_by_parameter.T
        imaginary_z = (
            coefficients_by_parameter @ imaginary_gram @ coefficients_by_parameter.T
        )
        values, vectors = np.linalg.eigh(1j * imaginary_z)
        smoothed = np.sqrt(values**2 + epsilon**2)
        value = float(np.trace(real_z) + np.sum(smoothed))
        sign = (vectors * (values / smoothed)) @ vectors.conjugate().T
        derivative_imaginary = np.real((1j * sign).T)
        gradient_coefficients = (
            2 * coefficients_by_parameter @ real_gram
            + derivative_imaginary
            @ coefficients_by_parameter
            @ imaginary_gram.T
            + derivative_imaginary.T
            @ coefficients_by_parameter
            @ imaginary_gram
        )
        return value, (gradient_coefficients @ null).ravel()

    rng = np.random.default_rng(seed)
    starts = [initial.ravel()]
    scales = [0.05, 0.2, 1.0]
    for restart in range(max(0, restarts - 1)):
        starts.append(
            initial.ravel()
            + scales[restart % len(scales)] * rng.normal(size=initial.size)
        )

    records: list[dict[str, object]] = []
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    schedule = [1e-3, 1e-5, 1e-7]
    for restart, start in enumerate(starts):
        point = start
        stage_records: list[dict[str, object]] = []
        for epsilon in schedule:
            solution = minimize(
                lambda vector: value_gradient(vector, epsilon),
                point,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": maxiter,
                    "ftol": 1e-14,
                    "gtol": 1e-10,
                    "maxls": 80,
                    "maxcor": 30,
                },
            )
            point = solution.x
            stage_records.append(
                {
                    "epsilon": epsilon,
                    "smoothed_objective": float(solution.fun),
                    "iterations": int(solution.nit),
                    "gradient_norm": float(np.linalg.norm(solution.jac)),
                    "success": bool(solution.success),
                    "message": str(solution.message),
                }
            )
        free = point.reshape(P, -1)
        coefficients_by_parameter = particular + free @ null.T
        z_matrix = coefficients_by_parameter @ gram @ coefficients_by_parameter.T
        covariance = z_matrix.real + matrix_absolute_i_antisymmetric(z_matrix.imag)
        risk = float(np.trace(covariance))
        records.append(
            {
                "restart": restart,
                "initialization": "SLD dual" if restart == 0 else "perturbed SLD dual",
                "unsmoothed_holevo_risk": risk,
                "stages": stage_records,
            }
        )
        candidates.append((risk, coefficients_by_parameter, z_matrix, covariance))
    best = min(candidates, key=lambda item: item[0])
    risks = np.asarray([candidate[0] for candidate in candidates])
    return {
        "risk": best[0],
        "coefficients_by_parameter": best[1],
        "z_matrix": best[2],
        "covariance": best[3],
        "records": records,
        "risk_spread": {
            "minimum": float(risks.min()),
            "maximum": float(risks.max()),
            "standard_deviation": float(risks.std()),
        },
        "null_dimension_per_influence_operator": int(null.shape[1]),
    }


def closure_phases() -> dict[str, float]:
    edge_phase = {edge: value for edge, value in zip(EDGES, PHASES, strict=True)}
    result: dict[str, float] = {}
    for i, j, k in itertools.combinations(range(N), 3):
        value = edge_phase[i, j] + edge_phase[j, k] - edge_phase[i, k]
        result[f"{i + 1}{j + 1}{k + 1}"] = float(value)
    return result


def tolist_complex(matrix: np.ndarray) -> dict[str, object]:
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restarts", type=int, default=13)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--maxiter", type=int, default=6000)
    args = parser.parse_args()

    rho = working_state()
    tangents = phase_tangents(rho)
    basis = hermitian_basis(N)
    slds, qfi, weak = sld_data(rho, tangents)
    gram, mean, derivatives = operator_coordinates(rho, tangents, basis)
    sdp = solve_holevo_sdp(gram, mean, derivatives)
    nonlinear = nonlinear_multistart(
        gram,
        mean,
        derivatives,
        slds,
        qfi,
        basis,
        restarts=args.restarts,
        seed=args.seed,
        maxiter=args.maxiter,
    )

    covariance = nonlinear["covariance"]
    fisher = np.linalg.inv(covariance)
    edge_gains = 24 / np.diag(covariance)
    qfi_covariance = np.linalg.inv(qfi)
    z_matrix = nonlinear["z_matrix"]
    coefficients = nonlinear["coefficients_by_parameter"]
    holevo_risk = float(nonlinear["risk"])
    summary = {
        "model": "N=4, six edge phases, p_i=1/4, |g_ij|=0.5",
        "edge_order": EDGE_LABELS,
        "edge_phases_radians": PHASES.tolist(),
        "edge_phases_over_pi": (PHASES / np.pi).tolist(),
        "closure_phase_convention": "Phi_ijk = phi_ij + phi_jk - phi_ik",
        "closure_phases_radians": closure_phases(),
        "closure_phases_over_pi": {
            key: value / np.pi for key, value in closure_phases().items()
        },
        "rho": tolist_complex(rho),
        "rho_eigenvalues": np.linalg.eigvalsh(rho).tolist(),
        "qfi": qfi.tolist(),
        "qfi_eigenvalues": np.linalg.eigvalsh(qfi).tolist(),
        "weak_commutator": weak.tolist(),
        "weak_commutator_rank_tolerance_1e-10": int(
            np.linalg.matrix_rank(weak, tol=1e-10)
        ),
        "i_weak_commutator_eigenvalues": np.linalg.eigvalsh(1j * weak).tolist(),
        "repetitive_uniform_edge_first": {
            "cfim": (np.eye(P) / 24).tolist(),
            "A_risk": 144.0,
        },
        "formal_sld_qfi_benchmark": {
            "covariance": qfi_covariance.tolist(),
            "A_risk": float(np.trace(qfi_covariance)),
            "A_risk_gain_over_repetitive": float(144 / np.trace(qfi_covariance)),
            "edge_directional_fisher_gains": (
                24 / np.diag(qfi_covariance)
            ).tolist(),
        },
        "unit_weight_holevo_optimum": {
            "weight": "I_6",
            "unsmoothed_holevo_risk": holevo_risk,
            "A_risk_gain_over_repetitive": float(144 / holevo_risk),
            "SNR_gain_over_repetitive": float(np.sqrt(144 / holevo_risk)),
            "covariance": covariance.tolist(),
            "covariance_diagonal": np.diag(covariance).tolist(),
            "fisher": fisher.tolist(),
            "fisher_eigenvalues": np.linalg.eigvalsh(fisher).tolist(),
            "edge_directional_fisher_gains": edge_gains.tolist(),
            "mean_edge_directional_fisher_gain": float(np.mean(edge_gains)),
            "optimal_Z": tolist_complex(z_matrix),
            "trace_real_Z": float(np.trace(z_matrix.real)),
            "trace_norm_i_imag_Z": float(
                np.sum(np.abs(np.linalg.eigvalsh(1j * z_matrix.imag)))
            ),
            "unbiasedness_residual": float(
                np.max(np.abs(coefficients @ derivatives.T - np.eye(P)))
            ),
            "zero_mean_residual": float(np.max(np.abs(coefficients @ mean))),
        },
        "convex_sdp_crosscheck": {
            "selected_solver": sdp["selected_solver"],
            "reported_objective": sdp["reported_objective"],
            "absolute_difference_from_multistart": float(
                abs(float(sdp["reported_objective"]) - holevo_risk)
            ),
            "gram_factorization_residual": sdp["gram_factorization_residual"],
            "solver_records": sdp["solver_records"],
        },
        "multistart_convergence": {
            "number_of_restarts": args.restarts,
            "seed": args.seed,
            "risk_spread": nonlinear["risk_spread"],
            "null_dimension_per_influence_operator": nonlinear[
                "null_dimension_per_influence_operator"
            ],
            "records": nonlinear["records"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"saved: {args.output}")
    print(f"rho minimum eigenvalue: {min(summary['rho_eigenvalues']):.10f}")
    print(f"weak-commutator rank: {summary['weak_commutator_rank_tolerance_1e-10']}")
    print(f"unit-weight Holevo risk: {holevo_risk:.10f}")
    print(f"A-risk gain: {144 / holevo_risk:.10f}")
    print("edge gains:", np.array2string(edge_gains, precision=9))
    print(
        "multistart risk range:",
        f"[{nonlinear['risk_spread']['minimum']:.10f}, "
        f"{nonlinear['risk_spread']['maximum']:.10f}]",
    )
    print(
        "SDP/multistart objective difference:",
        f"{abs(float(sdp['reported_objective']) - holevo_risk):.3e}",
    )


if __name__ == "__main__":
    main()
