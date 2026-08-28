#!/usr/bin/env python3
"""Canonical asymptotic SLD-score diagnostic for the N=4 complex workpoint.

The six edge phases are ordered as (12,13,14,23,24,34) and equal

    (0, pi/12, pi/6, pi/6, pi/12, -pi/6).

All station populations are 1/4 and all visibility magnitudes are 1/2, so
rho_ij=(1/8) exp(i phi_ij) for i<j.  The script computes the SLD QFIM J and
the mean-commutator matrix

    D_ab = Tr[rho [L_a,L_b]]/(2i).

It also reports an explicit D-invariance residual for the SLD tangent space.
Because this residual is nonzero at the selected workpoint, the covariance

    V_can = J^{-1} + | i J^{-1} D J^{-1} |

is labelled only as the canonical asymptotic SLD-score receiver.  This script
does not claim that V_can is the exact unrestricted Holevo optimum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N = 4
G_ABS = 0.5
EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
PHASES = np.array(
    [0.0, np.pi / 12, np.pi / 6, np.pi / 6, np.pi / 12, -np.pi / 6]
)
J_REP = np.eye(len(EDGES)) / 24


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2


def working_point() -> np.ndarray:
    """Return rho with rho_ij=sqrt(p_i p_j) g_ij and p_i=1/4."""
    rho = np.eye(N, dtype=complex) / N
    for (i, j), phase in zip(EDGES, PHASES, strict=True):
        rho[i, j] = (G_ABS / N) * np.exp(1j * phase)
        rho[j, i] = rho[i, j].conj()
    return rho


def phase_tangents(rho: np.ndarray) -> list[np.ndarray]:
    """Derivatives with respect to the six physical edge phases."""
    derivatives: list[np.ndarray] = []
    for i, j in EDGES:
        derivative = np.zeros_like(rho)
        derivative[i, j] = 1j * rho[i, j]
        derivative[j, i] = -1j * rho[j, i]
        derivatives.append(derivative)
    return derivatives


def sld_operators(
    rho: np.ndarray, derivatives: list[np.ndarray]
) -> list[np.ndarray]:
    """Solve d rho=(rho L+L rho)/2 in the eigenbasis of full-rank rho."""
    values, vectors = np.linalg.eigh(rho)
    denominator = values[:, None] + values[None, :]
    slds: list[np.ndarray] = []
    for derivative in derivatives:
        derivative_eigenbasis = vectors.conj().T @ derivative @ vectors
        sld_eigenbasis = 2 * derivative_eigenbasis / denominator
        sld = vectors @ sld_eigenbasis @ vectors.conj().T
        slds.append(hermitian(sld))
    return slds


def sld_inner(rho: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Real SLD inner product Re Tr(rho left right)."""
    return float(np.real(np.trace(rho @ left @ right)))


def qfi_and_mean_commutator(
    rho: np.ndarray, slds: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(slds)
    qfi = np.empty((dimension, dimension), dtype=float)
    commutator = np.empty((dimension, dimension), dtype=float)
    for a, left in enumerate(slds):
        for b, right in enumerate(slds):
            product = np.trace(rho @ left @ right)
            qfi[a, b] = product.real
            commutator[a, b] = product.imag
    return hermitian(qfi).real, (commutator - commutator.T) / 2


def commutation_operator(
    rho: np.ndarray, operator: np.ndarray
) -> np.ndarray:
    r"""Return D_rho(operator), defined by

    <Y,D_rho(X)>_SLD = Tr[rho[Y,X]]/(2i).

    In the eigenbasis of rho this is
    [D_rho(X)]_ij=i(lambda_i-lambda_j)X_ij/(lambda_i+lambda_j).
    """
    values, vectors = np.linalg.eigh(rho)
    operator_eigenbasis = vectors.conj().T @ operator @ vectors
    multiplier = 1j * (values[:, None] - values[None, :]) / (
        values[:, None] + values[None, :]
    )
    result = vectors @ (multiplier * operator_eigenbasis) @ vectors.conj().T
    return hermitian(result)


def d_invariance_diagnostic(
    rho: np.ndarray, slds: list[np.ndarray], qfi: np.ndarray, d_matrix: np.ndarray
) -> dict[str, object]:
    """Project D_rho(L_b) onto span_R{L_a} in the SLD metric."""
    inverse_qfi = np.linalg.inv(qfi)
    absolute_residuals: list[float] = []
    relative_residuals: list[float] = []
    projection_coefficients = np.empty_like(qfi)
    metric_consistency = 0.0
    residual_orthogonality = 0.0
    total_residual_squared = 0.0
    total_image_squared = 0.0

    for b, sld in enumerate(slds):
        image = commutation_operator(rho, sld)
        overlaps = np.array([sld_inner(rho, left, image) for left in slds])
        metric_consistency = max(
            metric_consistency, float(np.max(np.abs(overlaps - d_matrix[:, b])))
        )
        coefficients = inverse_qfi @ overlaps
        projection_coefficients[:, b] = coefficients
        projection = sum(
            (coefficient * basis for coefficient, basis in zip(coefficients, slds)),
            np.zeros_like(rho),
        )
        residual = image - projection
        residual_squared = max(sld_inner(rho, residual, residual), 0.0)
        image_squared = max(sld_inner(rho, image, image), 0.0)
        absolute_residuals.append(float(np.sqrt(residual_squared)))
        relative_residuals.append(
            float(np.sqrt(residual_squared / image_squared))
            if image_squared > 1e-30
            else 0.0
        )
        total_residual_squared += residual_squared
        total_image_squared += image_squared
        residual_orthogonality = max(
            residual_orthogonality,
            max(abs(sld_inner(rho, basis, residual)) for basis in slds),
        )

    aggregate_relative = float(
        np.sqrt(total_residual_squared / total_image_squared)
        if total_image_squared > 1e-30
        else 0.0
    )
    return {
        "definition": (
            "For every SLD L_b, project D_rho(L_b) onto the real SLD span "
            "using <X,Y>_SLD=Re Tr(rho X Y); the residual vanishes iff this "
            "SLD tangent space is D-invariant."
        ),
        "projection_coefficients_columns": projection_coefficients.tolist(),
        "absolute_residual_SLD_norm_per_edge": absolute_residuals,
        "relative_residual_per_edge": relative_residuals,
        "aggregate_relative_residual": aggregate_relative,
        "maximum_relative_residual": float(max(relative_residuals)),
        "D_matrix_overlap_consistency_max_abs": metric_consistency,
        "projection_orthogonality_max_abs": float(residual_orthogonality),
        "is_D_invariant_at_tolerance_1e-10": aggregate_relative < 1e-10,
    }


def matrix_absolute_hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return |H| for a numerically Hermitian matrix H."""
    values, vectors = np.linalg.eigh(hermitian(matrix))
    return hermitian(vectors @ np.diag(np.abs(values)) @ vectors.conj().T)


def closure_phases() -> tuple[list[tuple[int, int, int]], np.ndarray]:
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    phase_map = {edge: phase for edge, phase in zip(EDGES, PHASES, strict=True)}

    def oriented_phase(i: int, j: int) -> float:
        return float(phase_map[(i, j)] if i < j else -phase_map[(j, i)])

    closures = np.array(
        [
            oriented_phase(i, j)
            + oriented_phase(j, k)
            - oriented_phase(i, k)
            for i, j, k in triangles
        ]
    )
    return triangles, closures


def to_real_matrix(matrix: np.ndarray, tolerance: float = 1e-11) -> np.ndarray:
    if np.max(np.abs(matrix.imag)) > tolerance:
        raise RuntimeError("expected a numerically real matrix")
    return matrix.real


def main() -> None:
    parser = argparse.ArgumentParser()
    default_output = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "n4_complex_workpoint_asymptotic_score_diagnostic.json"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    rho = working_point()
    derivatives = phase_tangents(rho)
    slds = sld_operators(rho, derivatives)
    qfi, d_matrix = qfi_and_mean_commutator(rho, slds)
    inverse_qfi = np.linalg.inv(qfi)
    incompatibility_term = hermitian(1j * inverse_qfi @ d_matrix @ inverse_qfi)
    absolute_incompatibility = matrix_absolute_hermitian(incompatibility_term)
    v_canonical = to_real_matrix(hermitian(inverse_qfi + absolute_incompatibility))
    fisher_canonical = np.linalg.inv(v_canonical)
    edge_gains = np.diag(np.linalg.inv(J_REP)) / np.diag(v_canonical)
    triangles, closures = closure_phases()
    d_invariance = d_invariance_diagnostic(rho, slds, qfi, d_matrix)

    values = np.linalg.eigvalsh(rho)
    qfi_eigenvalues = np.linalg.eigvalsh(qfi)
    repetitive_risk = float(np.trace(np.linalg.inv(J_REP)))
    canonical_risk = float(np.trace(v_canonical))

    sld_equation_residual = max(
        float(
            np.linalg.norm(
                derivative - (rho @ sld + sld @ rho) / 2, ord="fro"
            )
        )
        for derivative, sld in zip(derivatives, slds, strict=True)
    )
    derivative_qfi = np.array(
        [
            [float(np.real(np.trace(derivative @ sld))) for sld in slds]
            for derivative in derivatives
        ]
    )

    diagnostic: dict[str, object] = {
        "receiver_label": "canonical asymptotic SLD-score receiver",
        "interpretation_caveat": (
            "V_can is the canonical asymptotic covariance constructed from "
            "the SLD score and mean commutator.  At this workpoint the SLD "
            "model is not D-invariant, so this file does not identify V_can "
            "with the exact unrestricted Holevo optimum."
        ),
        "N": N,
        "station_populations": [0.25] * N,
        "visibility_magnitude_all_edges": G_ABS,
        "edge_order_one_indexed": [[i + 1, j + 1] for i, j in EDGES],
        "edge_phases_radians": PHASES.tolist(),
        "edge_phases_pi_units": (PHASES / np.pi).tolist(),
        "density_matrix_convention": (
            "rho_ii=p_i=1/4; rho_ij=sqrt(p_i p_j)|g_ij|exp(+i phi_ij) "
            "for i<j"
        ),
        "rho_real": rho.real.tolist(),
        "rho_imag": rho.imag.tolist(),
        "rho_eigenvalues": values.tolist(),
        "closure_phase_convention": "Phi_ijk=phi_ij+phi_jk-phi_ik",
        "closure_triangles_one_indexed": [
            [i + 1, j + 1, k + 1] for i, j, k in triangles
        ],
        "closure_phases_radians": closures.tolist(),
        "closure_phases_pi_units": (closures / np.pi).tolist(),
        "closure_redundancy_check": (
            "Phi_123-Phi_124+Phi_134-Phi_234=0"
        ),
        "closure_redundancy_residual": float(
            closures[0] - closures[1] + closures[2] - closures[3]
        ),
        "SLD_QFI_J": qfi.tolist(),
        "SLD_QFI_eigenvalues": qfi_eigenvalues.tolist(),
        "SLD_QFI_inverse": inverse_qfi.tolist(),
        "mean_commutator_definition": "D_ab=Tr(rho[L_a,L_b])/(2i)",
        "mean_commutator_D": d_matrix.tolist(),
        "mean_commutator_singular_values": np.linalg.svd(
            d_matrix, compute_uv=False
        ).tolist(),
        "mean_commutator_maximum_absolute_entry": float(
            np.max(np.abs(d_matrix))
        ),
        "D_invariance": d_invariance,
        "canonical_covariance_definition": (
            "V_can=J^{-1}+|i J^{-1} D J^{-1}|, where |H| is the "
            "spectral absolute value of Hermitian H"
        ),
        "canonical_incompatibility_matrix_i_Jinv_D_Jinv_real": (
            incompatibility_term.real.tolist()
        ),
        "canonical_incompatibility_matrix_i_Jinv_D_Jinv_imag": (
            incompatibility_term.imag.tolist()
        ),
        "canonical_absolute_incompatibility_term": to_real_matrix(
            absolute_incompatibility
        ).tolist(),
        "canonical_asymptotic_score_covariance_Vcan": v_canonical.tolist(),
        "canonical_asymptotic_score_Fisher_like_matrix": fisher_canonical.tolist(),
        "canonical_covariance_eigenvalues": np.linalg.eigvalsh(v_canonical).tolist(),
        "canonical_Fisher_like_eigenvalues": np.linalg.eigvalsh(
            fisher_canonical
        ).tolist(),
        "repetitive_benchmark": "J_rep=I_6/24",
        "repetitive_A_risk_Tr_Jrep_inverse": repetitive_risk,
        "canonical_A_risk_Tr_Vcan": canonical_risk,
        "canonical_A_risk_gain_over_repetitive": repetitive_risk
        / canonical_risk,
        "nuisance_aware_edge_Fisher_definition": "F_e=1/(V_can)_{ee}",
        "nuisance_aware_edge_Fisher": (1 / np.diag(v_canonical)).tolist(),
        "nuisance_aware_edge_gains_over_repetitive": edge_gains.tolist(),
        "mean_nuisance_aware_edge_gain": float(edge_gains.mean()),
        "minimum_nuisance_aware_edge_gain": float(edge_gains.min()),
        "maximum_nuisance_aware_edge_gain": float(edge_gains.max()),
        "validation": {
            "rho_trace_error": float(abs(np.trace(rho) - 1)),
            "rho_hermiticity_fro": float(
                np.linalg.norm(rho - rho.conj().T, ord="fro")
            ),
            "rho_minimum_eigenvalue": float(values.min()),
            "maximum_SLD_equation_residual_fro": sld_equation_residual,
            "QFI_derivative_formula_residual_fro": float(
                np.linalg.norm(derivative_qfi - qfi, ord="fro")
            ),
            "QFI_symmetry_fro": float(np.linalg.norm(qfi - qfi.T, ord="fro")),
            "D_antisymmetry_fro": float(
                np.linalg.norm(d_matrix + d_matrix.T, ord="fro")
            ),
            "Vcan_symmetry_fro": float(
                np.linalg.norm(v_canonical - v_canonical.T, ord="fro")
            ),
            "Vcan_minimum_eigenvalue": float(
                np.linalg.eigvalsh(v_canonical).min()
            ),
            "Vcan_inverse_residual_fro": float(
                np.linalg.norm(
                    fisher_canonical @ v_canonical - np.eye(len(EDGES)),
                    ord="fro",
                )
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"rho eig = {values}")
    print(f"QFI eig = {qfi_eigenvalues}")
    print(f"max |D_ab| = {np.max(np.abs(d_matrix)):.10f}")
    print(
        "D-invariance aggregate relative residual = "
        f"{d_invariance['aggregate_relative_residual']:.10f}"
    )
    print(f"Tr(V_can) = {canonical_risk:.10f}")
    print(f"A-risk gain = {repetitive_risk / canonical_risk:.10f}")
    print(f"edge gains = {edge_gains}")


if __name__ == "__main__":
    main()
