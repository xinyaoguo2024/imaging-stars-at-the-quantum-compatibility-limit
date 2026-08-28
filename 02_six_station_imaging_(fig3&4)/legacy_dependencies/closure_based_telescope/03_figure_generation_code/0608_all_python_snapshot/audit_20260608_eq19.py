from __future__ import annotations

import numpy as np

from audit_20260608_common import OUT, write_csv

import make_eight_station_cfi_qfi_note_tables_gauge_marginalized as cfi


def current_generators() -> list[np.ndarray]:
    mats: list[np.ndarray] = []
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        mat = np.zeros((3, 3), dtype=complex)
        mat[i, j] = 1j
        mat[j, i] = -1j
        mats.append(mat)
    return mats


def projected_sld_fisher(
    bmat: np.ndarray,
    deff: np.ndarray,
    generators: list[np.ndarray],
) -> tuple[float, np.ndarray, float]:
    mmat = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    for e, ae in enumerate(generators):
        rhs[e] = np.real(np.trace(ae @ deff))
        for ep, aep in enumerate(generators):
            mmat[e, ep] = 0.5 * np.real(np.trace(ae @ (bmat @ aep + aep @ bmat)))
    coeff = np.linalg.pinv(mmat, rcond=1e-12) @ rhs
    lmat = sum(float(x) * amat for x, amat in zip(coeff, generators))
    qfi = float(np.real(np.trace(deff @ lmat)))
    residual = deff - 0.5 * (bmat @ lmat + lmat @ bmat)
    rel_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(deff), 1e-300))
    return qfi, coeff, rel_residual


def run_eq19_checks() -> dict[str, object]:
    cases = [
        ("symmetric", 0.30, 0.30, 0.30, 1.50, 1.50, 1.50),
        ("moderately_asymmetric", 0.30, 0.20, 0.10, 1.50, 1.40, 1.60),
        ("weak_edge", 0.30, 0.01, 0.02, 1.50, 1.50, 1.50),
        ("load_asymmetric", 0.20, 0.11, 0.06, 1.20, 1.90, 1.55),
    ]
    generators = current_generators()
    rows: list[dict[str, object]] = []
    for name, g12, g23, g31, s1, s2, s3 in cases:
        bmat, edge_derivs = cfi.triangle_bmat_and_edge_derivatives(g12, g23, g31, s1, s2, s3)
        deff, schur_fisher, weights, _edge_j = cfi.triangle_effective_closure_derivative(bmat, edge_derivs)
        full_l, full_fisher = cfi.sld_from_b_and_d(bmat, deff)
        proj_fisher, coeff, rel_residual = projected_sld_fisher(bmat, deff, generators)
        full_rel_span = float(
            np.linalg.norm(full_l - sum(float(x) * a for x, a in zip(coeff, generators)))
            / max(np.linalg.norm(full_l), 1e-300)
        )
        rows.append(
            {
                "case": name,
                "g12": g12,
                "g23": g23,
                "g31": g31,
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "schur_fisher": schur_fisher,
                "full_sld_fisher": full_fisher,
                "eq19_projected_fisher": proj_fisher,
                "rel_projected_minus_schur": abs(proj_fisher - schur_fisher) / max(schur_fisher, 1e-300),
                "rel_full_minus_schur": abs(full_fisher - schur_fisher) / max(schur_fisher, 1e-300),
                "rel_sld_equation_residual": rel_residual,
                "rel_full_sld_outside_current_span": full_rel_span,
                "qeff_12": float(weights[0]),
                "qeff_23": float(weights[1]),
                "qeff_31": float(weights[2]),
                "x12": float(coeff[0]),
                "x23": float(coeff[1]),
                "x31": float(coeff[2]),
            }
        )
    csv_path = OUT / "eq19_numeric_checks.csv"
    write_csv(csv_path, rows)
    summary = {
        "csv": str(csv_path),
        "max_rel_projected_minus_schur": max(float(r["rel_projected_minus_schur"]) for r in rows),
        "max_rel_full_minus_schur": max(float(r["rel_full_minus_schur"]) for r in rows),
        "max_rel_sld_equation_residual": max(float(r["rel_sld_equation_residual"]) for r in rows),
        "max_rel_full_sld_outside_current_span": max(float(r["rel_full_sld_outside_current_span"]) for r in rows),
        "n_cases": len(rows),
    }
    return summary
