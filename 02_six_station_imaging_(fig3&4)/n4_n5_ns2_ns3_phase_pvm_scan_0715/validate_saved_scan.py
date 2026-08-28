#!/usr/bin/env python3
"""Independent reconstruction checks for the saved reduced joint PVMs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optimize_phase_pvm_scan import (
    compound_symmetric_state,
    copy_permutation_matrix,
    local_copy_tangents,
    make_blocks,
    phase_tangents,
    tensor_power,
)


def validate_case(root: Path, n: int, copies: int) -> dict[str, float | int]:
    rho = compound_symmetric_state(n, 0.5)
    _, tangents = phase_tangents(rho)
    sigma = tensor_power(rho, copies)
    local_tangents = local_copy_tangents(rho, tangents, copies)
    blocks = make_blocks(n, copies, sigma, local_tangents)
    data = np.load(root / f"n{n}_ns{copies}_phase_pvm.npz")

    if copies == 2:
        permutations = [
            np.array(
                [
                    [1.0 if row == (column % n) * n + column // n else 0.0
                     for column in range(n * n)]
                    for row in range(n * n)
                ],
                dtype=complex,
            )
        ]
    else:
        permutations = [
            copy_permutation_matrix(n, (1, 0, 2)),
            copy_permutation_matrix(n, (0, 2, 1)),
        ]

    dim = n**copies
    completeness = np.zeros((dim, dim), dtype=complex)
    probabilities: list[float] = []
    derivatives: list[np.ndarray] = []
    maximum_commutator = 0.0
    maximum_projector_gram = 0.0
    effect_count = 0

    for block in blocks:
        unitary = data[f"unitary_{block.label}"]
        for outcome in range(block.dimension):
            columns = np.column_stack(
                [basis @ unitary[:, outcome] for basis in block.bases]
            )
            gram = columns.conj().T @ columns
            maximum_projector_gram = max(
                maximum_projector_gram,
                float(np.linalg.norm(gram - np.eye(block.multiplicity), ord="fro")),
            )
            effect = columns @ columns.conj().T
            completeness += effect
            probabilities.append(float(np.trace(sigma @ effect).real))
            derivatives.append(
                np.array([np.trace(dot @ effect).real for dot in local_tangents])
            )
            for permutation in permutations:
                overlap = columns.conj().T @ permutation @ columns
                residual = permutation @ columns - columns @ overlap
                maximum_commutator = max(
                    maximum_commutator,
                    float(np.sqrt(2) * np.linalg.norm(residual, ord="fro")),
                )
            effect_count += 1

    p = np.asarray(probabilities)
    d = np.asarray(derivatives)
    fisher = np.einsum("ye,yf,y->ef", d, d, 1 / p)
    return {
        "N": n,
        "n_s": copies,
        "effect_count": effect_count,
        "probability_sum": float(p.sum()),
        "minimum_probability": float(p.min()),
        "full_completeness_fro": float(
            np.linalg.norm(completeness - np.eye(dim), ord="fro")
        ),
        "maximum_projector_gram_fro": maximum_projector_gram,
        "maximum_copy_permutation_commutator_fro": maximum_commutator,
        "full_fisher_vs_saved_fro": float(
            np.linalg.norm(fisher - data["fisher"], ord="fro")
        ),
        "minimum_fisher_eigenvalue": float(np.linalg.eigvalsh(fisher).min()),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    results = [
        validate_case(root, n, copies)
        for n, copies in ((4, 2), (5, 2), (4, 3), (5, 3))
    ]
    (root / "validation_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
