#!/usr/bin/env python3
"""Permutation-invariant phase-PVM scan for N=4,5 and n_s=2,3.

The benchmark state is compound symmetric,

    rho = ((1-g) I + g 11^T) / N,

and the unknowns are all E=N(N-1)/2 edge phases at fixed visibility
magnitudes.  Local coordinates are h=sqrt(n_s) (phi-phi_0), so the Fisher
matrix of n_s repetitive single-copy receivers is exactly the one-copy CFIM.

The joint receiver is a projective measurement in the commutant of copy
permutations.  For n_s=2 it is optimized in the symmetric and antisymmetric
blocks.  For n_s=3 it is optimized in the [3], [2,1], and [1,1,1] Schur
blocks; the mixed block has multiplicity two and is not resolved.  Positivity,
orthogonality, completeness, and copy-permutation invariance are therefore
exact by construction.  The A-optimal objective Tr[J^{-1}] is nonconvex, so
the reported result is the best of multiple Haar-random restarts, not a global
optimality certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def hermitian(a: np.ndarray) -> np.ndarray:
    return (a + a.conj().T) / 2


def unitary_retraction(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0, diagonal / np.abs(diagonal), 1)
    return q @ np.diag(phases.conj())


def compound_symmetric_state(n: int, g: float) -> np.ndarray:
    if not (-1 / (n - 1) < g < 1):
        raise ValueError("g lies outside the full-rank compound-symmetric range")
    rho = ((1 - g) * np.eye(n) + g * np.ones((n, n))) / n
    if np.linalg.eigvalsh(rho).min() <= 1e-12:
        raise ValueError("the local optimizer requires a full-rank state")
    return rho.astype(complex)


def phase_tangents(rho: np.ndarray) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    n = rho.shape[0]
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    tangents: list[np.ndarray] = []
    for i, j in edges:
        dot = np.zeros_like(rho)
        dot[i, j] = 1j * rho[i, j]
        dot[j, i] = -1j * rho[j, i]
        tangents.append(dot)
    return edges, tangents


def tensor_power(a: np.ndarray, copies: int) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for _ in range(copies):
        result = np.kron(result, a)
    return result


def local_copy_tangents(
    rho: np.ndarray, tangents: list[np.ndarray], copies: int
) -> list[np.ndarray]:
    """Derivatives in h=sqrt(copies)*(phi-phi0) coordinates."""
    local: list[np.ndarray] = []
    for dot in tangents:
        total = np.zeros((rho.shape[0] ** copies,) * 2, dtype=complex)
        for position in range(copies):
            factors = [rho] * copies
            factors = list(factors)
            factors[position] = dot
            term = factors[0]
            for factor in factors[1:]:
                term = np.kron(term, factor)
            total += term
        local.append(total / np.sqrt(copies))
    return local


def swap_basis(n: int) -> tuple[np.ndarray, np.ndarray]:
    symmetric: list[np.ndarray] = []
    antisymmetric: list[np.ndarray] = []
    for i in range(n):
        ket = np.zeros(n * n, dtype=complex)
        ket[n * i + i] = 1
        symmetric.append(ket)
    for i in range(n):
        for j in range(i + 1, n):
            plus = np.zeros(n * n, dtype=complex)
            minus = np.zeros(n * n, dtype=complex)
            plus[n * i + j] = plus[n * j + i] = 1 / np.sqrt(2)
            minus[n * i + j] = 1 / np.sqrt(2)
            minus[n * j + i] = -1 / np.sqrt(2)
            symmetric.append(plus)
            antisymmetric.append(minus)
    return np.column_stack(symmetric), np.column_stack(antisymmetric)


def copy_permutation_matrix(n: int, permutation: tuple[int, int, int]) -> np.ndarray:
    dim = n**3
    matrix = np.zeros((dim, dim), dtype=complex)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                source = (i, j, k)
                target = tuple(source[permutation[a]] for a in range(3))
                source_index = np.ravel_multi_index(source, (n, n, n))
                target_index = np.ravel_multi_index(target, (n, n, n))
                matrix[target_index, source_index] = 1
    return matrix


def projector_basis(projector: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    values, vectors = np.linalg.eigh(hermitian(projector))
    return vectors[:, values > threshold]


def schur_three_bases(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aligned bases for [3], two copies of [2,1], and [1,1,1]."""
    ident = np.eye(n**3)
    swap_12 = copy_permutation_matrix(n, (1, 0, 2))
    swap_23 = copy_permutation_matrix(n, (0, 2, 1))
    swap_13 = copy_permutation_matrix(n, (2, 1, 0))
    cycle_123 = copy_permutation_matrix(n, (1, 2, 0))
    cycle_132 = copy_permutation_matrix(n, (2, 0, 1))
    p_sym = (ident + swap_12 + swap_23 + swap_13 + cycle_123 + cycle_132) / 6
    p_anti = (ident - swap_12 - swap_23 - swap_13 + cycle_123 + cycle_132) / 6
    p_mixed = ident - p_sym - p_anti

    u_sym = projector_basis(p_sym)
    u_anti = projector_basis(p_anti)
    p_plus = p_mixed @ ((ident + swap_12) / 2) @ p_mixed
    u_plus = projector_basis(p_plus)
    u_minus = (2 / np.sqrt(3)) * ((ident - swap_12) / 2) @ swap_23 @ u_plus
    complete = (
        u_sym @ u_sym.conj().T
        + u_plus @ u_plus.conj().T
        + u_minus @ u_minus.conj().T
        + u_anti @ u_anti.conj().T
    )
    if np.linalg.norm(complete - ident, ord="fro") > 1e-9:
        raise RuntimeError("three-copy Schur basis is incomplete")
    return u_sym, u_plus, u_minus, u_anti


@dataclass
class ReducedBlock:
    label: str
    bases: list[np.ndarray]
    multiplicity: int
    sigma: np.ndarray
    tangents: np.ndarray

    @property
    def dimension(self) -> int:
        return self.sigma.shape[0]


def make_blocks(
    n: int, copies: int, sigma: np.ndarray, local_tangents: list[np.ndarray]
) -> list[ReducedBlock]:
    if copies == 2:
        u_sym, u_anti = swap_basis(n)
        raw = [("symmetric", [u_sym]), ("antisymmetric", [u_anti])]
    elif copies == 3:
        u_sym, u_plus, u_minus, u_anti = schur_three_bases(n)
        raw = [
            ("symmetric", [u_sym]),
            ("mixed", [u_plus, u_minus]),
            ("antisymmetric", [u_anti]),
        ]
    else:
        raise ValueError("this scan supports only n_s=2 or 3")

    blocks: list[ReducedBlock] = []
    for label, bases in raw:
        representative = bases[0]
        sigma_block = hermitian(representative.conj().T @ sigma @ representative)
        tangent_blocks = np.asarray(
            [hermitian(representative.conj().T @ dot @ representative) for dot in local_tangents]
        )
        for other in bases[1:]:
            sigma_other = hermitian(other.conj().T @ sigma @ other)
            relative = np.linalg.norm(sigma_other - sigma_block, ord="fro")
            if relative > 1e-9:
                raise RuntimeError(f"unaligned {label} state blocks: {relative}")
            for index, dot in enumerate(local_tangents):
                tangent_other = hermitian(other.conj().T @ dot @ other)
                relative = np.linalg.norm(tangent_other - tangent_blocks[index], ord="fro")
                if relative > 1e-8:
                    raise RuntimeError(f"unaligned {label} tangent blocks: {relative}")
        blocks.append(
            ReducedBlock(
                label=label,
                bases=bases,
                multiplicity=len(bases),
                sigma=sigma_block,
                tangents=tangent_blocks,
            )
        )
    return blocks


@dataclass
class Evaluation:
    objective: float
    fisher: np.ndarray
    covariance: np.ndarray
    gradients: list[np.ndarray]
    probabilities: list[np.ndarray]
    derivatives: list[np.ndarray]


def evaluate(blocks: list[ReducedBlock], unitaries: list[np.ndarray]) -> Evaluation:
    probabilities: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    fisher = np.zeros((blocks[0].tangents.shape[0],) * 2)
    for block, unitary in zip(blocks, unitaries, strict=True):
        p0 = np.einsum(
            "iy,ij,jy->y", unitary.conj(), block.sigma, unitary
        ).real
        d0 = np.asarray(
            [
                np.einsum("iy,ij,jy->y", unitary.conj(), dot, unitary).real
                for dot in block.tangents
            ]
        ).T
        if p0.min() <= 1e-14:
            raise np.linalg.LinAlgError("encountered an effectively zero probability")
        fisher += block.multiplicity * np.einsum(
            "ye,yf,y->ef", d0, d0, 1 / p0
        )
        probabilities.append(block.multiplicity * p0)
        derivatives.append(block.multiplicity * d0)

    fisher = hermitian(fisher).real
    covariance = np.linalg.inv(fisher)
    objective = float(np.trace(covariance))
    inverse_squared = covariance @ covariance
    gradients: list[np.ndarray] = []
    for block, unitary, p_full, d_full in zip(
        blocks, unitaries, probabilities, derivatives, strict=True
    ):
        # Work with one irreducible-copy probability/derivative and multiply
        # the objective gradient by the Schur multiplicity.
        p0 = p_full / block.multiplicity
        d0 = d_full / block.multiplicity
        h = d0 @ inverse_squared.T
        quadratic = np.einsum("ye,ye->y", d0, h)
        tangent_action = np.einsum(
            "ye,eij,jy->iy", h, block.tangents, unitary, optimize=True
        )
        euclidean = block.multiplicity * (
            -2 * tangent_action / p0[None, :]
            + (block.sigma @ unitary) * quadratic[None, :] / p0[None, :] ** 2
        )
        overlap = unitary.conj().T @ euclidean
        gradient = euclidean - unitary @ hermitian(overlap)
        gradients.append(gradient)
    return Evaluation(objective, fisher, covariance, gradients, probabilities, derivatives)


@dataclass
class RestartRecord:
    seed: int
    objective: float
    gradient_norm: float
    steps_taken: int


def optimize_blocks(
    blocks: list[ReducedBlock], restarts: int, steps: int, seed: int
) -> tuple[list[np.ndarray], Evaluation, list[RestartRecord]]:
    rng = np.random.default_rng(seed)
    best_unitaries: list[np.ndarray] | None = None
    best_evaluation: Evaluation | None = None
    records: list[RestartRecord] = []

    for restart in range(restarts):
        unitaries = [
            unitary_retraction(
                rng.normal(size=(block.dimension, block.dimension))
                + 1j * rng.normal(size=(block.dimension, block.dimension))
            )
            for block in blocks
        ]
        current = evaluate(blocks, unitaries)
        learning_rate = 0.1
        gradient_norm = np.inf
        steps_taken = 0
        for iteration in range(steps):
            gradient_norm = float(
                np.sqrt(sum(np.linalg.norm(g, ord="fro") ** 2 for g in current.gradients))
            )
            if gradient_norm < 1e-9:
                break
            trial_step = learning_rate
            accepted = False
            for _ in range(24):
                trial_unitaries = [
                    unitary_retraction(unitary - trial_step * gradient)
                    for unitary, gradient in zip(unitaries, current.gradients, strict=True)
                ]
                try:
                    trial = evaluate(blocks, trial_unitaries)
                except np.linalg.LinAlgError:
                    trial_step *= 0.5
                    continue
                if trial.objective < (
                    current.objective - 1e-4 * trial_step * gradient_norm**2
                ):
                    unitaries = trial_unitaries
                    current = trial
                    learning_rate = min(1.4 * trial_step, 0.5)
                    accepted = True
                    break
                trial_step *= 0.5
            steps_taken = iteration + 1
            if not accepted:
                break
        gradient_norm = float(
            np.sqrt(sum(np.linalg.norm(g, ord="fro") ** 2 for g in current.gradients))
        )
        record = RestartRecord(
            seed=seed + restart,
            objective=current.objective,
            gradient_norm=gradient_norm,
            steps_taken=steps_taken,
        )
        records.append(record)
        print(
            f"restart {restart + 1:3d}/{restarts}: "
            f"risk={record.objective:.10g}, grad={gradient_norm:.3e}, steps={steps_taken}",
            flush=True,
        )
        if best_evaluation is None or current.objective < best_evaluation.objective:
            best_evaluation = current
            best_unitaries = [unitary.copy() for unitary in unitaries]

    if best_unitaries is None or best_evaluation is None:
        raise RuntimeError("all optimization restarts failed")
    return best_unitaries, best_evaluation, records


def refine_from_initial(
    blocks: list[ReducedBlock],
    unitaries: list[np.ndarray],
    steps: int,
) -> tuple[list[np.ndarray], Evaluation, RestartRecord]:
    """Continue Riemannian descent from a saved best solution."""
    current = evaluate(blocks, unitaries)
    learning_rate = 0.1
    steps_taken = 0
    for iteration in range(steps):
        gradient_norm = float(
            np.sqrt(sum(np.linalg.norm(g, ord="fro") ** 2 for g in current.gradients))
        )
        if gradient_norm < 1e-10:
            break
        trial_step = learning_rate
        accepted = False
        for _ in range(28):
            trial_unitaries = [
                unitary_retraction(unitary - trial_step * gradient)
                for unitary, gradient in zip(unitaries, current.gradients, strict=True)
            ]
            try:
                trial = evaluate(blocks, trial_unitaries)
            except np.linalg.LinAlgError:
                trial_step *= 0.5
                continue
            if trial.objective < (
                current.objective - 1e-4 * trial_step * gradient_norm**2
            ):
                unitaries = trial_unitaries
                current = trial
                learning_rate = min(1.4 * trial_step, 0.5)
                accepted = True
                break
            trial_step *= 0.5
        steps_taken = iteration + 1
        if not accepted:
            break
    gradient_norm = float(
        np.sqrt(sum(np.linalg.norm(g, ord="fro") ** 2 for g in current.gradients))
    )
    record = RestartRecord(
        seed=-1,
        objective=current.objective,
        gradient_norm=gradient_norm,
        steps_taken=steps_taken,
    )
    print(
        f"refinement: risk={record.objective:.10g}, "
        f"grad={gradient_norm:.3e}, steps={steps_taken}",
        flush=True,
    )
    return unitaries, current, record


def load_existing_case(
    output: Path, n: int, copies: int, blocks: list[ReducedBlock]
) -> tuple[list[np.ndarray], list[RestartRecord]]:
    path = output / f"n{n}_ns{copies}_phase_pvm.npz"
    if not path.exists():
        raise FileNotFoundError(f"cannot refine missing result: {path}")
    data = np.load(path)
    unitaries = [np.asarray(data[f"unitary_{block.label}"]) for block in blocks]
    objectives = np.asarray(data["restart_objectives"])
    gradients = np.asarray(data["restart_gradient_norms"])
    steps = np.asarray(data["restart_steps"])
    records = [
        RestartRecord(
            seed=index,
            objective=float(objective),
            gradient_norm=float(gradient),
            steps_taken=int(step),
        )
        for index, (objective, gradient, step) in enumerate(
            zip(objectives, gradients, steps, strict=True)
        )
    ]
    return unitaries, records


def one_copy_uniform_edge_fisher(n: int, g: float) -> float:
    # Each edge has two quadrature outcomes and is selected with weight 1/(N-1).
    return 2 * g**2 / (n * (n - 1))


def quantum_fisher_matrix(rho: np.ndarray, tangents: list[np.ndarray]) -> np.ndarray:
    values, vectors = np.linalg.eigh(rho)
    transformed = [vectors.conj().T @ dot @ vectors for dot in tangents]
    result = np.zeros((len(tangents), len(tangents)))
    for e, left in enumerate(transformed):
        for f, right in enumerate(transformed):
            total = 0.0j
            for a in range(len(values)):
                for b in range(len(values)):
                    total += (
                        2 * left[a, b] * right[b, a] / (values[a] + values[b])
                    )
            result[e, f] = total.real
    return hermitian(result).real


def diagnostics(
    n: int,
    copies: int,
    g: float,
    edges: list[tuple[int, int]],
    blocks: list[ReducedBlock],
    unitaries: list[np.ndarray],
    evaluation: Evaluation,
    records: list[RestartRecord],
) -> dict[str, object]:
    edge_dimension = len(edges)
    one_fisher = one_copy_uniform_edge_fisher(n, g)
    repetitive_risk = edge_dimension / one_fisher
    eig = np.linalg.eigvalsh(evaluation.fisher)
    rho = compound_symmetric_state(n, g)
    _, one_tangents = phase_tangents(rho)
    qfi = quantum_fisher_matrix(rho, one_tangents)
    qfi_values, qfi_vectors = np.linalg.eigh(qfi)
    qfi_inverse_half = qfi_vectors @ np.diag(qfi_values ** -0.5) @ qfi_vectors.T
    qfi_normalized = hermitian(
        qfi_inverse_half @ evaluation.fisher @ qfi_inverse_half
    ).real
    all_probabilities = np.concatenate(evaluation.probabilities)
    all_derivatives = np.vstack(evaluation.derivatives)
    fisher_recomputed = np.einsum(
        "ye,yf,y->ef", all_derivatives, all_derivatives, 1 / all_probabilities
    )
    completeness = 0.0
    orthogonality = 0.0
    for block, unitary in zip(blocks, unitaries, strict=True):
        completeness = max(
            completeness,
            float(np.linalg.norm(unitary @ unitary.conj().T - np.eye(block.dimension))),
        )
        gram = unitary.conj().T @ unitary
        orthogonality = max(
            orthogonality,
            float(np.linalg.norm(gram - np.eye(block.dimension))),
        )
    objectives = np.array([record.objective for record in records])
    gradients = np.array([record.gradient_norm for record in records])
    return {
        "N": n,
        "n_s": copies,
        "g": g,
        "raw_hilbert_dimension": n**copies,
        "edge_phase_dimension": edge_dimension,
        "edges_one_indexed": [[i + 1, j + 1] for i, j in edges],
        "local_coordinate": "h=sqrt(n_s)*(phi-phi0)",
        "schur_blocks": [
            {
                "label": block.label,
                "reduced_dimension": block.dimension,
                "multiplicity": block.multiplicity,
                "effect_rank": block.multiplicity,
            }
            for block in blocks
        ],
        "number_of_joint_outcomes": int(sum(block.dimension for block in blocks)),
        "one_copy_uniform_edge_fisher_diagonal": one_fisher,
        "repetitive_fisher_eigenvalues": [one_fisher] * edge_dimension,
        "repetitive_A_risk": repetitive_risk,
        "joint_fisher_eigenvalues": eig.tolist(),
        "qfi_eigenvalues": qfi_values.tolist(),
        "qfi_A_risk": float(np.trace(np.linalg.inv(qfi))),
        "joint_to_qfi_generalized_eigenvalues": np.linalg.eigvalsh(
            qfi_normalized
        ).tolist(),
        "maximum_joint_to_qfi_ratio": float(np.linalg.eigvalsh(qfi_normalized).max()),
        "joint_A_risk": evaluation.objective,
        "A_risk_gain": repetitive_risk / evaluation.objective,
        "A_risk_SNR_gain": float(np.sqrt(repetitive_risk / evaluation.objective)),
        "minimum_direction_Fisher_gain": float(eig.min() / one_fisher),
        "maximum_direction_Fisher_gain": float(eig.max() / one_fisher),
        "arithmetic_mean_Fisher_gain": float(eig.mean() / one_fisher),
        "harmonic_mean_Fisher_gain": repetitive_risk / evaluation.objective,
        "minimum_outcome_probability": float(all_probabilities.min()),
        "probability_sum": float(all_probabilities.sum()),
        "fisher_reconstruction_fro": float(
            np.linalg.norm(fisher_recomputed - evaluation.fisher, ord="fro")
        ),
        "reduced_completeness_fro": completeness,
        "reduced_projector_orthogonality_fro": orthogonality,
        "best_riemannian_gradient_norm": float(
            min(
                record.gradient_norm
                for record in records
                if abs(record.objective - evaluation.objective) < 1e-8
            )
        ),
        "restart_objective_min": float(objectives.min()),
        "restart_objective_median": float(np.median(objectives)),
        "restart_objective_max": float(objectives.max()),
        "restart_objective_std": float(objectives.std()),
        "restart_gradient_median": float(np.median(gradients)),
        "local_nonconvex_optimum_only": True,
    }


def save_case(
    output: Path,
    summary: dict[str, object],
    blocks: list[ReducedBlock],
    unitaries: list[np.ndarray],
    evaluation: Evaluation,
    records: list[RestartRecord],
) -> None:
    stem = f"n{summary['N']}_ns{summary['n_s']}_phase_pvm"
    with (output / f"{stem}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    arrays: dict[str, np.ndarray] = {
        "fisher": evaluation.fisher,
        "covariance": evaluation.covariance,
        "probabilities": np.concatenate(evaluation.probabilities),
        "derivatives": np.vstack(evaluation.derivatives),
        "restart_objectives": np.array([record.objective for record in records]),
        "restart_gradient_norms": np.array([record.gradient_norm for record in records]),
        "restart_steps": np.array([record.steps_taken for record in records]),
    }
    for block, unitary in zip(blocks, unitaries, strict=True):
        arrays[f"unitary_{block.label}"] = unitary
        arrays[f"sigma_{block.label}"] = block.sigma
        arrays[f"tangents_{block.label}"] = block.tangents
        for index, basis in enumerate(block.bases):
            arrays[f"schur_basis_{block.label}_{index}"] = basis
    np.savez_compressed(output / f"{stem}.npz", **arrays)


def write_scan_tables(output: Path, summaries: list[dict[str, object]]) -> None:
    fields = [
        "N",
        "n_s",
        "raw_hilbert_dimension",
        "number_of_joint_outcomes",
        "repetitive_A_risk",
        "joint_A_risk",
        "A_risk_gain",
        "A_risk_SNR_gain",
        "minimum_direction_Fisher_gain",
        "arithmetic_mean_Fisher_gain",
        "maximum_direction_Fisher_gain",
        "restart_objective_min",
        "restart_objective_median",
        "restart_objective_max",
        "best_riemannian_gradient_norm",
    ]
    ordered = sorted(summaries, key=lambda item: (item["n_s"], item["N"]))
    with (output / "scan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in ordered:
            writer.writerow({field: summary[field] for field in fields})

    lines = [
        "# N=4,5 and n_s=2,3 phase-PVM scan",
        "",
        "Benchmark: compound-symmetric full-rank state with g=0.5; all edge phases are estimated; local coordinates are h=sqrt(n_s)(phi-phi0).",
        "",
        "| N | n_s | raw dim | outcomes | repetitive A-risk | joint A-risk | A-risk gain | min FI gain | mean FI gain | max FI gain |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in ordered:
        lines.append(
            "| {N} | {n_s} | {raw_hilbert_dimension} | {number_of_joint_outcomes} | "
            "{repetitive_A_risk:.6g} | {joint_A_risk:.6g} | {A_risk_gain:.6g} | "
            "{minimum_direction_Fisher_gain:.6g} | {arithmetic_mean_Fisher_gain:.6g} | "
            "{maximum_direction_Fisher_gain:.6g} |".format(**summary)
        )
    lines.extend(
        [
            "",
            "These are best-of-restart local optima of a nonconvex A-optimal projective-measurement search, not global optimality certificates.",
            "For n_s=3, the mixed Schur effect has rank two and its representation multiplicity is not resolved.",
        ]
    )
    (output / "scan_summary.md").write_text("\n".join(lines) + "\n")


def parse_cases(text: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in text.split(","):
        n_text, ns_text = item.split("x")
        cases.append((int(n_text), int(ns_text)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="4x2,5x2,4x3,5x3")
    parser.add_argument("--g", type=float, default=0.5)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--refine-existing",
        action="store_true",
        help="continue descent from each case's saved best block unitaries",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for case_index, (n, copies) in enumerate(parse_cases(args.cases)):
        print(f"\n=== N={n}, n_s={copies} ===", flush=True)
        rho = compound_symmetric_state(n, args.g)
        edges, tangents = phase_tangents(rho)
        sigma = tensor_power(rho, copies)
        local_tangents = local_copy_tangents(rho, tangents, copies)
        blocks = make_blocks(n, copies, sigma, local_tangents)
        print(
            "raw dimension:", n**copies,
            "reduced blocks:",
            [(block.label, block.dimension, block.multiplicity) for block in blocks],
            flush=True,
        )
        if args.refine_existing:
            unitaries, records = load_existing_case(args.output, n, copies, blocks)
            unitaries, evaluation, refinement = refine_from_initial(
                blocks, unitaries, args.steps
            )
            records.append(refinement)
        else:
            unitaries, evaluation, records = optimize_blocks(
                blocks,
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed + 1000 * case_index,
            )
        summary = diagnostics(
            n, copies, args.g, edges, blocks, unitaries, evaluation, records
        )
        save_case(args.output, summary, blocks, unitaries, evaluation, records)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    # Merge with already completed summaries when a subset of cases is run.
    by_case = {(item["N"], item["n_s"]): item for item in summaries}
    for path in args.output.glob("n*_ns*_phase_pvm_summary.json"):
        with path.open() as handle:
            item = json.load(handle)
        by_case.setdefault((item["N"], item["n_s"]), item)
    write_scan_tables(args.output, list(by_case.values()))


if __name__ == "__main__":
    main()
