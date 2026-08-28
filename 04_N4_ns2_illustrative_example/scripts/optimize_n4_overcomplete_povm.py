#!/usr/bin/env python3
r"""Optimize and compare finite-copy N=4 phase receivers.

The benchmark is the full-rank compound-symmetric state used in Sec. II of
the Supplemental Material.  The saved Schur PVM is enlarged to a q=2
overcomplete rank-one POVM in every copy-permutation block,

    M_{\lambda y}=|a_{\lambda y}><a_{\lambda y}|\otimes I_{m_\lambda},
    A_\lambda A_\lambda^\dagger=I.

The script optimizes the Parseval frames on their row-Stiefel manifolds,
exports the explicit effects through the saved frames, and produces the
nuisance-aware physical-edge Fisher comparison used in the SM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = Path(os.environ.get("N4_DATA_DIR", str(PACKAGE / "data")))
FIGURES = Path(os.environ.get("N4_FIGURE_DIR", str(PACKAGE / "figures")))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from optimize_phase_pvm_scan import (  # noqa: E402
    compound_symmetric_state,
    local_copy_tangents,
    make_blocks,
    phase_tangents,
    quantum_fisher_matrix,
    tensor_power,
)


def hermitian(a: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(a) + np.asarray(a).conj().T)


def row_retraction(a: np.ndarray) -> np.ndarray:
    r"""Retract a wide matrix onto (A A^\dagger=I)."""
    q, r = np.linalg.qr(np.asarray(a).conj().T, mode="reduced")
    diagonal = np.diag(r)
    phases = np.where(np.abs(diagonal) > 0, diagonal / np.abs(diagonal), 1)
    return (q @ np.diag(phases.conj())).conj().T


def split_pvm(unitary: np.ndarray, redundancy: int) -> np.ndarray:
    return np.concatenate([unitary / np.sqrt(redundancy)] * redundancy, axis=1)


@dataclass
class Evaluation:
    risk: float
    fisher: np.ndarray
    covariance: np.ndarray
    gradients: list[np.ndarray]
    probabilities: list[np.ndarray]
    derivatives: list[np.ndarray]


def evaluate(blocks, frames: list[np.ndarray]) -> Evaluation:
    fisher = np.zeros((blocks[0].tangents.shape[0],) * 2)
    probabilities: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    for block, frame in zip(blocks, frames, strict=True):
        p = np.einsum("iy,ij,jy->y", frame.conj(), block.sigma, frame).real
        if p.min() <= 1e-15:
            raise np.linalg.LinAlgError("effect with effectively zero probability")
        d = np.asarray(
            [
                np.einsum("iy,ij,jy->y", frame.conj(), dot, frame).real
                for dot in block.tangents
            ]
        ).T
        fisher += block.multiplicity * np.einsum(
            "ye,yf,y->ef", d, d, 1 / p, optimize=True
        )
        probabilities.append(p)
        derivatives.append(d)

    fisher = hermitian(fisher).real
    covariance = np.linalg.inv(fisher)
    sensitivity = covariance @ covariance
    gradients: list[np.ndarray] = []
    for block, frame, p, d in zip(
        blocks, frames, probabilities, derivatives, strict=True
    ):
        h = d @ sensitivity.T
        quadratic = np.einsum("ye,ye->y", d, h)
        tangent_action = np.einsum(
            "ye,eij,jy->iy", h, block.tangents, frame, optimize=True
        )
        euclidean = block.multiplicity * (
            -2 * tangent_action / p[None, :]
            + (block.sigma @ frame) * quadratic[None, :] / p[None, :] ** 2
        )
        gradients.append(euclidean - hermitian(euclidean @ frame.conj().T) @ frame)
    return Evaluation(
        float(np.trace(covariance)),
        fisher,
        covariance,
        gradients,
        probabilities,
        derivatives,
    )


def gradient_norm(evaluation: Evaluation) -> float:
    return float(
        np.sqrt(sum(np.linalg.norm(g, ord="fro") ** 2 for g in evaluation.gradients))
    )


def completeness_error(frames: list[np.ndarray]) -> float:
    return float(
        max(
            np.linalg.norm(
                frame @ frame.conj().T - np.eye(frame.shape[0]), ord="fro"
            )
            for frame in frames
        )
    )


def descend(
    blocks,
    initial: list[np.ndarray],
    *,
    steps: int,
) -> tuple[list[np.ndarray], Evaluation, int]:
    frames = [frame.copy() for frame in initial]
    current = evaluate(blocks, frames)
    rate = 0.025
    steps_taken = 0
    for iteration in range(steps):
        norm = gradient_norm(current)
        if norm < 1e-9:
            break
        trial_rate = rate
        accepted = False
        for _ in range(30):
            trial_frames = [
                row_retraction(frame - trial_rate * gradient)
                for frame, gradient in zip(frames, current.gradients, strict=True)
            ]
            try:
                trial = evaluate(blocks, trial_frames)
            except np.linalg.LinAlgError:
                trial_rate *= 0.5
                continue
            if trial.risk < current.risk - 1e-4 * trial_rate * norm**2:
                frames = trial_frames
                current = trial
                rate = min(1.3 * trial_rate, 0.15)
                accepted = True
                break
            trial_rate *= 0.5
        steps_taken = iteration + 1
        if not accepted:
            break
    return frames, current, steps_taken


def optimize_case(
    copies: int,
    *,
    redundancy: int,
    restarts: int,
    steps: int,
    seed: int,
    perturbation: float,
) -> tuple[list, list[np.ndarray], Evaluation, list[dict[str, object]]]:
    rho = compound_symmetric_state(4, 0.5)
    _, tangents = phase_tangents(rho)
    sigma = tensor_power(rho, copies)
    dots = local_copy_tangents(rho, tangents, copies)
    blocks = make_blocks(4, copies, sigma, dots)
    saved = np.load(DATA / f"n4_ns{copies}_phase_pvm.npz")
    pvm = [saved[f"unitary_{block.label}"] for block in blocks]
    split = [split_pvm(unitary, redundancy) for unitary in pvm]
    best_frames = [frame.copy() for frame in split]
    best = evaluate(blocks, best_frames)
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = [
        {
            "restart": -1,
            "initialization": "exact_split_pvm",
            "risk": best.risk,
            "gradient_norm": gradient_norm(best),
            "steps": 0,
        }
    ]

    for restart in range(restarts):
        if restart == restarts - 1:
            initial = [
                row_retraction(
                    rng.normal(size=frame.shape)
                    + 1j * rng.normal(size=frame.shape)
                )
                for frame in split
            ]
            initialization = "random_tight_frame"
        else:
            scale = perturbation * (1 + restart / max(1, restarts - 1))
            initial = [
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
            initialization = f"perturbed_split_{scale:.6g}"
        frames, evaluation, steps_taken = descend(blocks, initial, steps=steps)
        record = {
            "restart": restart,
            "initialization": initialization,
            "risk": evaluation.risk,
            "gradient_norm": gradient_norm(evaluation),
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

    # A long continuation removes residual first-order error from the best run.
    best_frames, best, refined_steps = descend(
        blocks, best_frames, steps=max(steps, 4000)
    )
    records.append(
        {
            "restart": "refinement",
            "initialization": "saved_best",
            "risk": best.risk,
            "gradient_norm": gradient_norm(best),
            "steps": refined_steps,
        }
    )
    return blocks, best_frames, best, records


def refine_saved_case(
    copies: int,
    *,
    steps: int,
) -> tuple[list, list[np.ndarray], Evaluation, list[dict[str, object]]]:
    """Continue the descent from a previously saved q=2 POVM."""
    rho = compound_symmetric_state(4, 0.5)
    _, tangents = phase_tangents(rho)
    sigma = tensor_power(rho, copies)
    dots = local_copy_tangents(rho, tangents, copies)
    blocks = make_blocks(4, copies, sigma, dots)
    saved = np.load(DATA / f"n4_ns{copies}_phase_povm_q2.npz")
    frames = [saved[f"frame_{block.label}"] for block in blocks]
    initial = evaluate(blocks, frames)
    refined_frames, refined, steps_taken = descend(blocks, frames, steps=steps)
    records = [
        {
            "restart": "refine_existing",
            "initialization": "saved_q2_povm",
            "initial_risk": initial.risk,
            "initial_gradient_norm": gradient_norm(initial),
            "risk": refined.risk,
            "gradient_norm": gradient_norm(refined),
            "steps": steps_taken,
        }
    ]
    return blocks, refined_frames, refined, records


def fisher_summary(
    copies: int,
    blocks,
    frames: list[np.ndarray],
    evaluation: Evaluation,
    records: list[dict[str, object]],
) -> dict[str, object]:
    repetitive_fisher = 1 / 24
    edge_gains = 24 / np.diag(evaluation.covariance)
    mode_gains = np.linalg.eigvalsh(evaluation.fisher) / repetitive_fisher
    weighted_probabilities = np.concatenate(
        [
            block.multiplicity * p
            for block, p in zip(blocks, evaluation.probabilities, strict=True)
        ]
    )
    weighted_derivatives = np.vstack(
        [
            block.multiplicity * d
            for block, d in zip(blocks, evaluation.derivatives, strict=True)
        ]
    )
    reconstructed = np.einsum(
        "ye,yf,y->ef",
        weighted_derivatives,
        weighted_derivatives,
        1 / weighted_probabilities,
        optimize=True,
    )
    # Multiplicity enters the probability and derivative once, so the above
    # expression reconstructs the same block-weighted Fisher matrix.
    rho = compound_symmetric_state(4, 0.5)
    _, tangents = phase_tangents(rho)
    qfi = quantum_fisher_matrix(rho, tangents)
    qvalues, qvectors = np.linalg.eigh(qfi)
    qinvhalf = qvectors @ np.diag(qvalues**-0.5) @ qvectors.T
    normalized = hermitian(qinvhalf @ evaluation.fisher @ qinvhalf).real
    return {
        "N": 4,
        "n_s": copies,
        "g": 0.5,
        "receiver_class": (
            "q=2 overcomplete rank-one Parseval-frame POVM in each "
            "copy-permutation Schur block"
        ),
        "block_dimensions": [block.dimension for block in blocks],
        "block_multiplicities": [block.multiplicity for block in blocks],
        "block_outcomes": [frame.shape[1] for frame in frames],
        "number_of_reported_outcomes": int(sum(frame.shape[1] for frame in frames)),
        "risk": evaluation.risk,
        "A_risk_gain": 144 / evaluation.risk,
        "SNR_gain": float(np.sqrt(144 / evaluation.risk)),
        "edge_directional_Fisher_gains": edge_gains.tolist(),
        "edge_directional_Fisher_gain_mean": float(edge_gains.mean()),
        "edge_directional_Fisher_gain_min": float(edge_gains.min()),
        "edge_directional_Fisher_gain_max": float(edge_gains.max()),
        "Fisher_mode_gains": mode_gains.tolist(),
        "Fisher_mode_gain_mean": float(mode_gains.mean()),
        "minimum_outcome_probability": float(weighted_probabilities.min()),
        "probability_sum": float(weighted_probabilities.sum()),
        "completeness_error": completeness_error(frames),
        "Fisher_reconstruction_error": float(
            np.linalg.norm(reconstructed - evaluation.fisher, ord="fro")
        ),
        "maximum_QFI_normalized_eigenvalue": float(
            np.linalg.eigvalsh(normalized).max()
        ),
        "final_gradient_norm": gradient_norm(evaluation),
        "optimizer_records": records,
        "caveat": (
            "Best validated local optimum in the q=2 Schur-Parseval-frame "
            "class; not a global unrestricted-POVM certificate."
        ),
    }


def save_case(
    copies: int,
    blocks,
    frames: list[np.ndarray],
    evaluation: Evaluation,
    summary: dict[str, object],
) -> None:
    stem = f"n4_ns{copies}_phase_povm_q2"
    (DATA / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    arrays: dict[str, np.ndarray] = {
        "fisher": evaluation.fisher,
        "covariance": evaluation.covariance,
        "probabilities": np.concatenate(
            [
                block.multiplicity * p
                for block, p in zip(
                    blocks, evaluation.probabilities, strict=True
                )
            ]
        ),
        "derivatives": np.vstack(
            [
                block.multiplicity * d
                for block, d in zip(blocks, evaluation.derivatives, strict=True)
            ]
        ),
    }
    for block, frame in zip(blocks, frames, strict=True):
        arrays[f"frame_{block.label}"] = frame
        arrays[f"sigma_{block.label}"] = block.sigma
        arrays[f"tangents_{block.label}"] = block.tangents
    np.savez_compressed(DATA / f"{stem}.npz", **arrays)


def load_pvm_summary(copies: int) -> tuple[np.ndarray, dict[str, object]]:
    z = np.load(DATA / f"n4_ns{copies}_phase_pvm.npz")
    fisher = z["fisher"]
    covariance = np.linalg.inv(fisher)
    edge_gains = 24 / np.diag(covariance)
    return fisher, {
        "risk": float(np.trace(covariance)),
        "A_risk_gain": float(144 / np.trace(covariance)),
        "edge_gains": edge_gains,
    }


def make_figure(results: dict[int, dict[str, object]]) -> None:
    pvm_fisher, _ = load_pvm_summary(2)
    pvm_edge = 24 / np.diag(np.linalg.inv(pvm_fisher))
    qfi = quantum_fisher_matrix(
        compound_symmetric_state(4, 0.5),
        phase_tangents(compound_symmetric_state(4, 0.5))[1],
    )
    asymptotic_edge = 24 / np.diag(np.linalg.inv(qfi))
    series = [
        pvm_edge,
        np.asarray(results[2]["edge_directional_Fisher_gains"]),
        np.asarray(results[3]["edge_directional_Fisher_gains"]),
        asymptotic_edge,
    ]
    labels = [
        r"$n_s=2$ PVM",
        r"$n_s=2$ POVM",
        r"$n_s=3$ POVM",
        "Asymptotic limit",
    ]
    edge_labels = ["12", "13", "14", "23", "24", "34"]
    colors = plt.cm.tab10(np.arange(6))
    fig, ax = plt.subplots(figsize=(3.4, 2.65))
    x = np.arange(len(series))
    jitter = np.linspace(-0.17, 0.17, 6)
    for edge, (name, color, shift) in enumerate(
        zip(edge_labels, colors, jitter, strict=True)
    ):
        ax.scatter(
            x + shift,
            [values[edge] for values in series],
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=name,
            zorder=3,
        )
    for index, values in enumerate(series):
        ax.vlines(index, values.min(), values.max(), color="0.35", lw=0.8, zorder=1)
        ax.scatter(
            index,
            values.mean(),
            marker="D",
            s=30,
            facecolor="black",
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
    ax.axhline(1, color="0.45", ls="--", lw=0.8)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(r"$F_e/F_e^{\rm rep}$")
    ax.set_ylim(0.9, max(3.2, 1.08 * max(v.max() for v in series)))
    ax.grid(axis="y", color="0.88", lw=0.6)
    ax.legend(
        title="edge",
        ncol=6,
        loc="upper left",
        fontsize=6.6,
        title_fontsize=6.6,
        handletextpad=0.25,
        columnspacing=0.55,
        borderpad=0.25,
    )
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURES / "n4_finite_receiver_edge_gains.pdf")
    fig.savefig(FIGURES / "n4_finite_receiver_edge_gains.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redundancy", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps-ns2", type=int, default=3000)
    parser.add_argument("--steps-ns3", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--perturbation", type=float, default=0.035)
    parser.add_argument(
        "--refine-existing",
        action="store_true",
        help="continue from the saved q=2 POVMs instead of restarting",
    )
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    summaries: dict[int, dict[str, object]] = {}
    for copies, steps in ((2, args.steps_ns2), (3, args.steps_ns3)):
        if args.refine_existing:
            blocks, frames, evaluation, records = refine_saved_case(
                copies, steps=steps
            )
        else:
            blocks, frames, evaluation, records = optimize_case(
                copies,
                redundancy=args.redundancy,
                restarts=args.restarts,
                steps=steps,
                seed=args.seed + copies,
                perturbation=args.perturbation,
            )
        summary = fisher_summary(copies, blocks, frames, evaluation, records)
        summaries[copies] = summary
        save_case(copies, blocks, frames, evaluation, summary)
        print(json.dumps(summary, indent=2), flush=True)
    make_figure(summaries)


if __name__ == "__main__":
    main()
