#!/usr/bin/env python3
"""Fixed-|g| physical random-phase audit for the Fig. 5 closure loop.

Every off-diagonal coherence has magnitude exactly 0.5.  Residual edge phases
are drawn independently from a symmetric interval and the draw is retained
only when the resulting coherence matrix is positive semidefinite.  A random
station-gauge phase is then applied; it broadens individual edge phases without
changing closure phases or Fisher information.

The target S1-S2-S3 loop is fixed before sampling.  Since the prior is station-
exchangeable, this is a representative loop rather than a selected favorable
one.  The output used for Fig. 5 is the arithmetic mean Fisher gain of this
fixed loop at each magnitude.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import local_receiver_design as receiver  # noqa: E402
import make_fig5_promoted_uniform as fig5  # noqa: E402


N_VALUES = (6, 20)
TARGET_MAGNITUDE = 0.50
RESIDUAL_PHASE_HALF_WIDTH = 0.20
RNG_SEED = 20260716
DEFAULT_SAMPLES = 600
MAGNITUDE_GRID = np.arange(10.0, 22.0 + 1.0e-12, 1.0)


def graph_bases(
    edges: list[tuple[int, int]], n_station: int
) -> tuple[np.ndarray, np.ndarray]:
    incidence = np.zeros((len(edges), n_station), dtype=float)
    for edge_index, (i, j) in enumerate(edges):
        incidence[edge_index, i] = 1.0
        incidence[edge_index, j] = -1.0
    _left, singular_values, right = np.linalg.svd(
        incidence.T, full_matrices=True
    )
    rank = int(np.sum(singular_values > 1.0e-12))
    closure = right[rank:].T
    piston = receiver.incidence_basis(edges, n_station)
    return closure, piston


def draw_fixed_modulus_coherence(
    rng: np.random.Generator,
    n_station: int,
) -> tuple[np.ndarray, float, float, int]:
    """Return one PSD draw with exactly constant off-diagonal magnitude."""
    identity = np.eye(n_station)
    off_diagonal = 1.0 - identity
    attempts = 0
    while True:
        attempts += 1
        random_upper = rng.uniform(
            -RESIDUAL_PHASE_HALF_WIDTH,
            RESIDUAL_PHASE_HALF_WIDTH,
            size=(n_station, n_station),
        )
        residual_phase = np.triu(random_upper, 1)
        residual_phase -= residual_phase.T
        residual = (
            identity.astype(complex)
            + TARGET_MAGNITUDE
            * np.exp(1j * residual_phase)
            * off_diagonal
        )
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(residual)))
        if minimum_eigenvalue >= -1.0e-12:
            break

    # Station phases are gauge degrees of freedom.  Randomizing them makes the
    # individual edge phases cover the circle while preserving every closure.
    station_phase = rng.uniform(-np.pi, np.pi, size=n_station)
    gauge = np.exp(1j * station_phase)
    coherence = gauge[:, None] * residual * gauge.conj()[None, :]
    closure_phase = float(
        np.angle(
            coherence[0, 1]
            * coherence[1, 2]
            * np.conj(coherence[0, 2])
        )
    )
    upper = np.triu_indices(n_station, 1)
    maximum_magnitude_error = float(
        np.max(np.abs(np.abs(coherence[upper]) - TARGET_MAGNITUDE))
    )
    return coherence, closure_phase, minimum_eigenvalue, attempts


def functional_fisher(matrix: np.ndarray, direction: np.ndarray) -> float:
    return 1.0 / float(direction @ receiver.psd_pinv(matrix) @ direction)


def covariance_from_z(z_matrix: np.ndarray) -> np.ndarray:
    a_matrix = receiver.real_symmetrize(z_matrix.real)
    b_matrix = 0.5 * (z_matrix.imag - z_matrix.imag.T)
    sqrt_a = receiver.psd_power(a_matrix, 0.5)
    invsqrt_a = receiver.psd_power(a_matrix, -0.5)
    k_matrix = receiver.hermitize(
        invsqrt_a @ (1j * b_matrix) @ invsqrt_a
    )
    values, vectors = np.linalg.eigh(k_matrix)
    abs_k = (vectors * np.abs(values)) @ vectors.conj().T
    return receiver.real_symmetrize(
        a_matrix + (sqrt_a @ abs_k @ sqrt_a).real
    )


def prepare_geometry(n_station: int) -> dict[str, object]:
    edges = [
        (i, j) for i in range(n_station) for j in range(i + 1, n_station)
    ]
    n_edge = len(edges)
    closure, piston = graph_bases(edges, n_station)
    n_closure = closure.shape[1]
    phase_parameter_map = receiver.mixed_parameter_map(closure, piston)[
        :, n_edge:
    ]
    triangle = np.zeros(n_edge, dtype=float)
    triangle[edges.index((0, 1))] = 1.0
    triangle[edges.index((1, 2))] = 1.0
    triangle[edges.index((0, 2))] = -1.0
    closure_direction = closure.T @ triangle
    return {
        "N": n_station,
        "edges": edges,
        "E": n_edge,
        "C": n_closure,
        "phase_parameter_map": phase_parameter_map,
        "closure_direction": closure_direction,
    }


def precompute_source_draw(
    coherence: np.ndarray,
    geometry: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    n_station = int(geometry["N"])
    edges = geometry["edges"]
    phase_parameter_map = np.asarray(geometry["phase_parameter_map"])
    n_closure = int(geometry["C"])
    closure_direction = np.asarray(geometry["closure_direction"])
    visibilities = np.asarray([coherence[i, j] for i, j in edges])
    rho, raw_derivatives, _total = (
        receiver.conditional_state_and_raw_derivatives(
            visibilities,
            np.ones(n_station),
            np.ones(n_station),
            np.zeros(n_station),
            edges,
        )
    )
    phase_derivatives = np.einsum(
        "ip,ijk->pjk", phase_parameter_map, raw_derivatives, optimize=True
    )
    effects = receiver.pairwise_quadrature_effects(
        visibilities, edges, n_station, quadrature="phase"
    )
    outcome = receiver.effects_to_scores(rho, phase_derivatives, effects)
    efficient, fisher = receiver.efficient_scores(
        outcome,
        np.arange(n_closure),
        np.arange(n_closure, phase_parameter_map.shape[1]),
    )
    lift = receiver.score_compression_lift(
        rho, effects, outcome.probabilities, efficient, fisher
    )
    score_operators = lift.score_operators
    rho_incoherent = np.eye(n_station, dtype=complex) / n_station
    rho_source = coherence / n_station
    z_incoherent = np.einsum(
        "ij,ajk,bki->ab",
        rho_incoherent,
        score_operators,
        score_operators,
        optimize=True,
    )
    z_source = np.einsum(
        "ij,ajk,bki->ab",
        rho_source,
        score_operators,
        score_operators,
        optimize=True,
    )
    edge_fisher_q1 = functional_fisher(fisher, closure_direction)
    return (
        z_incoherent,
        z_source,
        edge_fisher_q1,
        float(np.linalg.norm(lift.commutator_b)),
        float(lift.kappa_operator),
    )


def gain_profile(
    z_incoherent: np.ndarray,
    z_source: np.ndarray,
    edge_fisher_q1: float,
    closure_direction: np.ndarray,
    source_fractions: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(source_fractions)
    for index, q_source in enumerate(source_fractions):
        z_matrix = (
            (1.0 - q_source) * z_incoherent + q_source * z_source
        ) / q_source**2
        covariance = covariance_from_z(z_matrix)
        promoted_fisher = 1.0 / float(
            closure_direction @ covariance @ closure_direction
        )
        edge_fisher = q_source**2 * edge_fisher_q1
        output[index] = promoted_fisher / edge_fisher
    return output


def direct_gain(
    coherence: np.ndarray,
    geometry: dict[str, object],
    magnitude: float,
) -> float:
    """Directly recompute the full detected model for shortcut validation."""
    n_station = int(geometry["N"])
    edges = geometry["edges"]
    phase_parameter_map = np.asarray(geometry["phase_parameter_map"])
    n_closure = int(geometry["C"])
    closure_direction = np.asarray(geometry["closure_direction"])
    occupation = float(fig5.ideal_occupation_from_ab_mag(magnitude))
    visibilities = np.asarray([coherence[i, j] for i, j in edges])
    rho, raw_derivatives, total_occupation = (
        receiver.conditional_state_and_raw_derivatives(
            visibilities,
            np.full(n_station, fig5.ETA),
            np.full(n_station, occupation),
            np.full(n_station, fig5.EPSILON_BG),
            edges,
        )
    )
    phase_derivatives = np.einsum(
        "ip,ijk->pjk", phase_parameter_map, raw_derivatives, optimize=True
    )
    effects = receiver.pairwise_quadrature_effects(
        visibilities, edges, n_station, quadrature="phase"
    )
    outcome = receiver.effects_to_scores(rho, phase_derivatives, effects)
    efficient, fisher = receiver.efficient_scores(
        outcome,
        np.arange(n_closure),
        np.arange(n_closure, phase_parameter_map.shape[1]),
    )
    lift = receiver.score_compression_lift(
        rho, effects, outcome.probabilities, efficient, fisher
    )
    scale = float(total_occupation)
    f_edge = scale * functional_fisher(fisher, closure_direction)
    f_promoted = scale * functional_fisher(
        lift.promoted_fisher, closure_direction
    )
    return f_promoted / f_edge


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_fisher_gain": float(np.mean(values)),
        "sem_fisher_gain": float(
            np.std(values, ddof=1) / np.sqrt(values.size)
        ),
        "median_fisher_gain": float(np.median(values)),
        "quantile_05_fisher_gain": float(np.quantile(values, 0.05)),
        "quantile_95_fisher_gain": float(np.quantile(values, 0.95)),
        "mean_snr_gain": float(np.mean(np.sqrt(values))),
        "snr_from_mean_fisher": float(np.sqrt(np.mean(values))),
    }


def plot_gain_diagnostics(
    gains: dict[int, np.ndarray],
    source_fractions: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    colors = {6: "#0077b6", 20: "#9d0208"}
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
        for n_station in N_VALUES:
            values = gains[n_station]
            mean = np.mean(values, axis=0)
            low = np.quantile(values, 0.05, axis=0)
            high = np.quantile(values, 0.95, axis=0)
            equal_real = (n_station - 1.0) / (
                1.0 - source_fractions * TARGET_MAGNITUDE
            )
            color = colors[n_station]
            axes[0].fill_between(
                MAGNITUDE_GRID, low, high, color=color, alpha=0.12
            )
            axes[0].plot(
                MAGNITUDE_GRID,
                mean,
                color=color,
                lw=1.7,
                label=rf"$N={n_station}$ phase-MC mean",
            )
            axes[0].plot(
                MAGNITUDE_GRID,
                equal_real,
                color=color,
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=rf"$N={n_station}$ equal-real",
            )
            axes[1].plot(
                MAGNITUDE_GRID,
                np.sqrt(mean),
                color=color,
                lw=1.7,
                label=rf"$N={n_station}$",
            )
        axes[0].set_xlabel("AB magnitude per station aperture")
        axes[0].set_ylabel(r"Fisher gain $F_{\rm coh}/F_{\rm edge}$")
        axes[0].set_title(r"Fixed typical loop $\Phi_{123}$")
        axes[0].legend(frameon=False, ncol=2)
        axes[1].set_xlabel("AB magnitude per station aperture")
        axes[1].set_ylabel(r"baseline/SNR gain $\sqrt{\langle F_{\rm coh}/F_{\rm edge}\rangle}$")
        axes[1].set_title("Quantity used to interpret Fig. 5")
        axes[1].legend(frameon=False)
        for label, ax in zip(("(a)", "(b)"), axes):
            ax.grid(axis="y", color="0.88", lw=0.5)
            ax.text(-0.16, 1.03, label, transform=ax.transAxes, fontweight="bold")
        fig.tight_layout(w_pad=2.1)
        fig.savefig(output_png, dpi=350, bbox_inches="tight")
        fig.savefig(output_pdf, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 10:
        raise ValueError("--samples must be at least ten")
    result_dir = Path(os.environ.get("FIG5_RESULT_DIR", str(ROOT / "data")))
    figure_dir = Path(
        os.environ.get("FIG5_MC_FIGURE_DIR", str(ROOT / "generated_outputs"))
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    terms = fig5.fisher_terms(MAGNITUDE_GRID)
    source_fractions = np.asarray(terms["a"] / terms["s0"], dtype=float)
    rng = np.random.default_rng(args.seed)
    gains: dict[int, np.ndarray] = {}
    closure_phases: dict[int, np.ndarray] = {}
    minimum_eigenvalues: dict[int, np.ndarray] = {}
    attempts: dict[int, np.ndarray] = {}
    commutator_norms: dict[int, np.ndarray] = {}
    kappa_operators: dict[int, np.ndarray] = {}
    edge_errors: dict[int, np.ndarray] = {}
    direct_shortcut_errors: list[float] = []
    max_magnitude_error = 0.0
    started = time.time()

    for n_station in N_VALUES:
        geometry = prepare_geometry(n_station)
        closure_direction = np.asarray(geometry["closure_direction"])
        array_gains = np.empty((args.samples, MAGNITUDE_GRID.size), dtype=float)
        array_phases = np.empty(args.samples, dtype=float)
        array_eigenvalues = np.empty(args.samples, dtype=float)
        array_attempts = np.empty(args.samples, dtype=int)
        array_commutators = np.empty(args.samples, dtype=float)
        array_kappa = np.empty(args.samples, dtype=float)
        array_edge_errors = np.empty(args.samples, dtype=float)
        reference_conditional_fisher = (
            2.0
            * TARGET_MAGNITUDE**2
            / (3.0 * n_station * (n_station - 1.0))
        )
        validation_draws: list[tuple[np.ndarray, np.ndarray]] = []
        for sample in range(args.samples):
            coherence, closure_phase, minimum_eigenvalue, n_attempt = (
                draw_fixed_modulus_coherence(rng, n_station)
            )
            upper = np.triu_indices(n_station, 1)
            max_magnitude_error = max(
                max_magnitude_error,
                float(
                    np.max(
                        np.abs(
                            np.abs(coherence[upper]) - TARGET_MAGNITUDE
                        )
                    )
                ),
            )
            (
                z_incoherent,
                z_source,
                edge_fisher_q1,
                commutator_norm,
                kappa_operator,
            ) = precompute_source_draw(coherence, geometry)
            profile = gain_profile(
                z_incoherent,
                z_source,
                edge_fisher_q1,
                closure_direction,
                source_fractions,
            )
            array_gains[sample] = profile
            array_phases[sample] = closure_phase
            array_eigenvalues[sample] = minimum_eigenvalue
            array_attempts[sample] = n_attempt
            array_commutators[sample] = commutator_norm
            array_kappa[sample] = kappa_operator
            array_edge_errors[sample] = abs(
                edge_fisher_q1 / reference_conditional_fisher - 1.0
            )
            if sample < 2:
                validation_draws.append((coherence.copy(), profile.copy()))
            if (sample + 1) % 25 == 0 or sample + 1 == args.samples:
                print(
                    f"N={n_station}: completed {sample + 1}/{args.samples} "
                    f"accepted draws in {time.time() - started:.1f} s",
                    flush=True,
                )

        for coherence, shortcut_profile in validation_draws:
            for magnitude_index in (0, 5, 10):
                direct = direct_gain(
                    coherence, geometry, MAGNITUDE_GRID[magnitude_index]
                )
                direct_shortcut_errors.append(
                    abs(direct / shortcut_profile[magnitude_index] - 1.0)
                )
        gains[n_station] = array_gains
        closure_phases[n_station] = array_phases
        minimum_eigenvalues[n_station] = array_eigenvalues
        attempts[n_station] = array_attempts
        commutator_norms[n_station] = array_commutators
        kappa_operators[n_station] = array_kappa
        edge_errors[n_station] = array_edge_errors

    gain_by_n = {}
    for n_station in N_VALUES:
        records = []
        equal_real = (n_station - 1.0) / (
            1.0 - source_fractions * TARGET_MAGNITUDE
        )
        for magnitude, q_source, values, real_gain in zip(
            MAGNITUDE_GRID,
            source_fractions,
            gains[n_station].T,
            equal_real,
        ):
            row = {
                "magnitude_ab": float(magnitude),
                "source_fraction_a_over_s0": float(q_source),
                **summarize(values),
                "equal_real_g_fisher_gain": float(real_gain),
            }
            records.append(row)
        gain_by_n[str(n_station)] = records

    elapsed = time.time() - started
    payload = {
        "schema_version": 2,
        "definition": (
            "Physical fixed-modulus random-phase Monte Carlo for one predeclared "
            "triangle closure under the phase-only uniform edge-first POVM and "
            "its coherent score-fluctuation lift."
        ),
        "ensemble": {
            "N_values": list(N_VALUES),
            "all_offdiagonal_magnitudes": TARGET_MAGNITUDE,
            "residual_edge_phase_prior": (
                f"iid Uniform[-{RESIDUAL_PHASE_HALF_WIDTH},"
                f"+{RESIDUAL_PHASE_HALF_WIDTH}] rad, conditioned on G >= 0"
            ),
            "station_gauge_phase_prior": "iid Uniform[-pi,+pi]",
            "physicality": "Hermitian PSD with unit diagonal in every accepted draw",
            "target_loop": "S1-S2-S3, fixed before sampling",
            "target_functional": "phi_12 + phi_23 - phi_13",
            "typical_loop_justification": (
                "The prior is invariant under station permutations, so every "
                "triangle loop has the same ensemble distribution."
            ),
        },
        "monte_carlo": {
            "accepted_samples_per_N": args.samples,
            "rng_seed": args.seed,
            "elapsed_seconds": elapsed,
            "averaging_for_fig5": (
                "Arithmetic mean Fisher gain of the fixed typical loop at each magnitude"
            ),
        },
        "validation": {
            "maximum_all_edge_magnitude_error": max_magnitude_error,
            "maximum_edge_fisher_relative_error": float(
                max(np.max(values) for values in edge_errors.values())
            ),
            "maximum_direct_vs_covariance_shortcut_relative_error": float(
                max(direct_shortcut_errors)
            ),
            "per_N": {
                str(n_station): {
                    "proposal_acceptance": float(
                        args.samples / np.sum(attempts[n_station])
                    ),
                    "minimum_coherence_eigenvalue": float(
                        np.min(minimum_eigenvalues[n_station])
                    ),
                    "median_coherence_eigenvalue": float(
                        np.median(minimum_eigenvalues[n_station])
                    ),
                    "closure_phase_mean": float(
                        np.mean(closure_phases[n_station])
                    ),
                    "closure_phase_std": float(
                        np.std(closure_phases[n_station], ddof=1)
                    ),
                    "closure_phase_05_95": [
                        float(np.quantile(closure_phases[n_station], 0.05)),
                        float(np.quantile(closure_phases[n_station], 0.95)),
                    ],
                    "mean_kappa_operator": float(
                        np.mean(kappa_operators[n_station])
                    ),
                    "mean_commutator_B_frobenius_norm": float(
                        np.mean(commutator_norms[n_station])
                    ),
                }
                for n_station in N_VALUES
            },
        },
        "gain_by_N_and_magnitude": gain_by_n,
    }
    output_json = result_dir / "fixed_modulus_phase_gain_summary.json"
    output_npz = result_dir / "fixed_modulus_phase_gain_samples.npz"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        output_npz,
        magnitude_grid=MAGNITUDE_GRID,
        source_fraction=source_fractions,
        gain_N6=gains[6],
        gain_N20=gains[20],
        closure_phase_N6=closure_phases[6],
        closure_phase_N20=closure_phases[20],
        minimum_eigenvalue_N6=minimum_eigenvalues[6],
        minimum_eigenvalue_N20=minimum_eigenvalues[20],
    )
    output_png = figure_dir / "fixed_modulus_phase_gain_N6_N20.png"
    output_pdf = figure_dir / "fixed_modulus_phase_gain_N6_N20.pdf"
    plot_gain_diagnostics(gains, source_fractions, output_png, output_pdf)
    print(output_json)
    print(output_npz)
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
