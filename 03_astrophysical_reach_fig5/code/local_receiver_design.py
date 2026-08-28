"""Local mixed-space POVM design used by the revised Fig. 2 simulations.

The raw optical model has E edge-amplitude and E edge-phase tangents.  The
science space is full-edge amplitude plus closure phase; station-piston phase
directions are nuisance parameters.  A locally optimized single-copy phase
POVM is mixed with a common full-edge amplitude branch.  The coherent panel is
then obtained from the *same effects* through score-operator compression.

The finite-depth ``n_s=10`` panel uses the QLAN covariance proxy.  This module
does not claim to construct an exact ten-copy POVM.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class OutcomeModel:
    probabilities: Array
    derivatives: Array
    scores: Array
    fisher: Array


@dataclass(frozen=True)
class LiftModel:
    fisher: Array
    covariance_a: Array
    commutator_b: Array
    covariance_v: Array
    promoted_fisher: Array
    score_operators: Array
    kappa_operator: float


def hermitize(matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=complex)
    return 0.5 * (matrix + matrix.conj().T)


def real_symmetrize(matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def psd_pinv(matrix: Array, *, rcond: float = 1.0e-11) -> Array:
    matrix = real_symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    # ``rcond`` is a relative spectral cutoff.  Using an absolute floor of one
    # here would incorrectly declare a perfectly conditioned matrix such as
    # 1e-14 * I to be singular.  That situation occurs naturally after a
    # visibility becomes small or a parameter is expressed in rescaled units.
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    inv = np.zeros_like(values)
    good = values > rcond * scale if scale > 0.0 else np.zeros_like(values, dtype=bool)
    inv[good] = 1.0 / values[good]
    return real_symmetrize((vectors * inv) @ vectors.T)


def psd_power(matrix: Array, power: float, *, rcond: float = 1.0e-11) -> Array:
    matrix = real_symmetrize(matrix)
    values, vectors = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    out = np.zeros_like(values)
    good = values > rcond * scale if scale > 0.0 else np.zeros_like(values, dtype=bool)
    out[good] = values[good] ** power
    return real_symmetrize((vectors * out) @ vectors.T)


def effects_to_scores(
    rho: Array,
    derivatives: Iterable[Array],
    effects: Iterable[Array],
    *,
    probability_floor: float = 1.0e-15,
) -> OutcomeModel:
    """Evaluate probabilities, local scores, and CFI for explicit effects."""
    rho = hermitize(rho)
    derivative_array = np.asarray([hermitize(item) for item in derivatives], dtype=complex)
    effect_array = np.asarray([hermitize(item) for item in effects], dtype=complex)
    probabilities = np.einsum("ij,xji->x", rho, effect_array).real
    if float(np.min(probabilities)) < -1.0e-10:
        raise ValueError(f"POVM produced a negative probability: {float(np.min(probabilities)):.3e}")
    probabilities = np.maximum(probabilities, 0.0)
    dp = np.einsum("aij,xji->xa", derivative_array, effect_array).real
    scores = np.zeros_like(dp)
    good = probabilities > probability_floor
    scores[good] = dp[good] / probabilities[good, None]
    fisher = (scores.T * probabilities) @ scores
    return OutcomeModel(
        probabilities=probabilities,
        derivatives=dp,
        scores=scores,
        fisher=real_symmetrize(fisher),
    )


def efficient_scores(
    outcome: OutcomeModel,
    science_indices: Array,
    nuisance_indices: Array,
) -> tuple[Array, Array]:
    """Return nuisance-efficient scores and their Schur-complement CFI."""
    science_indices = np.asarray(science_indices, dtype=int)
    nuisance_indices = np.asarray(nuisance_indices, dtype=int)
    science = outcome.scores[:, science_indices]
    if nuisance_indices.size == 0:
        efficient = science
    else:
        nuisance = outcome.scores[:, nuisance_indices]
        j_sn = outcome.fisher[np.ix_(science_indices, nuisance_indices)]
        j_nn = outcome.fisher[np.ix_(nuisance_indices, nuisance_indices)]
        coefficient = j_sn @ psd_pinv(j_nn)
        efficient = science - nuisance @ coefficient.T
    fisher = (efficient.T * outcome.probabilities) @ efficient
    return efficient, real_symmetrize(fisher)


def score_compression_lift(
    rho: Array,
    effects: Iterable[Array],
    probabilities: Array,
    efficient: Array,
    fisher: Array,
) -> LiftModel:
    """Construct A, B, and the canonical coherent-score QLAN covariance."""
    effect_array = np.asarray([hermitize(item) for item in effects], dtype=complex)
    fisher = real_symmetrize(fisher)
    fisher_inverse = psd_pinv(fisher)
    influences = efficient @ fisher_inverse
    score_operators = np.einsum("xa,xij->aij", influences, effect_array)
    rho = hermitize(rho)
    z_matrix = np.einsum(
        "ij,ajk,bki->ab",
        rho,
        score_operators,
        score_operators,
        optimize=True,
    )
    a_matrix = real_symmetrize(z_matrix.real)
    b_matrix = 0.5 * (z_matrix.imag - z_matrix.imag.T)
    sqrt_a = psd_power(a_matrix, 0.5)
    invsqrt_a = psd_power(a_matrix, -0.5)
    k_matrix = invsqrt_a @ (1j * b_matrix) @ invsqrt_a
    k_matrix = hermitize(k_matrix)
    k_values, k_vectors = np.linalg.eigh(k_matrix)
    abs_k = (k_vectors * np.abs(k_values)) @ k_vectors.conj().T
    correction = sqrt_a @ abs_k @ sqrt_a
    covariance_v = real_symmetrize(a_matrix + correction.real)
    return LiftModel(
        fisher=fisher,
        covariance_a=a_matrix,
        commutator_b=b_matrix,
        covariance_v=covariance_v,
        promoted_fisher=psd_pinv(covariance_v),
        score_operators=score_operators,
        kappa_operator=float(np.max(np.abs(k_values))) if len(k_values) else 0.0,
    )


def sld_operators(rho: Array, derivatives: Iterable[Array], *, floor: float = 1.0e-13) -> tuple[Array, Array]:
    """Return SLD operators and their real Gram/QFI matrix."""
    rho = hermitize(rho)
    values, vectors = np.linalg.eigh(rho)
    values = np.maximum(values, floor)
    denom = values[:, None] + values[None, :]
    operators = []
    for derivative in derivatives:
        transformed = vectors.conj().T @ hermitize(derivative) @ vectors
        local = np.zeros_like(transformed, dtype=complex)
        mask = denom > floor
        local[mask] = 2.0 * transformed[mask] / denom[mask]
        operators.append(hermitize(vectors @ local @ vectors.conj().T))
    operators_array = np.asarray(operators)
    gram = np.einsum("ij,ajk,bki->ab", rho, operators_array, operators_array, optimize=True)
    return operators_array, real_symmetrize(gram.real)


def incidence_basis(edges: list[tuple[int, int]], n_station: int) -> Array:
    incidence = np.zeros((len(edges), n_station), dtype=float)
    for edge_index, (i, j) in enumerate(edges):
        incidence[edge_index, i] = 1.0
        incidence[edge_index, j] = -1.0
    q_matrix, _ = np.linalg.qr(incidence[:, :-1], mode="reduced")
    return q_matrix


def mixed_parameter_map(closure_basis: Array, piston_basis: Array) -> Array:
    """Map (full amplitudes, closures, pistons) into raw (amplitudes, phases)."""
    closure_basis = np.asarray(closure_basis, dtype=float)
    piston_basis = np.asarray(piston_basis, dtype=float)
    n_edge, n_closure = closure_basis.shape
    n_piston = piston_basis.shape[1]
    transform = np.zeros((2 * n_edge, n_edge + n_closure + n_piston), dtype=float)
    transform[:n_edge, :n_edge] = np.eye(n_edge)
    transform[n_edge:, n_edge : n_edge + n_closure] = closure_basis
    transform[n_edge:, n_edge + n_closure :] = piston_basis
    return transform


def conditional_state_and_raw_derivatives(
    visibilities: Array,
    station_efficiencies: Array,
    station_occupations: Array,
    station_noise: Array,
    edges: list[tuple[int, int]],
) -> tuple[Array, Array, float]:
    """Build rho and derivatives ordered as (E amplitudes, E phases)."""
    visibilities = np.asarray(visibilities, dtype=complex)
    eta = np.asarray(station_efficiencies, dtype=float)
    occupations = np.asarray(station_occupations, dtype=float)
    noise = np.asarray(station_noise, dtype=float)
    n_station = len(eta)
    loads = eta * occupations + noise
    b_matrix = np.diag(loads).astype(complex)
    scales = np.zeros(len(edges), dtype=float)
    coherences = np.zeros(len(edges), dtype=complex)
    for edge_index, (i, j) in enumerate(edges):
        scale = float(np.sqrt(eta[i] * eta[j] * occupations[i] * occupations[j]))
        coherence = scale * visibilities[edge_index]
        scales[edge_index] = scale
        coherences[edge_index] = coherence
        b_matrix[i, j] = coherence
        b_matrix[j, i] = np.conj(coherence)
    total_occupation = float(np.trace(b_matrix).real)
    if total_occupation <= 0.0:
        raise ValueError("non-positive one-photon occupation")
    rho = hermitize(b_matrix / total_occupation)
    min_value = float(np.min(np.linalg.eigvalsh(rho)))
    if min_value < -2.0e-9:
        raise ValueError(f"conditional state is not PSD: lambda_min={min_value:.3e}")
    derivatives: list[Array] = []
    for edge_index, (i, j) in enumerate(edges):
        derivative = np.zeros((n_station, n_station), dtype=complex)
        visibility = visibilities[edge_index]
        magnitude = abs(visibility)
        phase_factor = visibility / magnitude if magnitude > 1.0e-14 else 1.0 + 0.0j
        value = scales[edge_index] * phase_factor / total_occupation
        derivative[i, j] = value
        derivative[j, i] = np.conj(value)
        derivatives.append(derivative)
    for edge_index, (i, j) in enumerate(edges):
        derivative = np.zeros((n_station, n_station), dtype=complex)
        value = 1j * coherences[edge_index] / total_occupation
        derivative[i, j] = value
        derivative[j, i] = np.conj(value)
        derivatives.append(derivative)
    return rho, np.asarray(derivatives), total_occupation


def pairwise_quadrature_effects(
    visibilities: Array,
    edges: list[tuple[int, int]],
    n_station: int,
    *,
    quadrature: str,
) -> Array:
    """Complete edge-first setting for amplitude or phase quadrature."""
    effects: list[Array] = []
    for visibility, (i, j) in zip(visibilities, edges):
        phase = float(np.angle(visibility))
        chi = phase if quadrature == "amplitude" else phase + 0.5 * np.pi
        for sign in (-1.0, 1.0):
            ket = np.zeros(n_station, dtype=complex)
            ket[i] = 1.0 / np.sqrt(2.0)
            ket[j] = sign * np.exp(-1j * chi) / np.sqrt(2.0)
            effects.append(np.outer(ket, ket.conj()) / (n_station - 1.0))
    return np.asarray(effects)


def projective_effects(observable: Array) -> Array:
    _values, vectors = np.linalg.eigh(hermitize(observable))
    return np.asarray([np.outer(vectors[:, k], vectors[:, k].conj()) for k in range(vectors.shape[1])])


def covariant_dark_effects(rho: Array) -> Array:
    """Finite dark-pair design; exact minimax at the balanced all-g point."""
    values, vectors = np.linalg.eigh(hermitize(rho))
    bright = vectors[:, int(np.argmax(values))]
    phase_index = int(np.argmax(np.abs(bright)))
    bright = bright * np.exp(-1j * np.angle(bright[phase_index]))
    trial = np.column_stack([bright, np.eye(len(bright), dtype=complex)])
    unitary, _ = np.linalg.qr(trial)
    if abs(np.vdot(unitary[:, 0], bright)) < 1.0 - 1.0e-8:
        raise RuntimeError("failed to construct bright/dark basis")
    dark = unitary[:, 1:]
    d_dark = dark.shape[1]
    effects: list[Array] = [np.outer(bright, bright.conj())]
    if d_dark == 1:
        effects.append(np.outer(dark[:, 0], dark[:, 0].conj()))
        return np.asarray(effects)
    for r in range(d_dark):
        for s in range(d_dark):
            if r == s:
                continue
            ket = (dark[:, r] + 1j * dark[:, s]) / np.sqrt(2.0)
            effects.append(np.outer(ket, ket.conj()) / (d_dark - 1.0))
    return np.asarray(effects)


def _schur_target(matrix: Array, target: Array, nuisance: Array) -> Array:
    matrix = real_symmetrize(matrix)
    target = np.asarray(target, dtype=int)
    nuisance = np.asarray(nuisance, dtype=int)
    if nuisance.size == 0:
        return matrix[np.ix_(target, target)]
    m_tt = matrix[np.ix_(target, target)]
    m_tn = matrix[np.ix_(target, nuisance)]
    m_nn = matrix[np.ix_(nuisance, nuisance)]
    return real_symmetrize(m_tt - m_tn @ psd_pinv(m_nn) @ m_tn.T)


def _schur_derivative(matrix: Array, derivative: Array, target: Array, nuisance: Array) -> Array:
    target = np.asarray(target, dtype=int)
    nuisance = np.asarray(nuisance, dtype=int)
    d_tt = derivative[np.ix_(target, target)]
    if nuisance.size == 0:
        return real_symmetrize(d_tt)
    m_tn = matrix[np.ix_(target, nuisance)]
    m_nn = matrix[np.ix_(nuisance, nuisance)]
    inverse = psd_pinv(m_nn)
    d_tn = derivative[np.ix_(target, nuisance)]
    d_nn = derivative[np.ix_(nuisance, nuisance)]
    out = d_tt - d_tn @ inverse @ m_tn.T - m_tn @ inverse @ d_tn.T
    out += m_tn @ inverse @ d_nn @ inverse @ m_tn.T
    return real_symmetrize(out)


def optimize_phase_setting_weights(
    j_amplitude: Array,
    j_settings: Array,
    h_quantum: Array,
    target: Array,
    nuisance: Array,
    *,
    amplitude_fraction: float,
    phase_fraction: float,
    n_iteration: int = 700,
) -> tuple[Array, dict[str, float]]:
    """Mirror-ascent maximin design in the QFI-whitened closure space."""
    n_setting = len(j_settings)
    log_weights = np.zeros(n_setting, dtype=float)
    h_effective = _schur_target(h_quantum, target, nuisance)
    h_inv_sqrt = psd_power(h_effective, -0.5)
    best_value = -np.inf
    best_weights = np.full(n_setting, 1.0 / n_setting)
    last_gap = np.inf
    for iteration in range(n_iteration):
        shifted = log_weights - float(np.max(log_weights))
        weights = np.exp(np.clip(shifted, -60.0, 0.0))
        weights /= float(np.sum(weights))
        matrix = amplitude_fraction * j_amplitude + phase_fraction * np.einsum(
            "k,kij->ij", weights, j_settings
        )
        effective = _schur_target(matrix, target, nuisance)
        normalized = real_symmetrize(h_inv_sqrt @ effective @ h_inv_sqrt)
        values, vectors = np.linalg.eigh(normalized)
        value = float(values[0])
        if value > best_value:
            best_value = value
            best_weights = weights.copy()
        temperature = max(0.0025, 0.060 * (0.992 ** iteration))
        soft = np.exp(-(values - values[0]) / temperature)
        soft /= float(np.sum(soft))
        gradient_matrix = (vectors * soft) @ vectors.T
        gradients = np.zeros(n_setting, dtype=float)
        for k, setting in enumerate(j_settings):
            derivative = phase_fraction * setting
            d_effective = _schur_derivative(matrix, derivative, target, nuisance)
            d_normalized = h_inv_sqrt @ d_effective @ h_inv_sqrt
            gradients[k] = float(np.trace(gradient_matrix @ d_normalized))
        centered = gradients - float(np.dot(weights, gradients))
        scale = max(float(np.max(np.abs(centered))), 1.0e-12)
        step = 0.85 / np.sqrt(iteration + 1.0)
        log_weights += step * centered / scale
        last_gap = float(np.max(gradients) - np.dot(weights, gradients))
    keep = best_weights > 1.0e-7
    best_weights = best_weights * keep
    best_weights /= float(np.sum(best_weights))
    return best_weights, {
        "maximin_qfi_efficiency": float(best_value),
        "mirror_kkt_gap": float(last_gap),
        "n_dictionary_settings": int(n_setting),
        "n_active_settings": int(np.sum(keep)),
    }


def effects_hash(effects: Iterable[Array]) -> str:
    digest = hashlib.sha256()
    for effect in effects:
        array = np.asarray(effect, dtype=np.complex128)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array.real).tobytes())
        digest.update(np.ascontiguousarray(array.imag).tobytes())
    return digest.hexdigest()


def _completeness_error(effects: Iterable[Array], n_station: int) -> float:
    total = np.sum(np.asarray(list(effects)), axis=0)
    return float(np.linalg.norm(total - np.eye(n_station), ord=2))


def locally_optimized_effects(
    rho: Array,
    derivatives_raw: Array,
    transform: Array,
    visibilities: Array,
    edges: list[tuple[int, int]],
    *,
    amplitude_fraction: float,
    phase_fraction: float,
    random_seed: int = 73129,
    n_iteration: int = 450,
) -> tuple[Array, dict[str, object]]:
    """Build the score-dictionary maximin single-copy POVM."""
    n_station = rho.shape[0]
    n_edge = len(edges)
    n_closure = transform.shape[1] - n_edge - (n_station - 1)
    derivatives_new = np.einsum("ip,ijk->pjk", transform, derivatives_raw, optimize=True)
    slds, h_quantum = sld_operators(rho, derivatives_new)
    closure = np.arange(n_edge, n_edge + n_closure)
    nuisance = np.concatenate(
        [np.arange(n_edge), np.arange(n_edge + n_closure, transform.shape[1])]
    )
    h_cc = _schur_target(h_quantum, closure, nuisance)
    h_cn = h_quantum[np.ix_(closure, nuisance)]
    h_nn = h_quantum[np.ix_(nuisance, nuisance)]
    efficient_slds = slds[closure] - np.einsum(
        "ab,bij->aij", h_cn @ psd_pinv(h_nn), slds[nuisance], optimize=True
    )
    h_inv_sqrt = psd_power(h_cc, -0.5)
    directions = [np.eye(n_closure)[k] for k in range(n_closure)]
    rng = np.random.default_rng(random_seed)
    for _ in range(max(3 * n_closure, 30)):
        direction = rng.normal(size=n_closure)
        directions.append(direction / np.linalg.norm(direction))
    setting_effects: list[Array] = []
    setting_names: list[str] = []
    for index, direction in enumerate(directions):
        coefficient = h_inv_sqrt @ direction
        observable = np.einsum("a,aij->ij", coefficient, efficient_slds, optimize=True)
        setting_effects.append(projective_effects(observable))
        setting_names.append(f"efficient_sld_pvm_{index:03d}")
    setting_effects.append(covariant_dark_effects(rho))
    setting_names.append("finite_covariant_dark_design")
    setting_effects.append(pairwise_quadrature_effects(visibilities, edges, n_station, quadrature="phase"))
    setting_names.append("uniform_edge_phase")

    amplitude_effects = pairwise_quadrature_effects(
        visibilities, edges, n_station, quadrature="amplitude"
    )
    raw_amp = effects_to_scores(rho, derivatives_raw, amplitude_effects).fisher
    raw_settings = np.asarray(
        [effects_to_scores(rho, derivatives_raw, setting).fisher for setting in setting_effects]
    )
    j_amp = transform.T @ raw_amp @ transform
    j_settings = np.asarray([transform.T @ matrix @ transform for matrix in raw_settings])
    weights, diagnostics = optimize_phase_setting_weights(
        j_amp,
        j_settings,
        h_quantum,
        closure,
        nuisance,
        amplitude_fraction=amplitude_fraction,
        phase_fraction=phase_fraction,
        n_iteration=n_iteration,
    )
    effects: list[Array] = [amplitude_fraction * effect for effect in amplitude_effects]
    active_names = []
    active_weights = []
    for weight, name, setting in zip(weights, setting_names, setting_effects):
        if weight <= 0.0:
            continue
        active_names.append(name)
        active_weights.append(float(weight))
        effects.extend(phase_fraction * float(weight) * effect for effect in setting)
    effect_array = np.asarray(effects)
    diagnostics.update(
        {
            "phase_setting_names": active_names,
            "phase_setting_weights": active_weights,
            "completeness_error": _completeness_error(effect_array, n_station),
            "povm_hash": effects_hash(effect_array),
            "objective": "closure-QFI-whitened maximin after full-amplitude and piston nuisance",
        }
    )
    return effect_array, diagnostics


def evaluate_effects_in_mixed_space(
    rho: Array,
    derivatives_raw: Array,
    transform: Array,
    effects: Array,
    *,
    n_edge: int,
    n_closure: int,
) -> tuple[OutcomeModel, Array, LiftModel]:
    derivatives_new = np.einsum("ip,ijk->pjk", transform, derivatives_raw, optimize=True)
    outcome = effects_to_scores(rho, derivatives_new, effects)
    science = np.arange(n_edge + n_closure)
    nuisance = np.arange(n_edge + n_closure, transform.shape[1])
    efficient, fisher = efficient_scores(outcome, science, nuisance)
    lift = score_compression_lift(
        rho,
        effects,
        outcome.probabilities,
        efficient,
        fisher,
    )
    return outcome, fisher, lift


def edge_first_effects(
    visibilities: Array,
    edges: list[tuple[int, int]],
    n_station: int,
    *,
    amplitude_fraction: float,
    phase_fraction: float,
) -> Array:
    amplitude = pairwise_quadrature_effects(visibilities, edges, n_station, quadrature="amplitude")
    phase = pairwise_quadrature_effects(visibilities, edges, n_station, quadrature="phase")
    return np.concatenate([amplitude_fraction * amplitude, phase_fraction * phase], axis=0)


def closure_effective_fisher(fisher_mixed: Array, n_edge: int, n_closure: int) -> Array:
    closure = np.arange(n_edge, n_edge + n_closure)
    amplitude = np.arange(n_edge)
    return _schur_target(fisher_mixed, closure, amplitude)
