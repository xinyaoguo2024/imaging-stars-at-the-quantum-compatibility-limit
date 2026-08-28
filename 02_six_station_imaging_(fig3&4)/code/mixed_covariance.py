"""Numerical helpers for joint amplitude--closure Gaussian likelihoods.

The residual ordering is fixed to

    [amplitude edge 0, ..., amplitude edge E-1,
     closure 0, ..., closure C-1].

The supplied covariance is the measurement covariance before the historical
RML closure-phase floor is applied.  ``phase_floor`` is therefore added only
to the lower-right closure block.  Amplitudes remain in the full edge space.
"""

from __future__ import annotations

import numpy as np


def _validated_inputs(
    residual: np.ndarray,
    covariance: np.ndarray,
    n_amplitude: int,
    phase_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    residual_array = np.asarray(residual, dtype=float)
    if residual_array.ndim != 2:
        raise ValueError(
            "mixed residual must have shape (n_sample, E+C); "
            f"got {residual_array.shape}"
        )
    n_sample, n_coord = residual_array.shape
    n_amplitude = int(n_amplitude)
    if not 0 < n_amplitude < n_coord:
        raise ValueError(
            f"n_amplitude must lie in [1, E+C-1]; got {n_amplitude} "
            f"for E+C={n_coord}"
        )
    covariance_array = np.asarray(covariance, dtype=float)
    expected = (n_sample, n_coord, n_coord)
    if covariance_array.shape != expected:
        raise ValueError(
            "mixed covariance must have shape (n_sample, E+C, E+C); "
            f"expected {expected}, got {covariance_array.shape}"
        )
    if not np.all(np.isfinite(residual_array)):
        raise ValueError("mixed residual contains non-finite entries")
    if not np.all(np.isfinite(covariance_array)):
        raise ValueError("mixed covariance contains non-finite entries")
    if not np.isfinite(phase_floor) or phase_floor < 0.0:
        raise ValueError(f"phase_floor must be finite and nonnegative; got {phase_floor}")

    covariance_array = 0.5 * (
        covariance_array + np.swapaxes(covariance_array, 1, 2)
    )
    covariance_array = covariance_array.copy()
    n_closure = n_coord - n_amplitude
    covariance_array[:, n_amplitude:, n_amplitude:] += (
        float(phase_floor) ** 2
    ) * np.eye(n_closure)[None, :, :]
    return residual_array, covariance_array


def _batched_stable_cholesky(
    covariance: np.ndarray,
    *,
    eigenvalue_floor: float = 1.0e-24,
) -> np.ndarray:
    """Return lower Cholesky factors, regularizing only non-positive inputs."""
    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be finite and strictly positive")
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # A covariance generated from a rank-deficient Fisher matrix may be
        # positive semidefinite rather than definite.  Add the smallest
        # per-sample diagonal shift needed for a stable Cholesky factorization.
        minimum = np.min(np.linalg.eigvalsh(covariance), axis=1)
        shift = np.maximum(float(eigenvalue_floor) - minimum, 0.0)
        scale = np.maximum(
            np.max(np.abs(np.diagonal(covariance, axis1=1, axis2=2)), axis=1),
            1.0,
        )
        shift = np.maximum(shift, np.finfo(float).eps * scale)
        regularized = covariance + shift[:, None, None] * np.eye(covariance.shape[1])[None, :, :]
        return np.linalg.cholesky(regularized)


def mixed_covariance_cholesky(
    covariance: np.ndarray,
    *,
    n_amplitude: int,
    phase_floor: float = 0.0,
) -> np.ndarray:
    """Validate and factor a mixed covariance for reuse across RML iterations."""
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.ndim != 3 or covariance_array.shape[1] != covariance_array.shape[2]:
        raise ValueError(
            "mixed covariance must have shape (n_sample, E+C, E+C); "
            f"got {covariance_array.shape}"
        )
    residual_shape = covariance_array.shape[:2]
    _dummy_residual = np.zeros(residual_shape, dtype=float)
    _dummy_residual, covariance_array = _validated_inputs(
        _dummy_residual,
        covariance_array,
        n_amplitude,
        phase_floor,
    )
    return _batched_stable_cholesky(covariance_array)


def mixed_covariance_terms(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    n_amplitude: int,
    phase_floor: float = 0.0,
    cholesky: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten mixed residuals and apply the full mixed precision matrix.

    Returns ``(whitened, precision_residual)``.  The latter is
    ``covariance_with_phase_floor^{-1} @ residual`` and retains both
    amplitude--closure cross blocks.
    """
    if cholesky is None:
        residual_array, covariance_array = _validated_inputs(
            residual,
            covariance,
            n_amplitude,
            phase_floor,
        )
        cholesky_array = _batched_stable_cholesky(covariance_array)
    else:
        residual_array = np.asarray(residual, dtype=float)
        if residual_array.ndim != 2 or not np.all(np.isfinite(residual_array)):
            raise ValueError(
                "mixed residual must be a finite array with shape (n_sample, E+C); "
                f"got {residual_array.shape}"
            )
        cholesky_array = np.asarray(cholesky, dtype=float)
        expected = (residual_array.shape[0], residual_array.shape[1], residual_array.shape[1])
        if cholesky_array.shape != expected:
            raise ValueError(
                f"cached mixed Cholesky must have shape {expected}; got {cholesky_array.shape}"
            )
        if not np.all(np.isfinite(cholesky_array)) or np.any(
            np.diagonal(cholesky_array, axis1=1, axis2=2) <= 0.0
        ):
            raise ValueError("cached mixed Cholesky is not finite positive triangular data")
    whitened = np.linalg.solve(cholesky_array, residual_array[..., None])[..., 0]
    precision_residual = np.linalg.solve(
        np.swapaxes(cholesky_array, 1, 2),
        whitened[..., None],
    )[..., 0]
    return whitened, precision_residual


def mixed_quadratic_objective_and_gradient(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    n_amplitude: int,
    phase_floor: float = 0.0,
    cholesky: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the reduced Gaussian quadratic and its residual gradient.

    The objective is

        0.5 / (n_sample * (E+C)) * sum_s r_s^T Sigma_s^{-1} r_s.

    Its exact derivative with respect to the residual array has the same shape
    as ``residual``.  The final return value is the Cholesky-whitened residual,
    useful for reduced-chi-square diagnostics.
    """
    whitened, precision_residual = mixed_covariance_terms(
        residual,
        covariance,
        n_amplitude=n_amplitude,
        phase_floor=phase_floor,
        cholesky=cholesky,
    )
    normalization = float(np.asarray(residual).size)
    objective = 0.5 * float(np.sum(whitened**2)) / normalization
    gradient = precision_residual / normalization
    return objective, gradient, whitened
