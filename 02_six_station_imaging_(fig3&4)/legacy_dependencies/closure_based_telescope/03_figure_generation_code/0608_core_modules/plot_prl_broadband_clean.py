from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

H_PLANCK = 6.62607015e-34
C_LIGHT = 299_792_458.0
FNU_AB0 = 3631e-26
UAS_TO_RAD = np.deg2rad(1.0 / 3600.0) * 1e-6

SOURCE_COMPONENT_FRACTIONS = {
    "compact_accretion_disc_continuum": 0.45,
    "diffuse_outer_disc_continuum": 0.10,
    "broad_line_region_lines": 0.40,
    "inner_optical_jet_knot": 0.05,
}

# Median-binned NASA/IPAC Extragalactic Database (NED) photometric SED for
# 3C 273 over the optical band used here.  The bins are 10 nm wide where NED
# has published/homogenized points in 400--800 nm; gaps are filled by log-log
# interpolation.  The data are not simultaneous, so this is a reproducible
# spectral-weight benchmark rather than a variability model.
SOURCE_SPECTRUM_NAME = "3C 273 NED median optical SED"
SOURCE_SPECTRUM_NOTE = (
    "NED homogenized photometry, 400-800 nm, 10 nm median bins; "
    "log-log interpolation between populated bins; non-simultaneous variable-source data."
)
SOURCE_SED_LAMBDA_NM = np.array(
    [
        405.0,
        425.0,
        445.0,
        475.0,
        485.0,
        505.0,
        535.0,
        545.0,
        555.0,
        565.0,
        595.0,
        625.0,
        635.0,
        645.0,
        665.0,
        675.0,
        685.0,
        695.0,
        705.0,
        755.0,
        775.0,
        795.0,
    ],
    dtype=float,
)
SOURCE_SED_FNU_JY = np.array(
    [
        0.01010,
        0.02775,
        0.02900,
        0.02750,
        0.02495,
        0.01850,
        0.02630,
        0.02805,
        0.02740,
        0.02720,
        0.02510,
        0.02930,
        0.02630,
        0.02935,
        0.02685,
        0.02450,
        0.02760,
        0.02665,
        0.02730,
        0.02630,
        0.03720,
        0.03650,
    ],
    dtype=float,
)
SOURCE_SED_NED_POINT_COUNT = 60


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """Small dependency-free Gaussian blur used only for display/deconvolution smoothing."""
    if sigma <= 0.15:
        return image.copy()
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    tmp = np.pad(image, ((0, 0), (radius, radius)), mode="edge")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, tmp)
    out = np.pad(tmp, ((radius, radius), (0, 0)), mode="edge")
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, out)
    return out


ARRAY_SCALE = float(os.environ.get("ARRAY_SCALE", "2.00"))
TELESCOPE_DIAMETER_M = 5.0
SOURCE_AB_MAG = 12.8
FIBER_LENGTH_SCALE = 0.75
OBSERVING_DAYS = int(os.environ.get("OBSERVING_DAYS", "30"))

STATIONS_KM = ARRAY_SCALE * np.array(
    [
        [0.0, 0.0],
        [0.16, 0.06],
        [0.48, -0.32],
        [1.15, 0.90],
        [2.60, -1.85],
        [5.20, 3.20],
        [10.20, -4.40],
        [18.90, 5.20],
    ],
    dtype=float,
)

DEFAULT_COMBINER_HUB_KM = np.array([3.0, 0.0], dtype=float)
if "HUB_X_KM" in os.environ and "HUB_Y_KM" in os.environ:
    COMBINER_HUB_KM = np.array([float(os.environ["HUB_X_KM"]), float(os.environ["HUB_Y_KM"])], dtype=float)
else:
    COMBINER_HUB_KM = DEFAULT_COMBINER_HUB_KM
FIBER_LOSS_DB_PER_KM = float(os.environ.get("FIBER_LOSS_DB_PER_KM", "0.15"))
MODE_FALSE_POSITIVE = float(
    os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("STATION_FALSE_POSITIVE", "0.05"))
)
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", "0.0"))


def station_photon_count_bin(
    mag_ab: float,
    lam_lo: float,
    lam_hi: float,
    *,
    diameter_m: float = 4.0,
    capture_efficiency: float = 0.30,
    exposure_s: float = 300.0,
) -> float:
    area = np.pi * (diameter_m / 2.0) ** 2
    return (
        area
        * capture_efficiency
        * exposure_s
        * FNU_AB0
        * 10.0 ** (-0.4 * mag_ab)
        * np.log(lam_hi / lam_lo)
        / H_PLANCK
    )


def mode_occupation_ab(
    mag_ab: float,
    freq_hz: float,
    *,
    diameter_m: float = 4.0,
    capture_efficiency: float = 0.30,
) -> float:
    area = np.pi * (diameter_m / 2.0) ** 2
    fnu = FNU_AB0 * 10.0 ** (-0.4 * mag_ab)
    return area * capture_efficiency * fnu / (H_PLANCK * freq_hz)


def source_fnu_jy(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Frequency-dependent 3C 273 flux-density model in Jy."""
    freq = np.asarray(freq_hz, dtype=float)
    lambda_nm = C_LIGHT / freq * 1e9
    # The SED anchors are in increasing wavelength; interpolate in log space to
    # preserve power-law behavior between sparse photometric points.
    log_fnu = np.interp(
        np.log(lambda_nm),
        np.log(SOURCE_SED_LAMBDA_NM),
        np.log(SOURCE_SED_FNU_JY),
        left=np.log(SOURCE_SED_FNU_JY[0]),
        right=np.log(SOURCE_SED_FNU_JY[-1]),
    )
    out = np.exp(log_fnu)
    if np.isscalar(freq_hz):
        return float(out)
    return out


def source_ab_magnitude(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Effective AB magnitude of the adopted frequency-dependent 3C 273 SED."""
    mag = -2.5 * np.log10(np.asarray(source_fnu_jy(freq_hz)) / 3631.0)
    if np.isscalar(freq_hz):
        return float(mag)
    return mag


def source_mode_occupation(
    freq_hz: np.ndarray | float,
    *,
    diameter_m: float = 4.0,
    capture_efficiency: float = 0.30,
) -> np.ndarray | float:
    """Mean source occupation per station, temporal mode, and frequency."""
    area = np.pi * (diameter_m / 2.0) ** 2
    fnu_si = np.asarray(source_fnu_jy(freq_hz)) * 1e-26
    occ = area * capture_efficiency * fnu_si / (H_PLANCK * np.asarray(freq_hz))
    if np.isscalar(freq_hz):
        return float(occ)
    return occ


def make_source(n: int, half_width_uas: float) -> tuple[np.ndarray, np.ndarray]:
    half = half_width_uas * UAS_TO_RAD
    x = np.linspace(-half, half, n, endpoint=False)
    y = np.linspace(-half, half, n, endpoint=False)
    xg, yg = np.meshgrid(x, y)
    r = np.sqrt(xg**2 + yg**2)
    th = np.arctan2(yg, xg)
    uas = UAS_TO_RAD

    pa = np.deg2rad(-20.0)
    xp = xg * np.cos(pa) + yg * np.sin(pa)
    yp = -xg * np.sin(pa) + yg * np.cos(pa)

    compact_disc = np.exp(-(xp**2 / (2.0 * (7.0 * uas) ** 2) + yp**2 / (2.0 * (4.5 * uas) ** 2)))
    outer_continuum = np.exp(-(xg**2 + yg**2) / (2.0 * (22.0 * uas) ** 2))
    blr = np.exp(-((r - 50.0 * uas) ** 2) / (2.0 * (9.5 * uas) ** 2))
    blr *= 1.0 + 0.18 * np.cos(th - np.deg2rad(30.0))
    jet_knot = np.exp(-((xg - 46.0 * uas) ** 2 + (yg - 9.0 * uas) ** 2) / (2.0 * (5.5 * uas) ** 2))
    inner_jet = np.exp(-((xp - 16.0 * uas) ** 2 / (2.0 * (16.0 * uas) ** 2) + yp**2 / (2.0 * (3.5 * uas) ** 2)))

    components = {
        "compact_accretion_disc_continuum": compact_disc,
        "diffuse_outer_disc_continuum": outer_continuum,
        "broad_line_region_lines": blr,
        "inner_optical_jet_knot": 0.65 * jet_knot + 0.35 * inner_jet,
    }
    image = np.zeros_like(compact_disc)
    for name, component in components.items():
        normalized = component / np.sum(component)
        image += SOURCE_COMPONENT_FRACTIONS[name] * normalized
    image = np.clip(image, 0.0, None)
    image /= image.sum()
    return image, x / UAS_TO_RAD


def visibility_grid(image: np.ndarray, fov_rad: float) -> tuple[np.ndarray, np.ndarray]:
    n = image.shape[0]
    vis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=fov_rad / n))
    return vis, freq


def interp_vis(vis_grid: np.ndarray, uv_axis: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = vis_grid.shape[0]
    du = uv_axis[1] - uv_axis[0]
    fu = (u - uv_axis[0]) / du
    fv = (v - uv_axis[0]) / du
    iu = np.floor(fu).astype(int)
    iv = np.floor(fv).astype(int)
    valid = (iu >= 0) & (iu < n - 1) & (iv >= 0) & (iv < n - 1)
    out = np.zeros_like(u, dtype=complex)
    tu = fu[valid] - iu[valid]
    tv = fv[valid] - iv[valid]
    g00 = vis_grid[iv[valid], iu[valid]]
    g10 = vis_grid[iv[valid], iu[valid] + 1]
    g01 = vis_grid[iv[valid] + 1, iu[valid]]
    g11 = vis_grid[iv[valid] + 1, iu[valid] + 1]
    out[valid] = (
        (1 - tu) * (1 - tv) * g00
        + tu * (1 - tv) * g10
        + (1 - tu) * tv * g01
        + tu * tv * g11
    )
    return out


def edge_list(n_station: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_station) for j in range(i + 1, n_station)]


def root_cycle_basis(edges: list[tuple[int, int]], n_station: int) -> np.ndarray:
    edge_index = {e: k for k, e in enumerate(edges)}
    cycles = []
    for i in range(1, n_station):
        for j in range(i + 1, n_station):
            w = np.zeros(len(edges))
            w[edge_index[(0, i)]] = 1.0
            w[edge_index[(i, j)]] = 1.0
            w[edge_index[(0, j)]] = -1.0
            cycles.append(w)
    return np.column_stack(cycles)


def closure_space_noise(rng: np.random.Generator, w_basis: np.ndarray, sigma_edge: float) -> np.ndarray:
    coeff = rng.normal(size=w_basis.shape[1])
    noise = w_basis @ coeff
    rms = np.sqrt(np.mean(noise**2))
    if rms > 0:
        noise *= sigma_edge / rms
    return noise


def project_cycle_space(noise: np.ndarray, w_basis: np.ndarray) -> np.ndarray:
    """Project edge-phase perturbations onto the station-piston-protected cycle space."""
    gram = w_basis.T @ w_basis
    return w_basis @ np.linalg.solve(gram, w_basis.T @ noise)


def orthonormal_cycle_basis(w_basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the same cycle space as w_basis."""
    q, _ = np.linalg.qr(w_basis)
    return q[:, : w_basis.shape[1]]


def edge_cut_basis(edges: list[tuple[int, int]], n_station: int) -> np.ndarray:
    """Station-gauge tangent basis in oriented edge-phase coordinates."""
    cut = np.zeros((len(edges), n_station - 1), dtype=float)
    for edge_index, (i, j) in enumerate(edges):
        for station in (i, j):
            if station == 0:
                continue
            cut[edge_index, station - 1] += 1.0 if station == i else -1.0
    return cut


def qfi_from_bmat_derivatives(
    bmat: np.ndarray,
    derivatives: list[np.ndarray],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """SLD Fisher matrix for an unnormalized one-click matrix."""
    bmat = 0.5 * (bmat + bmat.conj().T)
    evals, evecs = np.linalg.eigh(bmat)
    if np.min(evals) < eig_floor:
        evals = np.maximum(evals, eig_floor)
        bmat = (evecs * evals) @ evecs.conj().T
        evals, evecs = np.linalg.eigh(0.5 * (bmat + bmat.conj().T))
    deriv_p = [evecs.conj().T @ deriv @ evecs for deriv in derivatives]
    fisher = np.zeros((len(derivatives), len(derivatives)), dtype=float)
    for a, da in enumerate(deriv_p):
        for b, db in enumerate(deriv_p):
            value = 0.0
            for m, lm in enumerate(evals):
                for n, ln in enumerate(evals):
                    denom = lm + ln
                    if denom > eig_floor:
                        value += 2.0 * np.real(da[m, n] * db[n, m]) / denom
            fisher[a, b] = value
    return 0.5 * (fisher + fisher.T)


def closure_fisher_after_gauge_marginalization(
    edge_fisher: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    n_station: int,
) -> np.ndarray:
    """Effective closure Fisher after eliminating station-gauge nuisance parameters."""
    cut = edge_cut_basis(edges, n_station)
    jqq = q_basis.T @ edge_fisher @ q_basis
    jqg = q_basis.T @ edge_fisher @ cut
    jgg = cut.T @ edge_fisher @ cut
    effective = jqq - jqg @ np.linalg.pinv(jgg, rcond=1e-12) @ jqg.T
    effective = 0.5 * (effective + effective.T)
    evals, evecs = np.linalg.eigh(effective)
    evals = np.maximum(evals, 0.0)
    return 0.5 * ((evecs * evals) @ evecs.T + ((evecs * evals) @ evecs.T).T)


def closure_qfi_from_station_modes(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-10,
) -> np.ndarray:
    """Conditional SLD-QFI matrix for cycle-space phase coordinates.

    The conditional one-photon density matrix is built directly in the N station
    modes after link loss.  The input must retain the complex visibility phases:
    replacing the mutual coherences by amplitudes alone can make the station
    coherence matrix non-physical for extended sources.  The returned matrix is
    per detected source photon and is expressed in the orthonormal edge-cycle
    coordinates q_basis.
    """
    n_station = len(station_efficiencies)
    bmat = np.diag(station_efficiencies).astype(complex)
    source_coherences = []
    for edge_index, (i, j) in enumerate(edges):
        coherence = np.sqrt(station_efficiencies[i] * station_efficiencies[j]) * visibilities[edge_index]
        source_coherences.append(coherence)
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros_like(bmat, dtype=complex)
        deriv[i, j] = 1j * source_coherences[edge_index]
        deriv[j, i] = -1j * np.conj(source_coherences[edge_index])
        edge_derivatives.append(deriv)

    trace = float(np.trace(bmat).real)
    edge_fisher = qfi_from_bmat_derivatives(bmat / trace, [deriv / trace for deriv in edge_derivatives], eig_floor=eig_floor)
    return closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)


def noisy_closure_fisher_from_station_modes(
    visibilities: np.ndarray,
    station_efficiencies: np.ndarray,
    station_noise: np.ndarray,
    u_mode: float,
    q_basis: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    eig_floor: float = 1e-12,
) -> np.ndarray:
    """Per-temporal-mode SLD Fisher matrix including station-local channel noise.

    The unnormalized one-click matrix is B = source + channel background, with
    diagonal B_ii = u eta_i + epsilon_i and off-diagonal source coherences
    B_ij = u sqrt(eta_i eta_j) V_ij.  Since phase parameters do not change
    Tr(B), the Fisher matrix per temporal mode is Tr(B) times the SLD-QFI of
    rho_B = B / Tr(B).  This avoids assigning unrelated station noise as a
    scalar penalty to every closure direction.
    """
    n_station = len(station_efficiencies)
    bmat = np.diag(u_mode * station_efficiencies + station_noise).astype(complex)
    source_coherences = []
    for edge_index, (i, j) in enumerate(edges):
        coherence = (
            u_mode
            * np.sqrt(station_efficiencies[i] * station_efficiencies[j])
            * visibilities[edge_index]
        )
        source_coherences.append(coherence)
        bmat[i, j] = coherence
        bmat[j, i] = np.conj(coherence)

    edge_derivatives = []
    for edge_index, (i, j) in enumerate(edges):
        deriv = np.zeros_like(bmat, dtype=complex)
        deriv[i, j] = 1j * source_coherences[edge_index]
        deriv[j, i] = -1j * np.conj(source_coherences[edge_index])
        edge_derivatives.append(deriv)

    edge_fisher = qfi_from_bmat_derivatives(bmat, edge_derivatives, eig_floor=eig_floor)
    return closure_fisher_after_gauge_marginalization(edge_fisher, q_basis, edges, n_station)


def sample_cycle_noise_from_fisher(
    rng: np.random.Generator,
    fisher: np.ndarray,
    q_basis: np.ndarray,
    *,
    max_std: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample edge-phase noise from a cycle-coordinate Fisher matrix."""
    evals, evecs = np.linalg.eigh(0.5 * (fisher + fisher.T))
    eval_floor = 1.0 / max_std**2
    safe = np.maximum(evals, eval_floor)
    coeff = evecs @ (rng.normal(size=len(evals)) / np.sqrt(safe))
    edge_noise = q_basis @ coeff
    coord_cov = (evecs / safe) @ evecs.T
    edge_cov = q_basis @ coord_cov @ q_basis.T
    return edge_noise, np.sqrt(np.maximum(np.diag(edge_cov), 0.0))


def grid_dirty(
    u: np.ndarray,
    v: np.ndarray,
    vis: np.ndarray,
    weights: np.ndarray,
    *,
    n: int,
    fov_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    du = 1.0 / fov_rad
    mid = n // 2
    grid = np.zeros((n, n), dtype=complex)
    psf_grid = np.zeros((n, n), dtype=complex)

    def deposit(us: np.ndarray, vs: np.ndarray, vals: np.ndarray, ww: np.ndarray) -> None:
        fu = us / du + mid
        fv = vs / du + mid
        iu0 = np.floor(fu).astype(int)
        iv0 = np.floor(fv).astype(int)
        tu = fu - iu0
        tv = fv - iv0
        for ou, wu in ((0, 1.0 - tu), (1, tu)):
            for ov, wv in ((0, 1.0 - tv), (1, tv)):
                iu = iu0 + ou
                iv = iv0 + ov
                valid = (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
                frac = wu[valid] * wv[valid] * ww[valid]
                np.add.at(grid, (iv[valid], iu[valid]), frac * vals[valid])
                np.add.at(psf_grid, (iv[valid], iu[valid]), frac)

    deposit(u, v, vis, weights)
    deposit(-u, -v, np.conj(vis), weights)

    dirty = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(psf_grid))).real
    peak = psf[mid, mid]
    dirty /= peak
    psf /= peak
    return dirty, psf


def robust_rms(image: np.ndarray) -> float:
    med = np.median(image)
    return 1.4826 * np.median(np.abs(image - med))


def hogbom_clean(
    dirty: np.ndarray,
    psf: np.ndarray,
    *,
    gain: float = 0.055,
    max_iter: int = 6500,
    threshold_factor: float = 1.0,
    beam_sigma_pix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = dirty.shape[0]
    center = n // 2
    residual = dirty.copy()
    comp = np.zeros_like(dirty)
    threshold = threshold_factor * robust_rms(dirty)

    for _ in range(max_iter):
        iy, ix = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        peak = residual[iy, ix]
        if abs(peak) < threshold:
            break
        amp = gain * peak
        comp[iy, ix] += amp
        shifted = np.roll(np.roll(psf, iy - center, axis=0), ix - center, axis=1)
        residual -= amp * shifted

    clean = gaussian_filter(comp, beam_sigma_pix) + residual
    return clean, residual


def linear_wiener_image(dirty: np.ndarray, psf: np.ndarray, *, alpha: float = 0.0008) -> np.ndarray:
    """Conservative linear deconvolution of the broadband dirty map."""
    d_ft = np.fft.fft2(np.fft.ifftshift(dirty))
    p_ft = np.fft.fft2(np.fft.ifftshift(psf))
    filt = np.conj(p_ft) / (np.abs(p_ft) ** 2 + alpha)
    image = np.fft.fftshift(np.fft.ifft2(d_ft * filt)).real
    image -= np.percentile(image, 0.5)
    return gaussian_filter(np.clip(image, 0.0, None), 0.12)


def multiscale_clean(
    dirty: np.ndarray,
    psf: np.ndarray,
    *,
    scales_pix: tuple[float, ...] = (0.0, 2.0, 4.5, 8.0, 14.0),
    gain: float = 0.12,
    max_iter: int = 1200,
    threshold_factor: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Small multiscale CLEAN variant for extended ring-like structures."""
    n = dirty.shape[0]
    center = n // 2
    residual = dirty.copy()
    model = np.zeros_like(dirty)
    threshold = threshold_factor * robust_rms(dirty)
    kernels = []
    responses = []
    yy, xx = np.indices((n, n))
    rr2 = (xx - center) ** 2 + (yy - center) ** 2
    for scale in scales_pix:
        if scale == 0:
            kernel = np.zeros_like(dirty)
            kernel[center, center] = 1.0
        else:
            kernel = np.exp(-0.5 * rr2 / scale**2)
            kernel /= np.sum(kernel)
        kernels.append(kernel)
        response = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(psf)) * np.fft.fft2(np.fft.ifftshift(kernel)))
        ).real
        peak = response[center, center]
        responses.append(response / peak)

    for _ in range(max_iter):
        best = None
        for scale, kernel, response in zip(scales_pix, kernels, responses):
            smoothed = residual if scale == 0 else gaussian_filter(residual, scale)
            iy, ix = np.unravel_index(np.argmax(np.abs(smoothed)), smoothed.shape)
            peak = smoothed[iy, ix]
            score = abs(peak) / (1.0 + 0.03 * scale)
            if best is None or score > best[0]:
                best = (score, peak, iy, ix, kernel, response)
        assert best is not None
        _, peak, iy, ix, kernel, response = best
        if abs(peak) < threshold:
            break
        amp = gain * peak
        model += amp * np.roll(np.roll(kernel, iy - center, axis=0), ix - center, axis=1)
        residual -= amp * np.roll(np.roll(response, iy - center, axis=0), ix - center, axis=1)

    clean = gaussian_filter(model, 0.7) + residual
    clean -= np.percentile(clean, 0.5)
    return np.clip(clean, 0.0, None), residual


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    clipped = image.copy()
    clipped -= np.percentile(clipped, 1.0)
    scale = np.percentile(clipped, 99.7)
    if scale <= 0:
        scale = np.max(np.abs(clipped))
    normalized = np.clip(clipped / scale, 0.0, None)
    return np.arcsinh(12.0 * normalized) / np.arcsinh(12.0)


def corrcoef_positive(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_for_display(a).ravel()
    bb = normalize_for_display(b).ravel()
    return float(np.corrcoef(aa, bb)[0, 1])


def phase_jitter_label(std_rad: float) -> str:
    if np.isclose(std_rad, np.pi / 20.0):
        return r"\pi/20"
    if np.isclose(std_rad, np.pi / 10.0):
        return r"\pi/10"
    return f"{std_rad:.2f} rad"


def main() -> None:
    rng = np.random.default_rng(273)
    drift_rng = np.random.default_rng(31415)
    n_station = len(STATIONS_KM)
    edges = edge_list(n_station)
    baselines_km = np.array([STATIONS_KM[j] - STATIONS_KM[i] for i, j in edges])
    b_lengths = np.linalg.norm(baselines_km, axis=1)
    w_basis = root_cycle_basis(edges, n_station)
    q_basis = orthonormal_cycle_basis(w_basis)
    n_closure = int((n_station - 1) * (n_station - 2) / 2)
    closure_rank_share = min(1.0, (n_station - 1.0) / n_closure)
    all_baseline_detector_gain = np.sqrt(len(edges) / n_closure)
    raw_baseline_noise_gain = float(os.environ.get("RAW_BASELINE_NOISE_GAIN", "1.0"))
    closure_schedule_duty = float(os.environ.get("CLOSURE_SCHEDULE_DUTY", str(closure_rank_share)))

    n_pix = 256
    half_width_uas = 80.0
    fov_rad = 2.0 * half_width_uas * UAS_TO_RAD
    truth, axis_uas = make_source(n_pix, half_width_uas)
    vgrid, uv_axis = visibility_grid(truth, fov_rad)

    n_time = 36
    n_lambda = 36
    observing_days = OBSERVING_DAYS
    exposure_s = 600.0
    reference_visibility = 0.1
    imaging_snr_boost = float(os.environ.get("IMAGING_SNR_BOOST", "1.0"))
    reconstruction_method = os.environ.get("RECON_METHOD", "wiener").strip().lower()
    # Residual local station-piston drift after the multi-night average.  The
    # user-facing value is the induced rms phase error on one baseline.
    post_average_drift_std = float(
        os.environ.get("POST_AVERAGE_DRIFT_STD", os.environ.get("BASELINE_PHASE_JITTER_STD", str(np.pi / 10.0)))
    )
    station_piston_std = post_average_drift_std / np.sqrt(2.0)
    output_suffix = os.environ.get("OUTPUT_SUFFIX", "")
    hub_distances_km = np.linalg.norm(STATIONS_KM - COMBINER_HUB_KM, axis=1)
    effective_hub_distances_km = FIBER_LENGTH_SCALE * hub_distances_km
    station_link_eff = 10.0 ** (-FIBER_LOSS_DB_PER_KM * effective_hub_distances_km / 10.0)
    station_channel_noise = np.full_like(station_link_eff, MODE_FALSE_POSITIVE)
    baseline_link_eff = np.array([np.sqrt(station_link_eff[i] * station_link_eff[j]) for i, j in edges])
    baseline_load_eff = np.array([(station_link_eff[i] + station_link_eff[j]) / 2.0 for i, j in edges])
    baseline_noise_eff = np.array([(station_channel_noise[i] + station_channel_noise[j]) / 2.0 for i, j in edges])
    # Each station sends only 1/(N-1) of its field intensity to a given
    # baseline.  The mutual coherence scales as sqrt(f_i f_j), hence as the
    # same fraction for equal splitting, not as the two-arm photon sum.
    edge_split_coherence_eff = baseline_link_eff / (n_station - 1.0)
    edge_split_load_eff = 2.0 * baseline_load_eff / (n_station - 1.0)
    split_pickoff_fraction = 1.0 / (n_station - 1.0)
    edge_split_channel_noise = 2.0 * split_pickoff_fraction * baseline_noise_eff + PAIR_FALSE_POSITIVE
    direct_channel_noise = float(np.sum(station_channel_noise))

    lam_edges = np.linspace(400e-9, 800e-9, n_lambda + 1)
    lam_centers = np.sqrt(lam_edges[:-1] * lam_edges[1:])
    rot = np.linspace(0.0, np.pi, n_time, endpoint=False)

    all_u: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    all_vis_split: list[np.ndarray] = []
    all_vis_direct: list[np.ndarray] = []
    all_vis_all_baseline_jitter: list[np.ndarray] = []
    all_weights_split: list[np.ndarray] = []
    all_weights_direct: list[np.ndarray] = []
    all_weights_all_baseline_jitter: list[np.ndarray] = []
    all_sigma_split: list[np.ndarray] = []
    all_sigma_split_projected: list[np.ndarray] = []
    all_sigma_direct: list[np.ndarray] = []
    all_visibility_amp: list[np.ndarray] = []

    for lam, lam_lo, lam_hi in zip(lam_centers, lam_edges[:-1], lam_edges[1:]):
        freq = C_LIGHT / lam
        freq_lo = C_LIGHT / lam_hi
        freq_hi = C_LIGHT / lam_lo
        df = freq_hi - freq_lo
        u_mode = source_mode_occupation(freq, diameter_m=TELESCOPE_DIAMETER_M)
        total_modes = exposure_s * observing_days * df

        for theta in rot:
            c = np.cos(theta)
            s = np.sin(theta)
            bx = baselines_km[:, 0]
            by = baselines_km[:, 1]
            u = (bx * c - by * s) * 1000.0 / lam
            v = (bx * s + by * c) * 1000.0 / lam
            uv_radius = np.sqrt(u**2 + v**2)
            vtrue = interp_vis(vgrid, uv_axis, u, v)
            amp = np.abs(vtrue)
            phase = np.angle(vtrue)
            phase_closure = q_basis @ (q_basis.T @ phase)
            nu_eff = np.clip(amp, 1e-4, 0.98)
            coherent_signal_split = edge_split_coherence_eff * u_mode
            noise_load_split = edge_split_load_eff * u_mode
            fisher_split = (
                total_modes
                * 4.0
                * coherent_signal_split**2
                * nu_eff**2
                / (noise_load_split + edge_split_channel_noise)
            )
            sigma_split = 1.0 / np.sqrt(np.maximum(fisher_split, 1e-18))
            sigma_split /= imaging_snr_boost
            sigma_split = np.minimum(sigma_split, 2.5)
            raw_split_noise = rng.normal(scale=sigma_split)
            noise_split = q_basis @ (q_basis.T @ raw_split_noise)
            cov_split_cycle = q_basis.T @ ((sigma_split**2)[:, None] * q_basis)
            cov_split_edge = q_basis @ cov_split_cycle @ q_basis.T
            sigma_split_projected = np.sqrt(np.maximum(np.diag(cov_split_edge), 0.0))
            fisher_direct_cycle = (
                total_modes
                * noisy_closure_fisher_from_station_modes(
                    vtrue, station_link_eff, station_channel_noise, u_mode, q_basis, edges
                )
                * closure_schedule_duty
                * imaging_snr_boost**2
            )
            noise_direct, sigma_direct = sample_cycle_noise_from_fisher(rng, fisher_direct_cycle, q_basis)
            # The all-visibility panel directly uses the measured complex
            # visibilities without a closure projection.  Its per-edge detector
            # noise is the same split-readout noise budget as edge-first; no
            # sqrt(E/C)-type prefactor is inserted into a single baseline.
            # RAW_BASELINE_NOISE_GAIN is retained only for explicit stress tests.
            station_pistons = drift_rng.normal(scale=station_piston_std, size=n_station)
            station_pistons -= np.mean(station_pistons)
            residual_drift = np.array([station_pistons[i] - station_pistons[j] for i, j in edges])
            noise_all_baseline = raw_split_noise / raw_baseline_noise_gain + residual_drift
            uv_taper = np.exp(-(uv_radius / 64e9) ** 2) * (0.30 + np.sqrt(np.maximum(uv_radius, 1.0) / 18e9))
            weight_split = uv_taper / (sigma_split_projected**2 + 0.12**2)
            weight_direct = uv_taper / (sigma_direct**2 + 0.12**2)
            weight_all_baseline = uv_taper / (
                (sigma_split / raw_baseline_noise_gain) ** 2 + post_average_drift_std**2 + 0.12**2
            )
            all_u.append(u)
            all_v.append(v)
            all_vis_split.append(amp * np.exp(1j * (phase_closure + noise_split)))
            all_vis_direct.append(amp * np.exp(1j * (phase_closure + noise_direct)))
            all_vis_all_baseline_jitter.append(amp * np.exp(1j * (phase + noise_all_baseline)))
            all_weights_split.append(weight_split)
            all_weights_direct.append(weight_direct)
            all_weights_all_baseline_jitter.append(weight_all_baseline)
            all_sigma_split.append(sigma_split)
            all_sigma_split_projected.append(sigma_split_projected)
            all_sigma_direct.append(sigma_direct)
            all_visibility_amp.append(amp)

    u_all = np.concatenate(all_u)
    v_all = np.concatenate(all_v)
    split_all = np.concatenate(all_vis_split)
    direct_all = np.concatenate(all_vis_direct)
    all_baseline_jitter_all = np.concatenate(all_vis_all_baseline_jitter)
    weights_split = np.concatenate(all_weights_split)
    weights_direct = np.concatenate(all_weights_direct)
    weights_all_baseline_jitter = np.concatenate(all_weights_all_baseline_jitter)
    sigma_split_all = np.concatenate(all_sigma_split)
    sigma_split_projected_all = np.concatenate(all_sigma_split_projected)
    sigma_direct_all = np.concatenate(all_sigma_direct)
    visibility_amp_all = np.concatenate(all_visibility_amp)

    zero_weight_split = 0.0025 * np.sum(weights_split)
    zero_weight_direct = 0.0025 * np.sum(weights_direct)
    zero_weight_all_baseline = 0.0025 * np.sum(weights_all_baseline_jitter)
    u_all = np.concatenate([u_all, np.array([0.0])])
    v_all = np.concatenate([v_all, np.array([0.0])])
    split_all = np.concatenate([split_all, np.array([1.0 + 0.0j])])
    direct_all = np.concatenate([direct_all, np.array([1.0 + 0.0j])])
    all_baseline_jitter_all = np.concatenate([all_baseline_jitter_all, np.array([1.0 + 0.0j])])
    weights_split = np.concatenate([weights_split, np.array([zero_weight_split])])
    weights_direct = np.concatenate([weights_direct, np.array([zero_weight_direct])])
    weights_all_baseline_jitter = np.concatenate(
        [weights_all_baseline_jitter, np.array([zero_weight_all_baseline])]
    )

    dirty_all_baseline, psf_all_baseline = grid_dirty(
        u_all, v_all, all_baseline_jitter_all, weights_all_baseline_jitter, n=n_pix, fov_rad=fov_rad
    )
    dirty_split, psf_split = grid_dirty(u_all, v_all, split_all, weights_split, n=n_pix, fov_rad=fov_rad)
    dirty_direct, psf_direct = grid_dirty(u_all, v_all, direct_all, weights_direct, n=n_pix, fov_rad=fov_rad)
    if reconstruction_method in {"dirty", "none", "no_clean", "no-deconv", "no_deconv"}:
        recon_all_baseline = dirty_all_baseline
        recon_split = dirty_split
        recon_direct = dirty_direct
        msclean_split = dirty_split
        msclean_direct = dirty_direct
        resid_split = np.zeros_like(dirty_split)
        resid_direct = np.zeros_like(dirty_direct)
    elif reconstruction_method in {"wiener", "linear", "regularized"}:
        recon_all_baseline = linear_wiener_image(dirty_all_baseline, psf_all_baseline, alpha=0.0012)
        recon_split = linear_wiener_image(dirty_split, psf_split, alpha=0.0012)
        recon_direct = linear_wiener_image(dirty_direct, psf_direct, alpha=0.0012)
        msclean_split, resid_split = multiscale_clean(dirty_split, psf_split)
        msclean_direct, resid_direct = multiscale_clean(dirty_direct, psf_direct)
    elif reconstruction_method in {"clean", "multiscale_clean"}:
        recon_all_baseline = linear_wiener_image(dirty_all_baseline, psf_all_baseline, alpha=0.0012)
        recon_split, resid_split = multiscale_clean(dirty_split, psf_split)
        recon_direct, resid_direct = multiscale_clean(dirty_direct, psf_direct)
        msclean_split = recon_split
        msclean_direct = recon_direct
    else:
        raise ValueError(
            "RECON_METHOD must be one of wiener, dirty/no_clean/no_deconv, or clean/multiscale_clean"
        )

    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
        }
    )
    fig = plt.figure(figsize=(7.35, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.36, wspace=0.33)
    if imaging_snr_boost != 1.0:
        fig.suptitle(f"Realistic broadband optical AGN model, SNR x{imaging_snr_boost:g}", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(STATIONS_KM[:, 0], STATIONS_KM[:, 1], s=26, color="#005f73")
    ax.scatter(
        [COMBINER_HUB_KM[0]],
        [COMBINER_HUB_KM[1]],
        s=58,
        marker="*",
        color="#ca6702",
        label="combiner hub",
        zorder=3,
    )
    for i, j in edges:
        ax.plot(
            [STATIONS_KM[i, 0], STATIONS_KM[j, 0]],
            [STATIONS_KM[i, 1], STATIONS_KM[j, 1]],
            color="0.78",
            lw=0.55,
            zorder=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east-west baseline coordinate (km)")
    ax.set_ylabel("north-south coordinate (km)")
    ax.set_title("Eight stations + central hub")
    ax.legend(loc="upper left", frameon=False, fontsize=7.4)

    ax = fig.add_subplot(gs[0, 1])
    stride = max(len(u_all) // 7000, 1)
    ax.scatter(u_all[::stride] / 1e9, v_all[::stride] / 1e9, s=0.8, color="#0a9396", alpha=0.35)
    ax.scatter(-u_all[::stride] / 1e9, -v_all[::stride] / 1e9, s=0.8, color="#0a9396", alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")

    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(normalize_for_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    ax.set_title("Input source")
    image_axes.append(ax)

    panels = [
        (recon_all_baseline, rf"All visibilities + piston drift, $\sigma_\phi={phase_jitter_label(post_average_drift_std)}$"),
        (recon_split, "Edge-first closure"),
        (recon_direct, "Scheduled closure-space"),
    ]
    for idx, (image, title) in enumerate(panels):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(normalize_for_display(image), origin="lower", extent=extent, cmap="inferno")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if idx == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        ax.set_title(title)
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.024,
        pad=0.018,
    )
    cbar.set_label("norm. brightness\n(arcsinh)", fontsize=7.0)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=6.6)

    png = OUTDIR / f"prl_broadband_clean_demo{output_suffix}.png"
    pdf = OUTDIR / f"prl_broadband_clean_demo{output_suffix}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    compare_panels = [
        (recon_all_baseline, "All vis. + piston drift"),
        (recon_split, "Edge-first closure"),
        (recon_direct, "Scheduled closure-space"),
    ]
    fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(7.25, 2.42), constrained_layout=True)
    if imaging_snr_boost != 1.0:
        fig_cmp.suptitle(f"Reconstruction comparison, SNR x{imaging_snr_boost:g}", fontsize=9.6)
    cmp_images = []
    for ax_cmp, (image, title) in zip(axes_cmp, compare_panels):
        shown = normalize_for_display(image)
        im = ax_cmp.imshow(shown, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0)
        cmp_images.append(im)
        ax_cmp.set_title(f"{title}\nr={corrcoef_positive(truth, image):.3f}", fontsize=7.8)
        ax_cmp.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        ax_cmp.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    cbar_cmp = fig_cmp.colorbar(cmp_images[-1], ax=axes_cmp, fraction=0.030, pad=0.018)
    cbar_cmp.set_label("norm. brightness\n(arcsinh)", fontsize=7.0)
    cbar_cmp.set_ticks([0.0, 0.5, 1.0])
    cbar_cmp.ax.tick_params(labelsize=6.6)
    compare_png = OUTDIR / f"prl_broadband_recon_compare{output_suffix}.png"
    compare_pdf = OUTDIR / f"prl_broadband_recon_compare{output_suffix}.pdf"
    fig_cmp.savefig(compare_png, dpi=260, bbox_inches="tight")
    fig_cmp.savefig(compare_pdf, bbox_inches="tight")

    stats = {
        "n_stations": n_station,
        "n_baselines": len(edges),
        "n_independent_closures": n_closure,
        "raw_baseline_noise_gain": float(raw_baseline_noise_gain),
        "closure_receiver_model": "N-mode SLD QFI matrix",
        "closure_rank_share_default": closure_rank_share,
        "closure_schedule_duty": closure_schedule_duty,
        "all_baseline_panel_uses_closure_projection": False,
        "baseline_min_km": float(np.min(b_lengths)),
        "baseline_max_km": float(np.max(b_lengths)),
        "combiner_hub_km": COMBINER_HUB_KM.tolist(),
        "hub_distance_min_km": float(np.min(hub_distances_km)),
        "hub_distance_max_km": float(np.max(hub_distances_km)),
        "fiber_length_scale": float(FIBER_LENGTH_SCALE),
        "effective_hub_distance_min_km": float(np.min(effective_hub_distances_km)),
        "effective_hub_distance_max_km": float(np.max(effective_hub_distances_km)),
        "fiber_loss_db_per_km": FIBER_LOSS_DB_PER_KM,
        "channel_noise_model": "pure_fibre_attenuation_plus_independent_mode_false_positive",
        "station_link_eff_min": float(np.min(station_link_eff)),
        "station_link_eff_max": float(np.max(station_link_eff)),
        "station_channel_noise_min": float(np.min(station_channel_noise)),
        "station_channel_noise_max": float(np.max(station_channel_noise)),
        "baseline_link_eff_min": float(np.min(baseline_link_eff)),
        "baseline_link_eff_max": float(np.max(baseline_link_eff)),
        "baseline_load_eff_min": float(np.min(baseline_load_eff)),
        "baseline_load_eff_max": float(np.max(baseline_load_eff)),
        "edge_split_channel_noise_percentiles": np.percentile(
            edge_split_channel_noise, [5, 25, 50, 75, 95]
        ).tolist(),
        "direct_channel_noise_sum": direct_channel_noise,
        "split_pickoff_fraction_per_station_per_baseline": float(split_pickoff_fraction),
        "edge_split_coherence_eff_percentiles": np.percentile(
            edge_split_coherence_eff, [5, 25, 50, 75, 95]
        ).tolist(),
        "edge_split_load_eff_percentiles": np.percentile(edge_split_load_eff, [5, 25, 50, 75, 95]).tolist(),
        "n_time_samples": n_time,
        "n_lambda_bins": n_lambda,
        "observing_days": observing_days,
        "exposure_s_per_sample": exposure_s,
        "mode_false_positive_per_station_mode": MODE_FALSE_POSITIVE,
        "pair_false_positive_per_pair_combiner": PAIR_FALSE_POSITIVE,
        "additive_noise_model": "independent detector/background false-positive; not fibre loss",
        "n_complex_visibility_samples": int(len(u_all)),
        "wavelength_min_nm": 400.0,
        "wavelength_max_nm": 800.0,
        "field_half_width_uas": half_width_uas,
        "source_spectrum_name": SOURCE_SPECTRUM_NAME,
        "source_spectrum_note": SOURCE_SPECTRUM_NOTE,
        "source_spectrum_ned_points_400_800nm": SOURCE_SED_NED_POINT_COUNT,
        "source_spectrum_lambda_nm": SOURCE_SED_LAMBDA_NM.tolist(),
        "source_spectrum_fnu_jy": SOURCE_SED_FNU_JY.tolist(),
        "telescope_diameter_m": float(TELESCOPE_DIAMETER_M),
        "source_component_fractions": SOURCE_COMPONENT_FRACTIONS,
        "baseline_phase_jitter_std_rad": float(post_average_drift_std),
        "post_average_drift_std_rad": float(post_average_drift_std),
        "station_piston_std_rad": float(station_piston_std),
        "imaging_snr_boost": imaging_snr_boost,
        "reconstruction_method": reconstruction_method,
        "visibility_amp_percentiles": np.percentile(visibility_amp_all, [5, 25, 50, 75, 95]).tolist(),
        "sigma_phase_split_percentiles_rad": np.percentile(sigma_split_all, [5, 25, 50, 75, 95]).tolist(),
        "sigma_phase_split_projected_percentiles_rad": np.percentile(
            sigma_split_projected_all, [5, 25, 50, 75, 95]
        ).tolist(),
        "sigma_phase_direct_percentiles_rad": np.percentile(sigma_direct_all, [5, 25, 50, 75, 95]).tolist(),
        "image_correlation_all_baseline_jitter": corrcoef_positive(truth, recon_all_baseline),
        "image_correlation_split": corrcoef_positive(truth, recon_split),
        "image_correlation_direct": corrcoef_positive(truth, recon_direct),
        "multiscale_clean_correlation_split": corrcoef_positive(truth, msclean_split),
        "multiscale_clean_correlation_direct": corrcoef_positive(truth, msclean_direct),
        "residual_rms_split": float(robust_rms(resid_split)),
        "residual_rms_direct": float(robust_rms(resid_direct)),
        "stations_km": STATIONS_KM.tolist(),
    }
    stats_path = OUTDIR / f"prl_broadband_clean_stats{output_suffix}.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(compare_pdf)
    print(compare_png)
    print(stats_path)


if __name__ == "__main__":
    main()
