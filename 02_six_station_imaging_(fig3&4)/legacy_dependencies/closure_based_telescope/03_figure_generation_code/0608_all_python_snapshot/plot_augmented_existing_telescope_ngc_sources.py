from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import plot_augmented_existing_telescope_closure_networks as aug
import plot_augmented_far_snr100_weighting_test as wt
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
from scan_augmented_far_hybrid_density_nearest import reconstruct_band_hybrid


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTFIG = ROOT / "output" / "figures"
OUTFIG.mkdir(parents=True, exist_ok=True)
SOURCE_MORPHOLOGY = os.environ.get("NGC_SOURCE_MORPHOLOGY", "lopsided_crescent").strip().lower()

TARGET_MODES = {
    "uniform_p1_fill05": {"label": "uniform-area p=1 + low fill", "power": 1.0, "fill": 0.5},
}
DISPLAY_MODE = "uniform_p1_fill05"


@dataclass(frozen=True)
class SourceModel:
    key: str
    name: str
    dec_deg: float
    tau_hbeta_days: float
    distance_mpc: float
    mbh_msun: float
    blr_radius_uas: float
    blr_width_uas: float
    disc_sigma_major_uas: float
    disc_sigma_minor_uas: float
    position_angle_deg: float
    blr_fraction: float
    sed_lambda_nm: tuple[float, ...]
    sed_fnu_mjy: tuple[float, ...]
    sed_reference: str
    note: str


NGC4151 = SourceModel(
    key="ngc4151",
    name="NGC 4151",
    dec_deg=39.4057,
    tau_hbeta_days=6.6,
    distance_mpc=15.8,
    mbh_msun=4.57e7,
    # HST Cepheid distances place NGC 4151 near 15.8 Mpc.  With this
    # distance, the H-beta reverberation lag of about 6.6 light-days
    # corresponds to roughly 72 microarcsec.
    blr_radius_uas=72.0,
    blr_width_uas=12.0,
    disc_sigma_major_uas=7.8,
    disc_sigma_minor_uas=4.8,
    position_angle_deg=-25.0,
    blr_fraction=0.42,
    # Compact-nuclear optical benchmark.  The 510 nm continuum is anchored to
    # F_lambda ~ 7--9e-14 erg s^-1 cm^-2 A^-1.  Modest broad Balmer bumps are
    # included near the observed H beta and H alpha wavelengths.
    sed_lambda_nm=(400.0, 430.0, 460.0, 488.0, 510.0, 550.0, 620.0, 659.0, 700.0, 760.0, 800.0),
    sed_fnu_mjy=(47.0, 54.0, 61.0, 76.0, 69.0, 70.0, 65.0, 92.0, 67.0, 61.0, 56.0),
    sed_reference=(
        "NGC 4151 nuclear continuum anchored to optical spectroscopy at 5100 A "
        "(F_lambda about 7--9e-14 erg s^-1 cm^-2 A^-1), with approximate broad-line bumps."
    ),
    note="Northern Seyfert benchmark; BLR radius set by H-beta RM lag, not direct imaging.",
)

NGC3783 = SourceModel(
    key="ngc3783",
    name="NGC 3783",
    dec_deg=-37.7386,
    tau_hbeta_days=10.2,
    distance_mpc=41.0,
    mbh_msun=2.9e7,
    blr_radius_uas=43.0,
    blr_width_uas=8.0,
    disc_sigma_major_uas=5.5,
    disc_sigma_minor_uas=3.7,
    position_angle_deg=20.0,
    blr_fraction=0.42,
    # SMARTS monitoring gives B-band nuclear/PSF flux varying from <9 to >24
    # mJy and V varying by more than a factor of two.  We use a mid-state SED
    # rather than a variability realization.
    sed_lambda_nm=(400.0, 440.0, 470.0, 491.0, 510.0, 550.0, 620.0, 663.0, 700.0, 760.0, 800.0),
    sed_fnu_mjy=(12.0, 16.5, 17.5, 22.0, 18.0, 17.0, 15.5, 25.0, 16.0, 14.5, 13.5),
    sed_reference=(
        "NGC 3783 mid-state optical SED anchored to SMARTS B/V nuclear photometry "
        "(B varies from below 9 to above 24 mJy), with approximate broad-line bumps."
    ),
    note="Southern Seyfert benchmark; BLR radius set by H-beta RM lag, not direct imaging.",
)


def sed_effective_ab_mag(source: SourceModel, wavelength_nm: float = 550.0) -> float:
    fnu_jy = np.interp(
        math.log(wavelength_nm),
        np.log(np.asarray(source.sed_lambda_nm, dtype=float)),
        np.log(np.asarray(source.sed_fnu_mjy, dtype=float) * 1e-3),
    )
    return float(-2.5 * math.log10(math.exp(fnu_jy) / 3631.0))


def make_source_factory(source: SourceModel) -> Callable[[int, float], tuple[np.ndarray, np.ndarray]]:
    def make_source(n: int, half_width_uas: float) -> tuple[np.ndarray, np.ndarray]:
        half = half_width_uas * base.UAS_TO_RAD
        x = np.linspace(-half, half, n, endpoint=False)
        y = np.linspace(-half, half, n, endpoint=False)
        xg, yg = np.meshgrid(x, y)
        uas = base.UAS_TO_RAD
        r = np.sqrt(xg**2 + yg**2)
        th = np.arctan2(yg, xg)

        pa = np.deg2rad(source.position_angle_deg)
        xp = xg * np.cos(pa) + yg * np.sin(pa)
        yp = -xg * np.sin(pa) + yg * np.cos(pa)

        if SOURCE_MORPHOLOGY in {"asym", "lopsided", "crescent", "lopsided_crescent", "lopsided_spotted", "blr_spots"}:
            components = lopsided_crescent_components(source, xg, yg, xp, yp, r, th, uas)
        else:
            compact_disc = np.exp(
                -(
                    xp**2 / (2.0 * (source.disc_sigma_major_uas * uas) ** 2)
                    + yp**2 / (2.0 * (source.disc_sigma_minor_uas * uas) ** 2)
                )
            )
            outer_continuum = np.exp(-(xg**2 + yg**2) / (2.0 * (0.45 * source.blr_radius_uas * uas) ** 2))
            blr = np.exp(
                -((r - source.blr_radius_uas * uas) ** 2) / (2.0 * (source.blr_width_uas * uas) ** 2)
            )
            blr *= 1.0 + 0.20 * np.cos(th - np.deg2rad(35.0)) + 0.08 * np.cos(2.0 * th + np.deg2rad(20.0))
            hotspot = np.exp(
                -(
                    (xg - 0.82 * source.blr_radius_uas * uas) ** 2
                    + (yg - 0.22 * source.blr_radius_uas * uas) ** 2
                )
                / (2.0 * (0.12 * source.blr_radius_uas * uas) ** 2)
            )
            inner_asymmetry = np.exp(
                -(
                    (xp - 0.30 * source.blr_radius_uas * uas) ** 2 / (2.0 * (0.22 * source.blr_radius_uas * uas) ** 2)
                    + yp**2 / (2.0 * (0.055 * source.blr_radius_uas * uas) ** 2)
                )
            )

            blr_fraction = source.blr_fraction
            compact_fraction = 0.39
            outer_fraction = 0.14
            asym_fraction = max(1.0 - blr_fraction - compact_fraction - outer_fraction, 0.0)
            components = [
                (compact_fraction, compact_disc),
                (outer_fraction, outer_continuum),
                (blr_fraction, blr),
                (asym_fraction, 0.60 * hotspot + 0.40 * inner_asymmetry),
            ]
        image = np.zeros_like(components[0][1])
        for fraction, component in components:
            normalized = component / np.sum(component)
            image += fraction * normalized
        image = np.clip(image, 0.0, None)
        image /= image.sum()
        return image, x / base.UAS_TO_RAD

    return make_source


def make_wavelength_source_factory(source: SourceModel) -> Callable[[int, float, float], tuple[np.ndarray, np.ndarray]]:
    """Default achromatic source factory used unless a morphology patch overrides it."""
    make_source = make_source_factory(source)

    def make_source_at_wavelength_nm(n: int, half_width_uas: float, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        return make_source(n, half_width_uas)

    return make_source_at_wavelength_nm


def lopsided_crescent_components(
    source: SourceModel,
    xg: np.ndarray,
    yg: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    r: np.ndarray,
    th: np.ndarray,
    uas: float,
) -> list[tuple[float, np.ndarray]]:
    """A deliberately phase-sensitive optical source morphology.

    The default benchmark is almost an annulus plus a compact core, so calibrated
    amplitudes already constrain much of the image.  This variant keeps the same
    total flux and angular scale but makes the image strongly non-centrosymmetric:
    an eccentric inner crescent, a BLR with azimuth-dependent thickness, and a
    curved, knotty jet.  It is meant as a discriminating stress test for phase
    and closure-phase information rather than a literal fit to NGC 4151.
    """
    radius = source.blr_radius_uas * uas
    width = source.blr_width_uas * uas

    # Eccentric continuum crescent: subtract a displaced hole from a broader
    # elliptical component, then add a compact unresolved-looking nucleus.
    outer = np.exp(
        -(
            (xp - 0.08 * radius) ** 2 / (2.0 * (0.30 * radius) ** 2)
            + (yp + 0.04 * radius) ** 2 / (2.0 * (0.18 * radius) ** 2)
        )
    )
    inner_hole = 0.90 * np.exp(
        -(
            (xp + 0.12 * radius) ** 2 / (2.0 * (0.22 * radius) ** 2)
            + (yp - 0.03 * radius) ** 2 / (2.0 * (0.12 * radius) ** 2)
        )
    )
    crescent = np.clip(outer - inner_hole, 0.0, None)
    compact_core = np.exp(
        -(
            (xp + 0.025 * radius) ** 2 / (2.0 * (0.075 * radius) ** 2)
            + (yp - 0.010 * radius) ** 2 / (2.0 * (0.045 * radius) ** 2)
        )
    )

    # A BLR annulus whose center radius, width, and brightness all depend on
    # azimuth.  This creates an obviously thick crescent arc and a thin faint arc.
    bright_phase = np.deg2rad(135.0)
    radial_center = radius * (1.0 + 0.10 * np.cos(th - bright_phase) - 0.045 * np.sin(2.0 * th))
    width_mod = width * (0.55 + 0.95 * 0.5 * (1.0 + np.cos(th - bright_phase)))
    width_mod = np.maximum(width_mod, 0.38 * width)
    blr = np.exp(-((r - radial_center) ** 2) / (2.0 * width_mod**2))
    brightness = 0.18 + 1.25 * 0.5 * (1.0 + np.cos(th - bright_phase))
    notch = 1.0 - 0.55 * np.exp(-0.5 * (np.angle(np.exp(1j * (th - np.deg2rad(-55.0)))) / 0.38) ** 2)
    blr *= np.clip(brightness * notch, 0.03, None)

    # Curved jet ridge plus two uneven knots.  The ridge is expressed in the
    # source position-angle frame so it rotates with the compact continuum.
    jet_curve = yp - (0.10 * radius * np.sin((xp / radius - 0.20) * np.pi))
    jet_gate = 1.0 / (1.0 + np.exp(-(xp - 0.06 * radius) / (0.08 * radius)))
    jet_ridge = jet_gate * np.exp(-jet_curve**2 / (2.0 * (0.040 * radius) ** 2))
    jet_ridge *= np.exp(-((xp - 0.44 * radius) ** 2) / (2.0 * (0.42 * radius) ** 2))
    knot_a = np.exp(
        -(
            (xp - 0.55 * radius) ** 2 / (2.0 * (0.080 * radius) ** 2)
            + (yp - 0.15 * radius) ** 2 / (2.0 * (0.055 * radius) ** 2)
        )
    )
    knot_b = np.exp(
        -(
            (xp - 0.82 * radius) ** 2 / (2.0 * (0.055 * radius) ** 2)
            + (yp + 0.08 * radius) ** 2 / (2.0 * (0.045 * radius) ** 2)
        )
    )
    jet = 0.45 * jet_ridge + 0.35 * knot_a + 0.20 * knot_b

    diffuse_tail = np.exp(
        -(
            (xg + 0.38 * radius) ** 2 / (2.0 * (0.30 * radius) ** 2)
            + (yg + 0.20 * radius) ** 2 / (2.0 * (0.20 * radius) ** 2)
        )
    )

    if SOURCE_MORPHOLOGY in {"lopsided_spotted", "blr_spots"}:
        # Bright irregular BLR clumps.  Each spot is elongated in a different
        # direction and slightly off the mean ring radius, so the morphology is
        # not reproducible from amplitudes alone.
        spots = np.zeros_like(blr)
        spot_specs = [
            (122.0, 1.03, 0.060, 0.030, 1.25, 18.0),
            (168.0, 0.91, 0.075, 0.035, 0.92, -28.0),
            (-38.0, 1.08, 0.050, 0.026, 1.10, 47.0),
            (-104.0, 0.97, 0.065, 0.032, 0.70, -12.0),
        ]
        for angle_deg, radius_scale, sigma_long, sigma_short, weight, tilt_deg in spot_specs:
            angle = np.deg2rad(angle_deg)
            x0 = radius_scale * radius * np.cos(angle)
            y0 = radius_scale * radius * np.sin(angle)
            tilt = angle + np.deg2rad(tilt_deg)
            dx = xg - x0
            dy = yg - y0
            xl = dx * np.cos(tilt) + dy * np.sin(tilt)
            ys = -dx * np.sin(tilt) + dy * np.cos(tilt)
            spot = np.exp(
                -(
                    xl**2 / (2.0 * (sigma_long * radius) ** 2)
                    + ys**2 / (2.0 * (sigma_short * radius) ** 2)
                )
            )
            # Carve a small notch into each Gaussian so the clump edge is not a
            # perfectly smooth ellipse.
            notch_angle = tilt + np.deg2rad(74.0)
            xn = dx * np.cos(notch_angle) + dy * np.sin(notch_angle)
            yn = -dx * np.sin(notch_angle) + dy * np.cos(notch_angle)
            notch = 1.0 - 0.32 * np.exp(
                -(
                    xn**2 / (2.0 * (0.030 * radius) ** 2)
                    + yn**2 / (2.0 * (0.016 * radius) ** 2)
                )
            )
            spots += weight * np.clip(spot * notch, 0.0, None)

        return [
            (0.18, compact_core),
            (0.16, crescent),
            (0.33, blr),
            (0.11, jet),
            (0.04, diffuse_tail),
            (0.18, spots),
        ]

    return [
        (0.22, compact_core),
        (0.20, crescent),
        (0.42, blr),
        (0.12, jet),
        (0.04, diffuse_tail),
    ]


def fnu_factory(source: SourceModel) -> Callable[[np.ndarray | float], np.ndarray | float]:
    sed_lambda = np.asarray(source.sed_lambda_nm, dtype=float)
    sed_fnu_jy = np.asarray(source.sed_fnu_mjy, dtype=float) * 1e-3

    def source_fnu_jy(freq_hz: np.ndarray | float) -> np.ndarray | float:
        freq = np.asarray(freq_hz, dtype=float)
        lambda_nm = base.C_LIGHT / freq * 1e9
        log_fnu = np.interp(
            np.log(lambda_nm),
            np.log(sed_lambda),
            np.log(sed_fnu_jy),
            left=np.log(sed_fnu_jy[0]),
            right=np.log(sed_fnu_jy[-1]),
        )
        out = np.exp(log_fnu)
        if np.isscalar(freq_hz):
            return float(out)
        return out

    return source_fnu_jy


@contextmanager
def patched_source(source: SourceModel) -> Iterator[None]:
    old_make_source = base.make_source
    old_fnu = base.source_fnu_jy
    old_dec = aug.SOURCE_DEC_DEG
    old_wavelength_source = getattr(base, "make_source_at_wavelength_nm", None)
    old_component_fractions = getattr(base, "source_component_flux_fractions", None)
    old_spectral_model = getattr(base, "SOURCE_COMPONENT_SPECTRAL_MODEL", None)
    wavelength_source = make_wavelength_source_factory(source)
    base.make_source = make_source_factory(source)
    base.make_source_at_wavelength_nm = wavelength_source
    base.source_fnu_jy = fnu_factory(source)
    base.source_component_flux_fractions = getattr(wavelength_source, "component_flux_fractions", lambda wavelength_nm: {})
    base.SOURCE_COMPONENT_SPECTRAL_MODEL = getattr(
        wavelength_source,
        "spectral_model_note",
        "achromatic morphology with the total source SED",
    )
    aug.SOURCE_DEC_DEG = source.dec_deg
    try:
        yield
    finally:
        base.make_source = old_make_source
        base.source_fnu_jy = old_fnu
        if old_wavelength_source is None:
            delattr(base, "make_source_at_wavelength_nm")
        else:
            base.make_source_at_wavelength_nm = old_wavelength_source
        if old_component_fractions is None:
            delattr(base, "source_component_flux_fractions")
        else:
            base.source_component_flux_fractions = old_component_fractions
        if old_spectral_model is None:
            delattr(base, "SOURCE_COMPONENT_SPECTRAL_MODEL")
        else:
            base.SOURCE_COMPONENT_SPECTRAL_MODEL = old_spectral_model
        aug.SOURCE_DEC_DEG = old_dec


def blr_masks_for_source(axis_uas: np.ndarray, source: SourceModel) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx**2 + yy**2)
    ring_half_width = max(2.2 * source.blr_width_uas, 10.0)
    ring = (rr > source.blr_radius_uas - ring_half_width) & (rr < source.blr_radius_uas + ring_half_width)
    core = rr < max(0.55 * source.blr_radius_uas, 22.0)
    return ring, core


def reconstruct_stack(
    bands: list[dict[str, np.ndarray]],
    strategy: str,
    truth: np.ndarray,
    *,
    mode_key: str,
) -> np.ndarray:
    mode = TARGET_MODES[mode_key]
    fov_rad = 2.0 * wt.HALF_WIDTH_UAS * base.UAS_TO_RAD
    stack = np.zeros_like(truth)
    total_weight = 0.0
    for band in bands:
        image, weight = reconstruct_band_hybrid(
            band,
            strategy,
            fov_rad,
            n_bin=40,
            power=float(mode["power"]),
            fill_alpha=float(mode["fill"]),
        )
        stack += weight * image
        total_weight += weight
    return base.normalize_for_display(stack / max(total_weight, 1e-30))


def image_metrics(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray, source: SourceModel) -> dict[str, float]:
    ring_mask, core_mask = blr_masks_for_source(axis_uas, source)
    return {
        "global_corr": float(base.corrcoef_positive(truth, image)),
        "blr_corr": float(opt.masked_corr(truth, image, ring_mask)),
        "ring_contrast": float(opt.ring_contrast(image, ring_mask, core_mask)),
    }


def orbital_period_years(source: SourceModel) -> float:
    radius_m = base.C_LIGHT * 86400.0 * source.tau_hbeta_days
    mass_kg = 6.67430e-11 * source.mbh_msun * 1.98847e30
    period_s = 2.0 * math.pi * math.sqrt(radius_m**3 / mass_kg)
    return period_s / (86400.0 * 365.25)


def plot_case(
    case: aug.NetworkCase,
    source: SourceModel,
    stats: dict,
    images: dict[str, dict[str, np.ndarray]],
    truth: np.ndarray,
    axis_uas: np.ndarray,
    *,
    mode_key: str,
) -> tuple[Path, Path]:
    stations, diameters, _, is_added = aug.station_table_from_case(case)
    extent = [axis_uas[0], axis_uas[-1], axis_uas[0], axis_uas[-1]]
    fig = plt.figure(figsize=(7.55, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.38, wspace=0.34)
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.0,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
        }
    )

    ax = fig.add_subplot(gs[0, 0])
    for added, marker, color, label in (
        (False, "o", "#005f73", "existing"),
        (True, "^", "#ae2012", "new 5 m"),
    ):
        mask = is_added == added
        if np.any(mask):
            ax.scatter(stations[mask, 0], stations[mask, 1], s=30 if added else 26, marker=marker, color=color, edgecolor="white", linewidth=0.4, label=label, zorder=3)
    ax.scatter([case.hub_km[0]], [case.hub_km[1]], s=58, marker="*", color="#ca6702", label="hub", zorder=4)
    for i in range(len(stations)):
        ax.text(stations[i, 0] + 0.18, stations[i, 1] + 0.18, f"S{i+1}\n{diameters[i]:g}m", fontsize=5.6)
    for i, j in base.edge_list(len(stations)):
        ax.plot([stations[i, 0], stations[j, 0]], [stations[i, 1], stations[j, 1]], color="0.82", lw=0.42, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title("stations and hub")
    ax.legend(loc="best", frameon=False, handletextpad=0.15)

    ax = fig.add_subplot(gs[0, 1])
    for wavelength, color, alpha in (("400", "#005f73", 0.50), ("800", "#ee9b00", 0.42)):
        coverage = stats["endpoint_coverage_g_lambda"][wavelength]
        uu = np.asarray(coverage["u"])
        vv = np.asarray(coverage["v"])
        ax.scatter(uu, vv, s=1.15, color=color, alpha=alpha, label=f"{wavelength} nm")
        ax.scatter(-uu, -vv, s=1.15, color=color, alpha=0.62 * alpha)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$u$ (G$\lambda$)")
    ax.set_ylabel(r"$v$ (G$\lambda$)")
    ax.set_title("UV coverage")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1)

    image_axes = []
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(opt.normalize_blr_display(truth), origin="lower", extent=extent, cmap="inferno")
    ax.set_title(f"Input {source.name}\nRM radius {source.blr_radius_uas:.0f} $\\mu$as")
    ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
    ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    image_axes.append(ax)

    labels = {
        "all": "All visibilities + drift",
        "split": "Edge-first closure",
        "direct": "Direct closure-space",
    }
    for col, strategy in enumerate(("all", "split", "direct")):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(opt.normalize_blr_display(images[mode_key][strategy]), origin="lower", extent=extent, cmap="inferno")
        metric = stats["metrics"][mode_key][strategy]
        ax.set_title(f"{labels[strategy]}\nBLR r={metric['blr_corr']:.2f}, all r={metric['global_corr']:.2f}")
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
        image_axes.append(ax)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap="inferno"),
        ax=image_axes,
        fraction=0.024,
        pad=0.018,
    )
    cbar.set_label("norm. brightness\n(BLR-emphasis arcsinh)", fontsize=6.8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    fig.suptitle(f"{case.title}: {source.name}, {TARGET_MODES[mode_key]['label']}", fontsize=10.1, weight="bold", y=0.995)
    safe = f"{case.key}_{source.key}_{mode_key}"
    png = OUTFIG / f"augmented_existing_telescope_{safe}.png"
    pdf = OUTFIG / f"augmented_existing_telescope_{safe}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def run_case(case_stats_path: Path, source: SourceModel) -> dict:
    case = wt.case_from_stats(case_stats_path)
    with patched_source(source):
        bands, base_stats, truth, axis_uas = wt.simulate_bands(case)
    images: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in TARGET_MODES}
    metrics: dict[str, dict[str, dict[str, float]]] = {mode: {} for mode in TARGET_MODES}
    for mode_key in TARGET_MODES:
        for strategy in ("all", "split", "direct"):
            images[mode_key][strategy] = reconstruct_stack(bands, strategy, truth, mode_key=mode_key)
            metrics[mode_key][strategy] = image_metrics(truth, images[mode_key][strategy], axis_uas, source)

    stats = dict(base_stats)
    stats.update(
        {
            "source": {
                "key": source.key,
                "name": source.name,
                "declination_deg": source.dec_deg,
                "effective_ab_mag_550nm": sed_effective_ab_mag(source, 550.0),
                "sed_lambda_nm": list(source.sed_lambda_nm),
                "sed_fnu_mjy": list(source.sed_fnu_mjy),
                "sed_reference": source.sed_reference,
                "tau_hbeta_days": source.tau_hbeta_days,
                "distance_mpc": source.distance_mpc,
                "mbh_msun": source.mbh_msun,
                "blr_radius_uas": source.blr_radius_uas,
                "blr_width_uas": source.blr_width_uas,
                "blr_orbital_period_years": orbital_period_years(source),
                "note": source.note,
            },
            "reconstruction_modes": TARGET_MODES,
            "display_mode": DISPLAY_MODE,
            "metrics": metrics,
        }
    )
    figures = {}
    for mode_key in (DISPLAY_MODE,):
        pdf, png = plot_case(case, source, stats, images, truth, axis_uas, mode_key=mode_key)
        figures[mode_key] = {"pdf": str(pdf), "png": str(png)}
    stats["figures"] = figures
    out_json = OUTFIG / f"augmented_existing_telescope_{case.key}_{source.key}_stats.json"
    out_json.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main() -> None:
    wt.SNR_BOOST = 1.0
    jobs = [
        (OUTFIG / "augmented_existing_telescope_maunakea_plus5_far_stats.json", NGC4151),
        (OUTFIG / "augmented_existing_telescope_ctio_plus4_far_stats.json", NGC3783),
    ]
    summary = {}
    for path, source in jobs:
        print(f"simulating {path.stem} with {source.name}")
        stats = run_case(path, source)
        summary[f"{stats['case']}_{source.key}"] = stats
        print(json.dumps(stats["metrics"], indent=2))
        print(json.dumps(stats["figures"], indent=2))
    out_summary = OUTFIG / "augmented_existing_telescope_ngc_sources_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(out_summary)


if __name__ == "__main__":
    main()
