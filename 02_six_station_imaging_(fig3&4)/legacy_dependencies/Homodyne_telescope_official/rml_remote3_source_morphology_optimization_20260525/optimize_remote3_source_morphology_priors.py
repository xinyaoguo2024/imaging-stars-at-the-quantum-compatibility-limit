from __future__ import annotations

import csv
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import eht_style_amplitude_closure_rml as amp_rml
import hawaii3_compact_case
import optimize_rml_prior_weights_remote4 as prior_scan
import plot_augmented_existing_telescope_ngc_sources as ngc
import plot_prl_broadband_blr_optimized as opt
import plot_prl_broadband_clean as base
import run_rml_validation_pipeline as val


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_source_morphology_optimization_20260525"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = prior_scan.STRATEGIES
BROADBAND_LAMBDA_MIN_NM = float(os.environ.get("MORPH_BROADBAND_LAMBDA_MIN_NM", "400.0"))
BROADBAND_LAMBDA_MAX_NM = float(os.environ.get("MORPH_BROADBAND_LAMBDA_MAX_NM", "800.0"))
BROADBAND_LAMBDA_STEP_NM = float(os.environ.get("MORPH_BROADBAND_LAMBDA_STEP_NM", "10.0"))
NGC4151_HA_REDSHIFT = 0.00332
NGC4151_HA_OBS_NM = 656.3 * (1.0 + NGC4151_HA_REDSHIFT)
NGC4151_HA_FWHM_KM_S = 6000.0
NGC4151_HA_FWHM_NM = 656.3 * NGC4151_HA_FWHM_KM_S / 299792.458
NGC4151_CONTINUUM_FNU_650_JY = 0.067


@dataclass(frozen=True)
class SourceVariant:
    key: str
    label: str
    note: str


VARIANTS = [
    SourceVariant(
        "ngc4151_core_disk_blr_halpha",
        "NGC 4151 core + disk + H-alpha BLR",
        "Host-like plume/halo components are removed; the 600-700 nm band contains a compact core, disk continuum, and a BLR component weighted by the observed broad H-alpha equivalent width.",
    ),
    SourceVariant(
        "broad_plume_irregular_blr",
        "broad plume + irregular BLR",
        "The narrow jet is replaced by a broad bent plume; the BLR has broad asymmetric sectors.",
    ),
    SourceVariant(
        "two_arc_diffuse_tail",
        "two broad BLR arcs + diffuse tail",
        "The BLR is dominated by two resolved arcs and a diffuse tail rather than a thin bright ridge.",
    ),
    SourceVariant(
        "patchy_resolved_ring",
        "patchy resolved BLR ring",
        "Several broad BLR patches are used; all clumps are wider than the nominal high-resolution beam.",
    ),
]


def configure_runtime(*, scan: bool) -> None:
    amp_rml.SOURCE = ngc.NGC4151
    amp_rml.N_RML = 64
    amp_rml.OBSERVING_DAYS = 30
    amp_rml.N_TIME_WINDOWS = 36
    amp_rml.EXPOSURE_S = 600.0
    amp_rml.EXPOSURE_GAP_S = 150.0
    amp_rml.SNR_BOOST = 1.0
    amp_rml.FIBER_LOSS_DB_PER_KM = 0.2
    amp_rml.MODE_FALSE_POSITIVE = 0.05
    amp_rml.PAIR_FALSE_POSITIVE = 0.0
    amp_rml.AMP_SIGMA_MODE = "physical"
    amp_rml.PHASE_FLOOR_RAD = 0.0
    amp_rml.AMP_GRAD_WEIGHT = 0.7
    amp_rml.PHASE_GRAD_WEIGHT = 2.4
    amp_rml.STEP = 0.018
    val.FIT_N_PIX = 32
    val.ADAM_ITER = 650 if scan else 1800
    val.ADAM_LR = 0.012
    val.ADAM_TARGET_AMP_CHI2 = 0.0
    val.ADAM_TARGET_PHASE_CHI2 = 0.0
    val.DISPLAY_SMOOTH_PIX = 1.0
    val.OPTIMIZER = "adam"


def angle_gaussian(theta: np.ndarray, center_deg: float, sigma_deg: float) -> np.ndarray:
    center = np.deg2rad(center_deg)
    sigma = np.deg2rad(sigma_deg)
    return np.exp(-0.5 * (np.angle(np.exp(1j * (theta - center))) / sigma) ** 2)


def normalized(component: np.ndarray) -> np.ndarray:
    out = np.clip(component, 0.0, None)
    total = float(np.sum(out))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones_like(out) / out.size
    return out / total


def curved_plume(
    xp: np.ndarray,
    yp: np.ndarray,
    radius: float,
    *,
    center: float,
    length: float,
    width: float,
    bend: float,
    gate_width: float = 0.10,
) -> np.ndarray:
    curve = yp - bend * radius * np.sin((xp / radius - 0.18) * np.pi)
    gate = 1.0 / (1.0 + np.exp(-(xp - center * radius) / (gate_width * radius)))
    plume = gate * np.exp(-curve**2 / (2.0 * (width * radius) ** 2))
    plume *= np.exp(-((xp - length * radius) ** 2) / (2.0 * (0.52 * radius) ** 2))
    return plume


def make_variant_named_components(
    variant: SourceVariant,
    source: ngc.SourceModel,
    xg: np.ndarray,
    yg: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    r: np.ndarray,
    th: np.ndarray,
    uas: float,
) -> list[tuple[float, np.ndarray]]:
    radius = source.blr_radius_uas * uas
    width = source.blr_width_uas * uas
    compact_core = np.exp(
        -(
            (xp + 0.02 * radius) ** 2 / (2.0 * (0.085 * radius) ** 2)
            + (yp - 0.01 * radius) ** 2 / (2.0 * (0.060 * radius) ** 2)
        )
    )
    crescent_outer = np.exp(
        -(
            (xp - 0.06 * radius) ** 2 / (2.0 * (0.34 * radius) ** 2)
            + (yp + 0.02 * radius) ** 2 / (2.0 * (0.22 * radius) ** 2)
        )
    )
    crescent_hole = 0.70 * np.exp(
        -(
            (xp + 0.14 * radius) ** 2 / (2.0 * (0.23 * radius) ** 2)
            + (yp - 0.01 * radius) ** 2 / (2.0 * (0.15 * radius) ** 2)
        )
    )
    soft_crescent = np.clip(crescent_outer - crescent_hole, 0.0, None)
    diffuse_halo = np.exp(-(xg**2 + yg**2) / (2.0 * (0.95 * radius) ** 2))

    if variant.key == "ngc4151_core_disk_blr_halpha":
        disk_outer = np.exp(
            -(
                (xp - 0.04 * radius) ** 2 / (2.0 * (0.36 * radius) ** 2)
                + (yp + 0.02 * radius) ** 2 / (2.0 * (0.22 * radius) ** 2)
            )
        )
        disk_hole = 0.62 * np.exp(
            -(
                (xp + 0.12 * radius) ** 2 / (2.0 * (0.22 * radius) ** 2)
                + (yp - 0.01 * radius) ** 2 / (2.0 * (0.13 * radius) ** 2)
            )
        )
        soft_crescent = np.clip(disk_outer - disk_hole, 0.0, None)
        disk_stream = curved_plume(xp, yp, radius, center=0.02, length=0.70, width=0.115, bend=0.08, gate_width=0.12)
        disk_counter_stream = curved_plume(
            -xp,
            -yp,
            radius,
            center=0.05,
            length=0.55,
            width=0.120,
            bend=0.05,
            gate_width=0.12,
        )
        disk_knot = np.exp(
            -(
                (xp - 0.78 * radius) ** 2 / (2.0 * (0.17 * radius) ** 2)
                + (yp + 0.04 * radius) ** 2 / (2.0 * (0.13 * radius) ** 2)
            )
        )
        disk_bridge = np.exp(
            -(
                (xp - 0.35 * radius) ** 2 / (2.0 * (0.38 * radius) ** 2)
                + (yp + 0.02 * radius) ** 2 / (2.0 * (0.080 * radius) ** 2)
            )
        )
        disk = (
            0.66 * soft_crescent
            + 0.11 * disk_stream
            + 0.06 * disk_counter_stream
            + 0.025 * disk_knot
            + 0.065 * disk_bridge
        )

        radial_center = radius * (
            1.0
            + 0.070 * np.cos(th - np.deg2rad(112.0))
            - 0.045 * np.sin(2.0 * th + np.deg2rad(25.0))
        )
        width_mod = width * (1.24 + 0.38 * np.cos(th + np.deg2rad(18.0)))
        annulus = np.exp(-((r - radial_center) ** 2) / (2.0 * np.maximum(width_mod, 0.85 * width) ** 2))
        brightness = (
            0.18
            + 1.35 * angle_gaussian(th, 118.0, 38.0)
            + 0.82 * angle_gaussian(th, -156.0, 34.0)
            + 0.45 * angle_gaussian(th, -25.0, 24.0)
            + 0.28 * angle_gaussian(th, 42.0, 28.0)
            + 0.20 * np.cos(3.0 * th + np.deg2rad(35.0))
        )
        clumps = (
            0.80
            * np.exp(
                -(
                    (xp - 0.42 * radius) ** 2 / (2.0 * (0.16 * radius) ** 2)
                    + (yp + 0.55 * radius) ** 2 / (2.0 * (0.13 * radius) ** 2)
                )
            )
            + 0.72
            * np.exp(
                -(
                    (xp + 0.64 * radius) ** 2 / (2.0 * (0.15 * radius) ** 2)
                    + (yp - 0.33 * radius) ** 2 / (2.0 * (0.12 * radius) ** 2)
                )
            )
        )
        diffuse_blr = np.exp(-((r - 0.92 * radius) ** 2) / (2.0 * (1.85 * width) ** 2))
        inner_line_bridge = np.exp(
            -(
                (xp - 0.24 * radius) ** 2 / (2.0 * (0.42 * radius) ** 2)
                + (yp + 0.04 * radius) ** 2 / (2.0 * (0.10 * radius) ** 2)
            )
        )
        blr = (
            0.86 * annulus * np.clip(brightness, 0.08, None)
            + 0.18 * diffuse_blr
            + 0.30 * clumps
            + 0.14 * inner_line_bridge
        )
        return [
            ("core", 0.34, compact_core),
            ("disk", 0.61, disk),
            ("blr", 0.05, blr),
        ]

    if variant.key == "broad_plume_irregular_blr":
        radial_center = radius * (1.0 + 0.055 * np.cos(th - np.deg2rad(115.0)) - 0.035 * np.sin(2.0 * th))
        width_mod = width * (0.95 + 0.25 * np.cos(th - np.deg2rad(20.0)))
        blr = np.exp(-((r - radial_center) ** 2) / (2.0 * np.maximum(width_mod, 0.75 * width) ** 2))
        brightness = (
            0.42
            + 0.95 * angle_gaussian(th, 118.0, 35.0)
            + 0.62 * angle_gaussian(th, -155.0, 28.0)
            + 0.38 * angle_gaussian(th, -22.0, 18.0)
            + 0.25 * np.cos(3.0 * th + np.deg2rad(35.0))
        )
        blr *= np.clip(brightness, 0.06, None)
        plume = curved_plume(xp, yp, radius, center=0.02, length=0.55, width=0.115, bend=0.18)
        knot = np.exp(-((xp - 0.72 * radius) ** 2 / (2.0 * (0.12 * radius) ** 2) + (yp + 0.06 * radius) ** 2 / (2.0 * (0.10 * radius) ** 2)))
        return [
            ("core", 0.20, compact_core),
            ("continuum", 0.18, soft_crescent),
            ("blr", 0.43, blr),
            ("plume", 0.14, 0.75 * plume + 0.25 * knot),
            ("halo", 0.05, diffuse_halo),
        ]

    if variant.key == "two_arc_diffuse_tail":
        radial_center = radius * (1.0 + 0.030 * np.sin(th + np.deg2rad(15.0)))
        blr_base = np.exp(-((r - radial_center) ** 2) / (2.0 * (1.10 * width) ** 2))
        arcs = 0.30 + 1.05 * angle_gaussian(th, 138.0, 42.0) + 0.86 * angle_gaussian(th, -38.0, 33.0)
        arcs += 0.28 * angle_gaussian(th, -172.0, 24.0)
        blr = blr_base * arcs
        tail = curved_plume(xp, yp, radius, center=0.05, length=0.62, width=0.16, bend=0.10)
        broad_knot = np.exp(-((xp - 0.63 * radius) ** 2 / (2.0 * (0.18 * radius) ** 2) + (yp - 0.04 * radius) ** 2 / (2.0 * (0.13 * radius) ** 2)))
        return [
            ("core", 0.22, compact_core),
            ("continuum", 0.15, soft_crescent),
            ("blr", 0.48, blr),
            ("plume", 0.10, 0.65 * tail + 0.35 * broad_knot),
            ("halo", 0.05, diffuse_halo),
        ]

    if variant.key == "patchy_resolved_ring":
        radial_center = radius * (1.0 + 0.040 * np.cos(th - np.deg2rad(40.0)) + 0.025 * np.sin(2.0 * th))
        blr = np.exp(-((r - radial_center) ** 2) / (2.0 * (0.95 * width) ** 2))
        patches = (
            0.20
            + 1.15 * angle_gaussian(th, 132.0, 22.0)
            + 0.92 * angle_gaussian(th, -148.0, 25.0)
            + 0.78 * angle_gaussian(th, -35.0, 20.0)
            + 0.62 * angle_gaussian(th, 42.0, 26.0)
        )
        blr *= patches
        plume = curved_plume(xp, yp, radius, center=0.10, length=0.45, width=0.13, bend=0.12)
        return [
            ("core", 0.18, compact_core),
            ("continuum", 0.16, soft_crescent),
            ("blr", 0.52, blr),
            ("plume", 0.09, plume),
            ("halo", 0.05, diffuse_halo),
        ]

    raise ValueError(f"Unknown source variant: {variant.key}")


def make_variant_components(
    variant: SourceVariant,
    source: ngc.SourceModel,
    xg: np.ndarray,
    yg: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    r: np.ndarray,
    th: np.ndarray,
    uas: float,
) -> list[tuple[float, np.ndarray]]:
    return [
        (fraction, component)
        for _name, fraction, component in make_variant_named_components(variant, source, xg, yg, xp, yp, r, th, uas)
    ]


COMPONENT_SPECTRAL_MODEL_NOTE = (
    "component-resolved NGC4151 SED: for the final core+disk+BLR H-alpha source, "
    "core and disk receive only smooth optical continuum, while the broad H-alpha "
    "excess over that continuum is assigned to the BLR component"
)


def variant_component_fractions(variant: SourceVariant) -> list[tuple[str, float]]:
    if variant.key == "ngc4151_core_disk_blr_halpha":
        return [("core", 0.34), ("disk", 0.61), ("blr", 0.05)]
    if variant.key == "broad_plume_irregular_blr":
        return [("core", 0.20), ("continuum", 0.18), ("blr", 0.43), ("plume", 0.14), ("halo", 0.05)]
    if variant.key == "two_arc_diffuse_tail":
        return [("core", 0.22), ("continuum", 0.15), ("blr", 0.48), ("plume", 0.10), ("halo", 0.05)]
    if variant.key == "patchy_resolved_ring":
        return [("core", 0.18), ("continuum", 0.16), ("blr", 0.52), ("plume", 0.09), ("halo", 0.05)]
    raise ValueError(f"Unknown source variant: {variant.key}")


def component_spectral_scale(component_name: str, wavelength_nm: float) -> float:
    lam = float(wavelength_nm)
    hbeta = np.exp(-0.5 * ((lam - 486.1) / 18.0) ** 2)
    halpha = np.exp(-0.5 * ((lam - 656.3) / 22.0) ** 2)
    if component_name == "blr":
        return float(0.45 + 0.95 * hbeta + 1.55 * halpha)
    if component_name == "core":
        return float((lam / 550.0) ** -0.35)
    if component_name == "disk":
        return float((lam / 650.0) ** -0.10)
    if component_name == "continuum":
        return float((lam / 550.0) ** -0.05)
    if component_name == "plume":
        return float((lam / 550.0) ** -0.25)
    if component_name == "halo":
        return float((lam / 550.0) ** 0.35)
    return 1.0


def total_source_fnu_jy(source: ngc.SourceModel, wavelength_nm: float) -> float:
    sed_lambda = np.asarray(source.sed_lambda_nm, dtype=float)
    sed_fnu_jy = np.asarray(source.sed_fnu_mjy, dtype=float) * 1.0e-3
    return float(
        np.exp(
            np.interp(
                np.log(float(wavelength_nm)),
                np.log(sed_lambda),
                np.log(sed_fnu_jy),
                left=np.log(sed_fnu_jy[0]),
                right=np.log(sed_fnu_jy[-1]),
            )
        )
    )


def ngc4151_halpha_continuum_fnu_jy(wavelength_nm: float) -> float:
    """Smooth nuclear continuum used to separate broad H-alpha from continuum."""
    lam = float(wavelength_nm)
    return float(NGC4151_CONTINUUM_FNU_650_JY * (lam / 650.0) ** -0.10)


def final_halpha_component_names(component_fractions: list[tuple[str, float]]) -> bool:
    return {name for name, _fraction in component_fractions} == {"core", "disk", "blr"}


def component_flux_fnu_jy(
    component_fractions: list[tuple[str, float]],
    source: ngc.SourceModel | None,
    wavelength_nm: float,
) -> dict[str, float]:
    lam = float(wavelength_nm)
    if source is not None and final_halpha_component_names(component_fractions):
        total = max(total_source_fnu_jy(source, lam), 1.0e-300)
        continuum = min(total, ngc4151_halpha_continuum_fnu_jy(lam))
        line = max(total - continuum, 0.0)
        continuum_weights = {
            "core": 0.34 * (lam / 650.0) ** -0.20,
            "disk": 0.61 * (lam / 650.0) ** 0.05,
            "blr": 0.05,
        }
        norm = sum(continuum_weights.values())
        fluxes = {name: continuum * weight / norm for name, weight in continuum_weights.items()}
        fluxes["blr"] += line
        return fluxes

    total = total_source_fnu_jy(source, lam) if source is not None else 1.0
    raw = {
        name: max(float(fraction) * component_spectral_scale(name, lam), 1.0e-300)
        for name, fraction in component_fractions
    }
    raw_total = sum(raw.values())
    return {name: total * value / raw_total for name, value in raw.items()}


def component_flux_fractions(
    component_fractions: list[tuple[str, float]],
    wavelength_nm: float,
    source: ngc.SourceModel | None = None,
) -> dict[str, float]:
    fluxes = component_flux_fnu_jy(component_fractions, source, wavelength_nm)
    total = sum(fluxes.values())
    return {name: value / total for name, value in fluxes.items()}


def broadband_component_flux_fractions(component_fractions: list[tuple[str, float]], source: ngc.SourceModel) -> dict[str, float]:
    accum = {name: 0.0 for name, _fraction in component_fractions}
    total_weight = 0.0
    edges_nm = np.arange(
        BROADBAND_LAMBDA_MIN_NM,
        BROADBAND_LAMBDA_MAX_NM + 0.5 * BROADBAND_LAMBDA_STEP_NM,
        BROADBAND_LAMBDA_STEP_NM,
    )
    edges_nm[-1] = BROADBAND_LAMBDA_MAX_NM
    for lo_nm, hi_nm in zip(edges_nm[:-1], edges_nm[1:]):
        center_nm = float(np.sqrt(lo_nm * hi_nm))
        freq = base.C_LIGHT / (center_nm * 1.0e-9)
        freq_lo = base.C_LIGHT / (hi_nm * 1.0e-9)
        freq_hi = base.C_LIGHT / (lo_nm * 1.0e-9)
        photon_weight = total_source_fnu_jy(source, center_nm) * (freq_hi - freq_lo) / freq
        fractions = component_flux_fractions(component_fractions, center_nm, source=source)
        for name, value in fractions.items():
            accum[name] += photon_weight * value
        total_weight += photon_weight
    return {name: value / total_weight for name, value in accum.items()}


def image_from_named_components(
    named_components: list[tuple[str, float, np.ndarray]],
    flux_fractions: dict[str, float],
) -> np.ndarray:
    image = np.zeros_like(named_components[0][2])
    for name, _base_fraction, component in named_components:
        image += flux_fractions[name] * normalized(component)
    return normalized(image)


def source_factory_for_variant(variant: SourceVariant) -> Callable[[ngc.SourceModel], Callable[[int, float], tuple[np.ndarray, np.ndarray]]]:
    def make_source_factory(source: ngc.SourceModel):
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
            components = make_variant_named_components(variant, source, xg, yg, xp, yp, r, th, uas)
            fractions = broadband_component_flux_fractions(variant_component_fractions(variant), source)
            return image_from_named_components(components, fractions), x / base.UAS_TO_RAD

        return make_source

    return make_source_factory


def wavelength_source_factory_for_variant(
    variant: SourceVariant,
) -> Callable[[ngc.SourceModel], Callable[[int, float, float], tuple[np.ndarray, np.ndarray]]]:
    def make_wavelength_source_factory(source: ngc.SourceModel):
        base_component_fractions = variant_component_fractions(variant)

        def make_source_at_wavelength_nm(
            n: int,
            half_width_uas: float,
            wavelength_nm: float,
        ) -> tuple[np.ndarray, np.ndarray]:
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
            components = make_variant_named_components(variant, source, xg, yg, xp, yp, r, th, uas)
            fractions = component_flux_fractions(base_component_fractions, wavelength_nm, source=source)
            return image_from_named_components(components, fractions), x / base.UAS_TO_RAD

        make_source_at_wavelength_nm.component_flux_fractions = lambda wavelength_nm: component_flux_fractions(
            base_component_fractions,
            wavelength_nm,
            source=source,
        )
        make_source_at_wavelength_nm.broadband_component_flux_fractions = lambda: broadband_component_flux_fractions(
            base_component_fractions,
            source,
        )
        make_source_at_wavelength_nm.spectral_model_note = COMPONENT_SPECTRAL_MODEL_NOTE
        return make_source_at_wavelength_nm

    return make_wavelength_source_factory


@contextmanager
def patched_variant(variant: SourceVariant):
    old_factory = ngc.make_source_factory
    old_wavelength_factory = ngc.make_wavelength_source_factory
    old_morph = ngc.SOURCE_MORPHOLOGY
    ngc.make_source_factory = source_factory_for_variant(variant)
    ngc.make_wavelength_source_factory = wavelength_source_factory_for_variant(variant)
    ngc.SOURCE_MORPHOLOGY = variant.key
    try:
        yield
    finally:
        ngc.make_source_factory = old_factory
        ngc.make_wavelength_source_factory = old_wavelength_factory
        ngc.SOURCE_MORPHOLOGY = old_morph


def angular_profile(image: np.ndarray, axis_uas: np.ndarray, n_bin: int = 72):
    source = ngc.NGC4151
    xx, yy = np.meshgrid(axis_uas, axis_uas)
    rr = np.sqrt(xx * xx + yy * yy)
    theta = np.arctan2(yy, xx)
    half_width = max(2.2 * source.blr_width_uas, 10.0)
    mask = (rr > source.blr_radius_uas - half_width) & (rr < source.blr_radius_uas + half_width)
    bins = np.linspace(-np.pi, np.pi, n_bin + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    prof = np.zeros(n_bin, dtype=float)
    for k, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = mask & (theta >= lo) & (theta < hi)
        prof[k] = float(np.mean(image[m])) if np.any(m) else 0.0
    prof /= max(float(np.mean(prof[prof > 0])), 1e-30) if np.any(prof > 0) else 1.0
    return centers, prof


def profile_rmse(truth: np.ndarray, image: np.ndarray, axis_uas: np.ndarray) -> float:
    _, truth_prof = angular_profile(truth, axis_uas)
    _, image_prof = angular_profile(image, axis_uas)
    return float(np.sqrt(np.mean((image_prof - truth_prof) ** 2)))


def scan_config_rows(variant: SourceVariant, data: dict):
    rows = []
    for idx, cfg in enumerate(prior_scan.make_configs(), start=1):
        print(f"[scan {variant.key} {idx}/33] {cfg.key}", flush=True)
        with patched_variant(variant):
            row = prior_scan.run_single_direct_scan(
                cfg,
                case=data["case"],
                bands=data["bands"],
                truth=data["truth"],
                axis_uas=data["axis_uas"],
                prior=data["prior"],
                starts=data["starts"],
            )
        clean = {key: value for key, value in row.items() if key != "result"}
        clean["variant"] = variant.key
        clean["variant_label"] = variant.label
        rows.append(clean)
    best = min(rows, key=lambda row: row["score"])
    cfg = prior_scan.PriorConfig(str(best["config"]), float(best["prior"]), float(best["tv"]), float(best["entropy"]))
    return cfg, rows


def simulate_variant(variant: SourceVariant):
    configure_runtime(scan=True)
    case = hawaii3_compact_case.make_hawaii3_compact_remote_case()
    with patched_variant(variant):
        print(f"[simulate] {variant.key}", flush=True)
        bands, stats, truth, axis_uas = amp_rml.simulate_case(case)
    prior_full = amp_rml.broad_gaussian_prior(axis_uas)
    prior = val.rebin_image_average(prior_full, val.FIT_N_PIX)
    starts = val.build_starts(bands, truth, prior_full)
    return {
        "case": case,
        "bands": bands,
        "stats": stats,
        "truth": truth,
        "axis_uas": axis_uas,
        "prior": prior,
        "starts": starts,
    }


def final_strategy_rows(best_cfg: prior_scan.PriorConfig, results: list[dict], truth: np.ndarray, axis_uas: np.ndarray, variant: SourceVariant):
    rows = prior_scan.final_rows(best_cfg, results, truth, axis_uas)
    for row in rows:
        row["variant"] = variant.key
        row["variant_label"] = variant.label
    return rows


def evaluate_variant(variant: SourceVariant, data: dict, cfg: prior_scan.PriorConfig):
    with patched_variant(variant):
        results = prior_scan.evaluate_all_strategies(
            cfg,
            case=data["case"],
            bands=data["bands"],
            truth=data["truth"],
            axis_uas=data["axis_uas"],
            prior=data["prior"],
            starts=data["starts"],
        )
    return results


def plot_variant_final(variant: SourceVariant, data: dict, cfg: prior_scan.PriorConfig, results: list[dict]):
    axis = data["axis_uas"]
    truth = data["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in results}
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.25), constrained_layout=True)
    panels = [("truth", "Input source", truth)] + [
        (strategy, label, result_by[strategy]["best"]["image"]) for strategy, label, *_ in STRATEGIES
    ]
    for col, (strategy, label, image) in enumerate(panels):
        ax = axes[0, col]
        ax.imshow(opt.normalize_blr_display(image), origin="lower", extent=extent, cmap="inferno")
        if strategy == "truth":
            ax.set_title(label)
        else:
            m = result_by[strategy]["best"]["metrics"]
            r = result_by[strategy]["best"]["residuals"]
            ax.set_title(
                f"{label}\nBLR r={m['blr_corr']:.3f}, all r={m['global_corr']:.3f}\n"
                rf"$\chi_A^2$={r['amp_reduced_chi2']:.2f}, $\chi_\phi^2$={r['phase_reduced_chi2']:.2f}",
                fontsize=8.0,
            )
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        if col == 0:
            ax.set_ylabel(r"$\Delta\delta$ ($\mu$as)")
    theta, truth_prof = angular_profile(truth, axis)
    axes[1, 0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    for strategy, label, _start, color in STRATEGIES:
        _, prof = angular_profile(result_by[strategy]["best"]["image"], axis)
        axes[1, 0].plot(np.rad2deg(theta), prof, color=color, lw=1.5, label=label)
    axes[1, 0].set_title("BLR annular profile")
    axes[1, 0].set_xlabel("azimuth angle (deg)")
    axes[1, 0].set_ylabel("mean-normalized brightness")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=7)
    truth_disp = opt.normalize_blr_display(truth)
    for col, (strategy, label, *_rest) in enumerate(STRATEGIES, start=1):
        ax = axes[1, col]
        residual = opt.normalize_blr_display(result_by[strategy]["best"]["image"]) - truth_disp
        vmax = max(0.08, float(np.percentile(np.abs(residual), 99.0)))
        im = ax.imshow(residual, origin="lower", extent=extent, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{label} residual", fontsize=8.0)
        ax.set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{variant.label}: optimized prior p={cfg.prior:g}, TV={cfg.tv:g}, e={cfg.entropy:g}",
        weight="bold",
    )
    png = OUT / f"{variant.key}_optimized_strategy_comparison.png"
    pdf = OUT / f"{variant.key}_optimized_strategy_comparison.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_summary(rows: list[dict]):
    variants = [variant.key for variant in VARIANTS]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=True)
    metrics = [("blr_corr", "BLR correlation", "higher is better"), ("profile_rmse", "profile RMSE", "lower is better"), ("phase_chi2", "phase chi-square", "lower is better")]
    colors = {"all": "#8d99ae", "split": "#0077b6", "direct": "#d00000"}
    x = np.arange(len(variants))
    width = 0.23
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for offset, strategy in zip((-width, 0.0, width), ("all", "split", "direct")):
            vals = [float(next(row[metric] for row in rows if row["variant"] == variant and row["strategy"] == strategy)) for variant in variants]
            ax.bar(x + offset, vals, width=width, color=colors[strategy], alpha=0.86, label=strategy)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Remote3 source-morphology benchmark after per-source prior optimization", weight="bold")
    png = OUT / "source_variant_metric_summary.png"
    pdf = OUT / "source_variant_metric_summary.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    all_scan_rows: list[dict] = []
    all_final_rows: list[dict] = []
    best_priors: dict[str, dict] = {}
    figures: dict[str, dict] = {}
    for variant in VARIANTS:
        data = simulate_variant(variant)
        cfg, scan_rows = scan_config_rows(variant, data)
        best_priors[variant.key] = {"config": cfg.key, "prior": cfg.prior, "tv": cfg.tv, "entropy": cfg.entropy}
        all_scan_rows.extend(scan_rows)
        print(f"[best {variant.key}] {cfg}", flush=True)
        results = evaluate_variant(variant, data, cfg)
        final_rows = final_strategy_rows(cfg, results, data["truth"], data["axis_uas"], variant)
        all_final_rows.extend(final_rows)
        pdf, png = plot_variant_final(variant, data, cfg, results)
        figures[variant.key] = {"png": str(png), "pdf": str(pdf)}

    summary_pdf, summary_png = plot_summary(all_final_rows)
    scan_csv = OUT / "source_variant_prior_scan_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_scan_rows)
    final_csv = OUT / "source_variant_optimized_strategy_metrics.csv"
    with final_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_final_rows)

    payload = {
        "runtime": {
            "case": hawaii3_compact_case.make_hawaii3_compact_remote_case().key,
            "observing_days": amp_rml.OBSERVING_DAYS,
            "n_time_windows": amp_rml.N_TIME_WINDOWS,
            "exposure_s": amp_rml.EXPOSURE_S,
            "snr_boost": amp_rml.SNR_BOOST,
            "fiber_loss_db_per_km": amp_rml.FIBER_LOSS_DB_PER_KM,
            "mode_false_positive": amp_rml.MODE_FALSE_POSITIVE,
            "amp_weight": amp_rml.AMP_GRAD_WEIGHT,
            "phase_weight": amp_rml.PHASE_GRAD_WEIGHT,
            "fit_n_pix": val.FIT_N_PIX,
            "shown_n_pix": amp_rml.N_RML,
        },
        "variants": [variant.__dict__ for variant in VARIANTS],
        "best_priors": best_priors,
        "final_rows": all_final_rows,
        "figures": {**figures, "summary_png": str(summary_png), "summary_pdf": str(summary_pdf), "scan_csv": str(scan_csv), "final_csv": str(final_csv)},
    }
    json_path = OUT / "source_variant_optimization_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Remote3 source morphology optimization\n\n"
        "This diagnostic keeps the observing setup fixed at remote3, 36 x 10 min, SNR boost 1,\n"
        "and 0.2 dB/km fiber loss.  It tests three modified AGN input morphologies with broader,\n"
        "resolved tails and less sinusoidal BLR angular structure.  Each source variant gets an\n"
        "independent RML prior-weight scan before all/edge/direct are compared.\n"
    )
    print(summary_png)
    print(final_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
