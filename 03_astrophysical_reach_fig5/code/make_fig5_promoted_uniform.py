#!/usr/bin/env python3
"""Make the promoted same-POVM redshift--magnitude baseline plot.

The plotted phase target is one oriented triangle closure functional.  This is
a phase-only state-of-the-art comparison: the full receiver budget is assigned
to the phase POVM, with no amplitude/phase split.  The Fisher expressions below
are per temporal-mode copy.  M_TOTAL counts accumulated temporal-mode copies
(total time-frequency modes), while the promoted expression is the per-copy
asymptotic Fisher information of a joint receiver with depth n_s -> infinity.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "generated_outputs"
RESULT_DIR = ROOT / "generated_outputs"
FIGURE_STEM = os.environ.get("FIG5_FIGURE_STEM", "fig5_astrophysical_reach")
PROMOTED_CURVE_LABEL = os.environ.get(
    "FIG5_PROMOTED_CURVE_LABEL", "promoted same-POVM"
)
EDGE_CURVE_LABEL = os.environ.get(
    "FIG5_EDGE_CURVE_LABEL", "single-copy uniform edge-first measurement"
)

AXIS_LABEL_FONTSIZE = 12.0
TICK_FONTSIZE = 10.8
LEGEND_FONTSIZE = 9.0
SOURCE_LABEL_FONTSIZE = 8.5
COLORBAR_FONTSIZE = 11.2

# Requested spatial-array, mode-count, and receiver parameters.
N_STATION = 20
M_TOTAL = 1.0e11
NU = 0.50
ETA = 0.20
EPSILON_BG = 1.0e-11
D_TEL = 10.0
WAVELENGTH = 650.0e-9
KAPPA_CL = 0.50
PHASE_BUDGET_FRACTION = 1.0

EDGE_DIMENSION = N_STATION * (N_STATION - 1) // 2
CLOSURE_DIMENSION = (N_STATION - 1) * (N_STATION - 2) // 2

MASS_REFERENCE_MSUN = 1.0e8
MAG_MIN = 10.0
MAG_MAX = 21.0
BASELINE_MIN_KM = 1.0e2
BASELINE_COLORBAR_MAX_KM = 1.0e5
BASELINE_CONTOUR_LEVELS_KM = np.array([1.0e2, 1.0e3, 1.0e4])
BASELINE_CONTOUR_COLORS = {
    1.0e2: "#2ca02c",
    1.0e3: "#f6c85f",
    1.0e4: "#7bdff2",
}

# Physical constants and the same lightweight flat-LCDM cosmology as the
# baseline source-scaling script.
C_LIGHT = 299_792_458.0
H_PLANCK = 6.62607015e-34
G_NEWTON = 6.67430e-11
M_SUN = 1.98847e30
PC = 3.085677581491367e16
MPC = 1.0e6 * PC
FNU0_AB = 3631.0e-26
H0 = 70.0 * 1000.0 / MPC
OMEGA_M = 0.30
OMEGA_L = 0.70


@dataclass(frozen=True)
class SourceMarker:
    name: str
    kind: str
    z: float
    mag_ab: float
    mbh_msun: float
    mbh_err_minus_msun: float
    mbh_err_plus_msun: float
    mass_reference: str


def source_from_log_mass(
    name: str,
    kind: str,
    z: float,
    mag_ab: float,
    log10_mass: float,
    log_err_minus: float,
    log_err_plus: float,
    mass_reference: str,
) -> SourceMarker:
    """Convert a quoted logarithmic mass interval into asymmetric linear errors."""
    mass = 10.0**log10_mass
    mass_low = 10.0 ** (log10_mass - log_err_minus)
    mass_high = 10.0 ** (log10_mass + log_err_plus)
    return SourceMarker(
        name,
        kind,
        z,
        mag_ab,
        mass,
        mass - mass_low,
        mass_high - mass,
        mass_reference,
    )


# Literature benchmark masses.  The displayed uncertainties are those quoted
# by the selected mass determination (or, for OJ 287, the conservative 1%
# model uncertainty quoted for the orbital solution); they should not be read
# as a homogeneous error model across the heterogeneous source classes.
SOURCES = [
    SourceMarker(
        "NGC 1068", "Seyfert/torus", 0.0038, 10.0,
        16.6e6, 0.1e6, 0.1e6, "Gallimore et al., revised H2O-maser mass"
    ),
    SourceMarker(
        "NGC 4151", "Seyfert/BLR", 0.0033, 11.5,
        4.57e7, 0.47e7, 0.57e7, "Bentz et al. 2006, reverberation mapping"
    ),
    SourceMarker(
        "NGC 1275", "radio AGN/jet", 0.0176, 12.6,
        1.1e9, 0.5e9, 0.9e9, "Riffel et al. 2020, M-sigma estimate"
    ),
    source_from_log_mass(
        "Mrk 421", "blazar/jet", 0.031, 13.0,
        8.28, 0.11, 0.11, "Barth et al. 2003, stellar velocity dispersion"
    ),
    source_from_log_mass(
        "Mrk 501", "blazar/jet", 0.034, 13.9,
        9.21, 0.13, 0.13, "Barth et al. 2003, stellar velocity dispersion"
    ),
    source_from_log_mass(
        "BL Lac", "blazar/jet", 0.069, 14.0,
        8.23, 0.30, 0.30,
        "Woo and Urry 2002; 0.30-dex M-sigma scatter"
    ),
    SourceMarker(
        "3C 273", "quasar/jet", 0.158, 12.9,
        2.6e8, 1.1e8, 1.1e8,
        "GRAVITY Collaboration 2018, spatially resolved BLR dynamics"
    ),
    SourceMarker(
        "OJ 287", "blazar/binary", 0.306, 14.5,
        1.84e10, 0.0184e10, 0.0184e10, "Valtonen et al. orbital solution"
    ),
    SourceMarker(
        "PKS 1510-089", "blazar/jet", 0.361, 16.5,
        5.71e7, 0.58e7, 0.62e7, "Rakshit 2020, reverberation mapping"
    ),
    source_from_log_mass(
        "3C 279", "blazar/jet", 0.536, 17.8,
        8.90, 0.50, 0.50, "Nilsson et al. 2009, host luminosity"
    ),
]


def build_cosmology_grid(z_max: float = 12.0, n: int = 40_000) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(0.0, z_max, n)
    inv_e = 1.0 / np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)
    dz = np.diff(z)
    integral = np.zeros_like(z)
    integral[1:] = np.cumsum(0.5 * (inv_e[:-1] + inv_e[1:]) * dz)
    d_comoving = (C_LIGHT / H0) * integral
    return z, d_comoving / (1.0 + z)


Z_COSMOLOGY, DA_COSMOLOGY = build_cosmology_grid()


def angular_diameter_distance(z: np.ndarray | float) -> np.ndarray:
    return np.interp(z, Z_COSMOLOGY, DA_COSMOLOGY)


def ideal_occupation_from_ab_mag(mag_ab: np.ndarray | float) -> np.ndarray:
    area = math.pi * (D_TEL / 2.0) ** 2
    optical_frequency = C_LIGHT / WAVELENGTH
    fnu = FNU0_AB * 10.0 ** (-0.4 * np.asarray(mag_ab, dtype=float))
    return area * fnu / (H_PLANCK * optical_frequency)


def fisher_terms(mag_ab: np.ndarray | float) -> dict[str, np.ndarray]:
    """Return the phase-only per-temporal-mode-copy Fisher quantities."""
    u = ideal_occupation_from_ab_mag(mag_ab)
    a = ETA * u
    s0 = a + EPSILON_BG
    c = a * NU
    f_edge = 2.0 * c**2 / (3.0 * (N_STATION - 1.0) * s0)
    f_prom = 2.0 * c**2 / (3.0 * (s0 - c))
    return {"u": u, "a": a, "s0": s0, "c": c, "F_edge": f_edge, "F_prom": f_prom}


def closure_phase_sigma(mag_ab: np.ndarray | float, strategy: str) -> np.ndarray:
    values = fisher_terms(mag_ab)
    key = {"promoted": "F_prom", "edge": "F_edge"}[strategy]
    fisher = np.asarray(values[key])
    return 1.0 / np.sqrt(M_TOTAL * fisher)


def magnitude_for_closure_sigma(target_sigma: float, strategy: str) -> float:
    lo, hi = -5.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(closure_phase_sigma(mid, strategy)) > target_sigma:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def schwarzschild_radius(m_bh_msun: float) -> float:
    return 2.0 * G_NEWTON * m_bh_msun * M_SUN / C_LIGHT**2


def required_baseline_km(
    z: np.ndarray | float,
    mag_ab: np.ndarray | float,
    strategy: str,
    m_bh_msun: float = MASS_REFERENCE_MSUN,
) -> np.ndarray:
    theta_rs = schwarzschild_radius(m_bh_msun) / np.maximum(angular_diameter_distance(z), 1.0)
    sigma_phi = closure_phase_sigma(mag_ab, strategy)
    baseline_m = sigma_phi * WAVELENGTH / (2.0 * math.pi * KAPPA_CL * theta_rs)
    return baseline_m / 1000.0


def source_style(src: SourceMarker) -> tuple[str, str, str]:
    kind = src.kind.lower()
    if "seyfert" in kind:
        return "#d62728", "o", "Seyfert"
    if "quasar" in kind:
        return "#9467bd", "D", "quasar"
    if "radio agn" in kind:
        return "#ff7f0e", "P", "radio AGN"
    return "#1f77b4", "s", "blazar/jet"


def format_alpha(alpha: float) -> str:
    if alpha < 1.0:
        return f"{alpha:.2f}"
    exponent = int(math.floor(math.log10(alpha)))
    if exponent < 2:
        decimal_places = max(0, 1 - exponent)
        return f"{alpha:.{decimal_places}f}"
    mantissa = alpha / 10.0**exponent
    return rf"{mantissa:.1f}\times10^{{{exponent}}}"


def format_alpha_error(error: float) -> str:
    """Use two readable digits, retaining sub-0.01 uncertainties when needed."""
    if error <= 0.0:
        return "0"
    if error < 0.01:
        decimals = max(3, int(math.ceil(-math.log10(error))))
        return f"{error:.{decimals}f}"
    if error < 1.0:
        return f"{error:.2f}"
    return format_alpha(error)


def format_lmin_value(l_km: float) -> str:
    if l_km < 10.0:
        return f"{l_km:.1f}"
    if l_km < 1000.0:
        return f"{l_km:.0f}"
    return f"{l_km:.0f}"


def format_lmin(l_km: float) -> str:
    return rf"{format_lmin_value(l_km)}\,{{\rm km}}"


def format_lmin_error(error_km: float) -> str:
    if error_km < 1.0:
        return f"{error_km:.1f}"
    return f"{error_km:.0f}"


def style_axes(ax: mpl.axes.Axes) -> None:
    ax.tick_params(direction="out", length=4.5, width=0.9, labelsize=TICK_FONTSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_contour_path_effects(contour_set: mpl.contour.ContourSet, stroke_width: float) -> None:
    contour_set.set_path_effects(
        [pe.Stroke(linewidth=stroke_width, foreground="black", alpha=0.45), pe.Normal()]
    )


def add_baseline_contours(
    ax: mpl.axes.Axes,
    z_grid: np.ndarray,
    mag_grid: np.ndarray,
    promoted_values: np.ndarray,
    edge_values: np.ndarray,
) -> list[mpl.lines.Line2D]:
    handles: list[mpl.lines.Line2D] = []
    for level in BASELINE_CONTOUR_LEVELS_KM:
        color = BASELINE_CONTOUR_COLORS[level]
        promoted = ax.contour(
            z_grid,
            mag_grid,
            promoted_values,
            levels=[level],
            colors=[color],
            linewidths=1.25,
            linestyles="-",
            alpha=0.98,
            zorder=4,
        )
        add_contour_path_effects(promoted, 2.8)
        edge = ax.contour(
            z_grid,
            mag_grid,
            edge_values,
            levels=[level],
            colors=[color],
            linewidths=1.15,
            linestyles="--",
            alpha=0.98,
            zorder=4,
        )
        add_contour_path_effects(edge, 2.6)
        handles.append(
            mpl.lines.Line2D([0], [0], color=color, lw=1.55, label=f"{int(level):d} km")
        )
    return handles


def add_sigma_pi_limits(ax: mpl.axes.Axes) -> tuple[float, float]:
    m_prom = magnitude_for_closure_sigma(math.pi, "promoted")
    m_edge = magnitude_for_closure_sigma(math.pi, "edge")
    specs = [
        (m_prom, "-", r"collective $\sigma_{\Phi_{\rm cl}}=\pi$", 0.63),
        (m_edge, (0, (5, 2)), r"single-copy $\sigma_{\Phi_{\rm cl}}=\pi$", 0.63),
    ]
    blended = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for mag_limit, linestyle, label, x_fraction in specs:
        ax.axhline(mag_limit, color="black", lw=1.25, ls=linestyle, alpha=0.95, zorder=5)
        if MAG_MIN < mag_limit < MAG_MAX:
            ax.text(
                x_fraction,
                mag_limit - 0.22,
                label,
                transform=blended,
                fontsize=SOURCE_LABEL_FONTSIZE,
                ha="center",
                va="bottom",
                color="black",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.66),
                zorder=8,
            )
    return m_prom, m_edge


def add_source_markers(ax: mpl.axes.Axes) -> None:
    # Place every annotation on a cardinal direction from its own source marker.
    # Entries are (x offset [pt], y offset [pt], horizontal alignment,
    # vertical alignment).  Marker/label separation takes priority over
    # avoiding the contour curves.
    label_placements = {
        "NGC 1068": (-8, 0, "right", "top"),
        "NGC 4151": (8, 0, "left", "center"),
        "NGC 1275": (0, 8, "center", "bottom"),
        "Mrk 421": (8, 0, "left", "center"),
        "Mrk 501": (-8, 0, "right", "center"),
        "BL Lac": (8, 0, "left", "center"),
        "3C 273": (8, 0, "left", "center"),
        "OJ 287": (8, 0, "left", "center"),
        "PKS 1510-089": (-8, 0, "right", "center"),
        "3C 279": (-8, 0, "right", "center"),
    }
    for src in SOURCES:
        color, marker, _ = source_style(src)
        y_plot = max(MAG_MIN + 0.18, min(MAG_MAX - 0.18, src.mag_ab))
        ax.scatter(
            src.z,
            y_plot,
            s=46,
            c=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )
        l_req = float(required_baseline_km(src.z, src.mag_ab, "promoted", src.mbh_msun))
        l_req_mass_low = float(
            required_baseline_km(
                src.z,
                src.mag_ab,
                "promoted",
                src.mbh_msun - src.mbh_err_minus_msun,
            )
        )
        l_req_mass_high = float(
            required_baseline_km(
                src.z,
                src.mag_ab,
                "promoted",
                src.mbh_msun + src.mbh_err_plus_msun,
            )
        )
        l_err_minus = l_req - l_req_mass_high
        l_err_plus = l_req_mass_low - l_req
        alpha = src.mbh_msun / MASS_REFERENCE_MSUN
        alpha_err_minus = src.mbh_err_minus_msun / MASS_REFERENCE_MSUN
        alpha_err_plus = src.mbh_err_plus_msun / MASS_REFERENCE_MSUN
        label = (
            f"{src.name}\n"
            + rf"$\alpha={{{format_alpha(alpha)}}}"
            + rf"^{{+{format_alpha_error(alpha_err_plus)}}}"
            + rf"_{{-{format_alpha_error(alpha_err_minus)}}}$"
            + "\n"
            + rf"$L_{{\min}}={format_lmin_value(l_req)}"
            + rf"^{{+{format_lmin_error(l_err_plus)}}}"
            + rf"_{{-{format_lmin_error(l_err_minus)}}}\,{{\rm km}}$"
        )
        dx, dy, horizontal_alignment, vertical_alignment = label_placements[src.name]
        ax.annotate(
            label,
            (src.z, y_plot),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=SOURCE_LABEL_FONTSIZE,
            ha=horizontal_alignment,
            va=vertical_alignment,
            multialignment="center",
            color="black",
            arrowprops=dict(
                arrowstyle="-", color="0.18", lw=0.45, alpha=0.55, shrinkA=1.5, shrinkB=3.0
            ),
            bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.64),
            zorder=8,
        )


def make_figure() -> tuple[float, float]:
    z = np.logspace(-3, 0.0, 360)
    mag = np.linspace(MAG_MIN, MAG_MAX, 260)
    z_grid, mag_grid = np.meshgrid(z, mag)
    l_prom = required_baseline_km(z_grid, mag_grid, "promoted")
    l_edge = required_baseline_km(z_grid, mag_grid, "edge")

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    norm = mpl.colors.LogNorm(vmin=BASELINE_MIN_KM, vmax=BASELINE_COLORBAR_MAX_KM)
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_under("#12091f")
    cmap.set_over("#fff4b2")
    mesh = ax.pcolormesh(
        z_grid,
        mag_grid,
        l_prom,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.015, extend="both")
    colorbar.set_label(
        r"collective required baseline $L_{\rm req}$ for $M_0=10^8M_\odot$ [km]",
        fontsize=COLORBAR_FONTSIZE,
    )
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    contour_handles = add_baseline_contours(ax, z_grid, mag_grid, l_prom, l_edge)
    m_prom, m_edge = add_sigma_pi_limits(ax)
    add_source_markers(ax)

    marker_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="#d62728", lw=0, markerfacecolor="#d62728", markeredgecolor="white", label="Seyfert"),
        mpl.lines.Line2D([0], [0], marker="P", color="#ff7f0e", lw=0, markerfacecolor="#ff7f0e", markeredgecolor="white", label="radio AGN"),
        mpl.lines.Line2D([0], [0], marker="s", color="#1f77b4", lw=0, markerfacecolor="#1f77b4", markeredgecolor="white", label="blazar/jet"),
        mpl.lines.Line2D([0], [0], marker="D", color="#9467bd", lw=0, markerfacecolor="#9467bd", markeredgecolor="white", label="quasar"),
    ]
    legend_handles = [
        *contour_handles,
        mpl.lines.Line2D([0], [0], color="0.25", lw=1.3, ls="-", label=PROMOTED_CURVE_LABEL),
        mpl.lines.Line2D([0], [0], color="0.25", lw=1.3, ls="--", label=EDGE_CURVE_LABEL),
        *marker_handles,
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=0.82,
        borderpad=0.55,
        handlelength=1.9,
    )

    ax.set_xscale("log")
    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(MAG_MAX, MAG_MIN)
    ax.set_xlabel("redshift $z$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("AB magnitude per station aperture", fontsize=AXIS_LABEL_FONTSIZE)
    style_axes(ax)

    for extension in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{FIGURE_STEM}.{extension}", dpi=300)
    plt.close(fig)
    return m_prom, m_edge


def build_results(m_prom: float, m_edge: float) -> dict[str, object]:
    check_mag = np.linspace(MAG_MIN, MAG_MAX, 1001)
    terms = fisher_terms(check_mag)
    f_edge = terms["F_edge"]
    f_prom = terms["F_prom"]
    ratio = f_prom / f_edge
    ratio_expected = (N_STATION - 1.0) * terms["s0"] / (terms["s0"] - terms["c"])
    ratio_relative_error = np.max(np.abs(ratio / ratio_expected - 1.0))
    source_endpoint = (N_STATION - 1.0) / (1.0 - NU)

    positivity_pass = bool(np.all(f_edge > 0.0) and np.all(f_prom > 0.0))
    ratio_pass = bool(ratio_relative_error < 1.0e-12)
    endpoint_pass = bool(math.isclose(source_endpoint, 38.0, rel_tol=0.0, abs_tol=1.0e-14))
    if not (positivity_pass and ratio_pass and endpoint_pass):
        raise RuntimeError("a requested Fisher-information numerical check failed")

    at_mag_20 = fisher_terms(20.0)
    source_records = []
    for src in SOURCES:
        source_records.append(
            {
                "name": src.name,
                "kind": src.kind,
                "z": src.z,
                "mag_ab": src.mag_ab,
                "mbh_msun": src.mbh_msun,
                "mbh_err_minus_msun": src.mbh_err_minus_msun,
                "mbh_err_plus_msun": src.mbh_err_plus_msun,
                "mass_reference": src.mass_reference,
                "alpha_vs_reference_mass": src.mbh_msun / MASS_REFERENCE_MSUN,
                "promoted_required_baseline_km": float(
                    required_baseline_km(src.z, src.mag_ab, "promoted", src.mbh_msun)
                ),
                "promoted_required_baseline_mass_low_km": float(
                    required_baseline_km(
                        src.z,
                        src.mag_ab,
                        "promoted",
                        src.mbh_msun - src.mbh_err_minus_msun,
                    )
                ),
                "promoted_required_baseline_mass_high_km": float(
                    required_baseline_km(
                        src.z,
                        src.mag_ab,
                        "promoted",
                        src.mbh_msun + src.mbh_err_plus_msun,
                    )
                ),
            }
        )

    return {
        "artifact": "fig5_astrophysical_reach",
        "parameters": {
            "N": N_STATION,
            "M_total": M_TOTAL,
            "nu": NU,
            "eta": ETA,
            "epsilon": EPSILON_BG,
            "telescope_diameter_m": D_TEL,
            "wavelength_nm": WAVELENGTH * 1.0e9,
            "kappa_cl": KAPPA_CL,
            "phase_budget_fraction": PHASE_BUDGET_FRACTION,
            "reference_black_hole_mass_msun": MASS_REFERENCE_MSUN,
        },
        "dimensions_and_functionals": {
            "independent_closure_phase_dimension_C": CLOSURE_DIMENSION,
            "E_formula": "N(N-1)/2",
            "C_formula": "(N-1)(N-2)/2",
            "phase_target": "one oriented triangle closure functional only",
            "resource_allocation": "100% of the receiver budget is assigned to phase; no amplitude branch is present",
        },
        "mode_copy_and_spatial_accounting": {
            "N_station_spatial_block": "one simultaneous N-station spatial array with E edges and C independent closure phases",
            "F_edge": "Fisher information per temporal-mode copy",
            "F_prom": "per-copy asymptotic Fisher information after the same-POVM coherent-score lift at joint depth n_s -> infinity",
            "M_total_definition": "accumulated temporal-mode copies / total time-frequency modes",
            "joint_depth_n_s": "number of temporal copies jointly measured by the promoted receiver; n_s -> infinity defines the per-copy asymptotic F_prom and is not M_total",
            "sigma_cl": "1/sqrt(M_total F_per_copy)",
            "distinction": "N specifies the spatial block, n_s specifies receiver joint depth, and M_total specifies accumulated temporal-mode copies; none is a black-hole mass",
            "extra_normalization": "none; in particular, no 50:50 phase penalty or (N-1)/2 normalization is applied",
        },
        "formulas": {
            "occupation": "u(m_AB)=A Fnu0 10^(-0.4 m_AB)/(h nu_opt), with A=pi(D/2)^2 and nu_opt=c_light/wavelength",
            "a": "a=eta u",
            "s0": "s0=a+epsilon",
            "c": "c=a nu",
            "copy_local_uniform_edge_first": "F_edge=2c^2/[3(N-1)s0]",
            "same_POVM_ns_to_infinity_coherent_score_lift": "F_prom=2c^2/[3(s0-c)]",
            "closure_uncertainty": "sigma_cl=1/sqrt(M_total F)",
            "angular_uncertainty": "sigma_theta=sigma_cl/[kappa_cl 2 pi L/wavelength]",
            "baseline_requirement": "L_req=sigma_cl wavelength/[2 pi kappa_cl (R_s/D_A)]",
            "schwarzschild_radius": "R_s=2 G M_BH/c_light^2",
            "plot_mapping": "background and solid contours use F_prom; dashed contours use F_edge",
        },
        "sigma_cl_equals_pi": {
            "promoted_mag_ab": m_prom,
            "copy_local_edge_first_mag_ab": m_edge,
            "both_horizontal_lines_drawn": True,
        },
        "numerical_checks": {
            "positive_F": {
                "pass": positivity_pass,
                "magnitude_grid": [MAG_MIN, MAG_MAX, int(check_mag.size)],
                "minimum_F_edge": float(np.min(f_edge)),
                "minimum_F_prom": float(np.min(f_prom)),
            },
            "promoted_to_edge_ratio": {
                "pass": ratio_pass,
                "identity": "F_prom/F_edge=(N-1)s0/(s0-c)",
                "maximum_relative_error_on_grid": float(ratio_relative_error),
                "at_mag_ab_20": {
                    "actual": float(at_mag_20["F_prom"] / at_mag_20["F_edge"]),
                    "expected": float(
                        (N_STATION - 1.0)
                        * at_mag_20["s0"]
                        / (at_mag_20["s0"] - at_mag_20["c"])
                    ),
                },
            },
            "source_dominated_endpoint": {
                "pass": endpoint_pass,
                "limit": "(N-1)/(1-nu)",
                "value": source_endpoint,
                "required_value": 38.0,
            },
        },
        "source_markers": source_records,
    }


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if EDGE_DIMENSION != 190 or CLOSURE_DIMENSION != 171:
        raise RuntimeError("unexpected N=20 graph dimensions")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    m_prom, m_edge = make_figure()
    results = build_results(m_prom, m_edge)
    result_path = RESULT_DIR / "fig5_astrophysical_reach.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {FIGURE_DIR / 'fig5_astrophysical_reach.png'}")
    print(f"Wrote {FIGURE_DIR / 'fig5_astrophysical_reach.pdf'}")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
