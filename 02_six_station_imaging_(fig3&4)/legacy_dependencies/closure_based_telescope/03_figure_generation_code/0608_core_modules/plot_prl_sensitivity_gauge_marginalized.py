from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import make_eight_station_cfi_qfi_note_tables_gauge_marginalized as cfi_tables


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
BUNDLE = Path(__file__).resolve().parents[2]
OUTDIR = BUNDLE / "figures" / "main"
ROOT_FIGDIR = BUNDLE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
ROOT_FIGDIR.mkdir(parents=True, exist_ok=True)


H_PLANCK = 6.62607015e-34
C_LIGHT = 299_792_458.0
FNU_AB0 = 3631e-26
STATION_DIAMETER_M = float(os.environ.get("FIG1_STATION_DIAMETER_M", os.environ.get("FIG2_REMOTE_DIAMETER_M", "2.0")))
BASELINE_KM = 20.0
AGN_MARKER_MAG = 12.8
OJ287_MARKER_MAG = 14.8
OJ287_BURST_MARKER_MAG = 12.0
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", os.environ.get("EPS_STATION", "1e-9")))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", os.environ.get("EPS_PAIR", "0.0")))
DIRECT_FALSE_POSITIVE = float(os.environ.get("DIRECT_FALSE_POSITIVE", os.environ.get("EPS_DIRECT_EXTRA", "0.0")))
EXPOSURE_S_DEFAULT = float(os.environ.get("FIG1_EXPOSURE_S", os.environ.get("FIG2_EXPOSURE_S", os.environ.get("EXPOSURE_S", "0.050"))))
N_SAMPLES_REF = float(os.environ.get("FIG1_N_SAMPLES", os.environ.get("FIG2_N_TIME_WINDOWS", "36")))
LAMBDA_MIN_M = float(os.environ.get("FIG1_LAMBDA_MIN_NM", os.environ.get("FIG2_LAMBDA_MIN_NM", "600.0"))) * 1.0e-9
LAMBDA_MAX_M = float(os.environ.get("FIG1_LAMBDA_MAX_NM", os.environ.get("FIG2_LAMBDA_MAX_NM", "700.0"))) * 1.0e-9
DIAMETER_MIN_M = float(os.environ.get("FIG1_DIAMETER_MIN_M", str(max(0.8, 0.5 * STATION_DIAMETER_M))))
DIAMETER_MAX_M = float(os.environ.get("FIG1_DIAMETER_MAX_M", str(max(5.0, 2.5 * STATION_DIAMETER_M))))
FREQ_GRID_POINTS = int(os.environ.get("PRL_SENSITIVITY_FREQ_GRID_POINTS", "512"))


def exposure_label(exposure_s: float) -> str:
    if abs(exposure_s % 60.0) < 1.0e-12 and exposure_s >= 60.0:
        return f"{exposure_s / 60.0:g}min"
    return f"{exposure_s:g}s"


def photon_count_ab(
    mag_ab: np.ndarray | float,
    *,
    diameter_m: float = STATION_DIAMETER_M,
    capture_efficiency: float = 0.30,
    exposure_s: float = 300.0,
    lam_min_m: float = 400e-9,
    lam_max_m: float = 800e-9,
) -> np.ndarray:
    """Detected station photons in one broadband sample for a flat AB spectrum."""
    area = np.pi * (diameter_m / 2.0) ** 2
    photons_m0 = (
        area
        * capture_efficiency
        * exposure_s
        * FNU_AB0
        * np.log(lam_max_m / lam_min_m)
        / H_PLANCK
    )
    return photons_m0 * 10.0 ** (-0.4 * np.asarray(mag_ab))


def mode_occupation_ab(
    mag_ab: np.ndarray | float,
    freq_hz: np.ndarray,
    *,
    diameter_m: float = STATION_DIAMETER_M,
    capture_efficiency: float = 0.30,
) -> np.ndarray:
    """Mean source photon occupation per station, temporal mode, and frequency."""
    area = np.pi * (diameter_m / 2.0) ** 2
    fnu = FNU_AB0 * 10.0 ** (-0.4 * np.asarray(mag_ab))
    return area * capture_efficiency * np.asarray(fnu)[..., None] / (H_PLANCK * freq_hz)


def station_efficiencies_from_pair_attenuation_losses(pair_losses: tuple[float, float, float]) -> np.ndarray:
    """Convert pair coherent-amplitude loss fractions to station efficiencies.

    A pair loss L_ij means sqrt(eta_i eta_j)=1-L_ij.  Pure fibre attenuation
    only changes eta_i; it does not generate false positives.  The independent
    mode-local false-positive occupancy is supplied separately.
    """
    t12, t23, t31 = 1.0 - np.asarray(pair_losses, dtype=float)
    if min(t12, t23, t31) <= 0.0:
        raise ValueError(f"pair losses imply non-positive transmission: {pair_losses}")
    log_eta = np.array(
        [
            np.log(t12) + np.log(t31) - np.log(t23),
            np.log(t12) + np.log(t23) - np.log(t31),
            np.log(t23) + np.log(t31) - np.log(t12),
        ]
    )
    eta = np.exp(log_eta)
    if np.any(eta > 1.0 + 1e-9):
        raise ValueError(f"pair transmissions are not passive station-local realizable: {pair_losses}")
    return np.clip(eta, 0.0, 1.0)


def three_station_fisher_density(
    u: np.ndarray,
    *,
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    strategy: str,
) -> np.ndarray:
    """Per-temporal-frequency-mode FI for closure phase using the explicit SLD model.

    The input ``pair_losses`` are pure coherent-amplitude attenuation fractions
    L_ij, mapped by sqrt(eta_i eta_j)=1-L_ij.  False positives are independent
    mode-local occupancies, not fibre-loss products.
    """
    eta = station_efficiencies_from_pair_attenuation_losses(pair_losses)
    eps = np.full(3, MODE_FALSE_POSITIVE, dtype=float)
    eps_dir = np.full(3, DIRECT_FALSE_POSITIVE, dtype=float)

    u = np.asarray(u, dtype=float)
    s1 = eta[0] * u + eps[0]
    s2 = eta[1] * u + eps[1]
    s3 = eta[2] * u + eps[2]
    g12 = u * np.sqrt(eta[0] * eta[1]) * nus[0]
    g23 = u * np.sqrt(eta[1] * eta[2]) * nus[1]
    g31 = u * np.sqrt(eta[2] * eta[0]) * nus[2]

    if strategy == "split":
        f = 0.5
        f12 = 4.0 * (f * g12) ** 2 / (f * (s1 + s2) + PAIR_FALSE_POSITIVE)
        f23 = 4.0 * (f * g23) ** 2 / (f * (s2 + s3) + PAIR_FALSE_POSITIVE)
        f31 = 4.0 * (f * g31) ** 2 / (f * (s3 + s1) + PAIR_FALSE_POSITIVE)
        return 1.0 / (1.0 / f12 + 1.0 / f23 + 1.0 / f31)

    if strategy != "direct":
        raise ValueError(strategy)

    out = np.empty_like(u, dtype=float)
    for idx in np.ndindex(u.shape):
        out[idx] = cfi_tables.triangle_direct_fisher(
            float(g12[idx]),
            float(g23[idx]),
            float(g31[idx]),
            float(s1[idx] + eps_dir[0]),
            float(s2[idx] + eps_dir[1]),
            float(s3[idx] + eps_dir[2]),
        )
    return out


def edge_closure_fisher_density(
    g12: float,
    g23: float,
    g31: float,
    s1: float,
    s2: float,
    s3: float,
    *,
    split_fraction: float,
    pair_false_positive: float = PAIR_FALSE_POSITIVE,
) -> float:
    """Closure FI from separately measured edge phases."""
    f = split_fraction
    f12 = 4.0 * (f * g12) ** 2 / max(f * (s1 + s2) + pair_false_positive, 1e-300)
    f23 = 4.0 * (f * g23) ** 2 / max(f * (s2 + s3) + pair_false_positive, 1e-300)
    f31 = 4.0 * (f * g31) ** 2 / max(f * (s3 + s1) + pair_false_positive, 1e-300)
    f_edges = np.maximum([f12, f23, f31], 1e-300)
    return float(1.0 / np.sum(1.0 / f_edges))


def gauge_marginalized_gain_grid(chi: np.ndarray, *, nu_ref: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Gauge-marginalized three-station gain and intrinsic joint-measurement part.

    The reference model has equal station loads and visibility amplitudes
    (nu_ref, chi2*nu_ref, chi3*nu_ref).  Since edge SNR is proportional to
    the coherent visibility amplitude in this equal-load comparison, the two
    axes are also the edge-SNR ratios.
    """
    chi2, chi3 = np.meshgrid(chi, chi, indexing="xy")
    gain = np.empty_like(chi2, dtype=float)
    intrinsic = np.empty_like(chi2, dtype=float)
    s1 = s2 = s3 = 1.0
    s_direct = 1.0 + DIRECT_FALSE_POSITIVE / max(MODE_FALSE_POSITIVE, 1e-300)
    s_edge = 1.0
    pair_scaled = PAIR_FALSE_POSITIVE / max(MODE_FALSE_POSITIVE, 1e-300)
    split_gain = np.sqrt(2.0 * (s_edge + pair_scaled) / s_direct)
    for idx in np.ndindex(chi2.shape):
        g12 = nu_ref
        g23 = nu_ref * float(chi2[idx])
        g31 = nu_ref * float(chi3[idx])
        f_direct = cfi_tables.triangle_direct_fisher(g12, g23, g31, s_direct, s_direct, s_direct)
        f_edge_split = edge_closure_fisher_density(
            g12,
            g23,
            g31,
            s_edge,
            s_edge,
            s_edge,
            split_fraction=0.5,
            pair_false_positive=pair_scaled,
        )
        gain[idx] = np.sqrt(max(f_direct, 0.0) / max(f_edge_split, 1e-300))
        intrinsic[idx] = gain[idx] / split_gain
    return gain, intrinsic


def fisher_three_station(
    mag_ab: np.ndarray | float,
    n_samples: float,
    *,
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    strategy: str,
    exposure_s: float = 600.0,
    diameter_m: float = STATION_DIAMETER_M,
    capture_efficiency: float = 0.30,
    lam_min_m: float = 400e-9,
    lam_max_m: float = 800e-9,
) -> np.ndarray:
    """Broadband FI from the explicit three-station lossy/noisy closure model."""
    freq = np.linspace(C_LIGHT / lam_max_m, C_LIGHT / lam_min_m, FREQ_GRID_POINTS)
    u_f = mode_occupation_ab(
        mag_ab,
        freq,
        diameter_m=diameter_m,
        capture_efficiency=capture_efficiency,
    )
    density = three_station_fisher_density(
        u_f,
        pair_losses=pair_losses,
        nus=nus,
        strategy=strategy,
    )
    return n_samples * exposure_s * np.trapezoid(density, freq, axis=-1)


def snr_three_station(
    mag_ab: np.ndarray | float,
    n_samples: float,
    *,
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    strategy: str,
    exposure_s: float = 600.0,
    diameter_m: float = STATION_DIAMETER_M,
    lam_min_m: float = 400e-9,
    lam_max_m: float = 800e-9,
) -> np.ndarray:
    fisher = fisher_three_station(
        mag_ab,
        n_samples,
        pair_losses=pair_losses,
        nus=nus,
        strategy=strategy,
        exposure_s=exposure_s,
        diameter_m=diameter_m,
        lam_min_m=lam_min_m,
        lam_max_m=lam_max_m,
    )
    return np.sqrt(fisher)


def limiting_magnitude(
    n_samples: np.ndarray,
    *,
    target_snr: float,
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    strategy: str,
    exposure_s: float = 600.0,
    diameter_m: float = STATION_DIAMETER_M,
    lam_min_m: float = 400e-9,
    lam_max_m: float = 800e-9,
) -> np.ndarray:
    limits = []
    for ns in np.asarray(n_samples, dtype=float):
        lo, hi = -5.0, 24.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            snr = snr_three_station(
                mid,
                ns,
                pair_losses=pair_losses,
                nus=nus,
                strategy=strategy,
                exposure_s=exposure_s,
                diameter_m=diameter_m,
                lam_min_m=lam_min_m,
                lam_max_m=lam_max_m,
            )
            if float(snr) >= target_snr:
                lo = mid
            else:
                hi = mid
        limits.append(0.5 * (lo + hi))
    return np.asarray(limits)


def main() -> None:
    target_snr = 3.0
    exposure_s = EXPOSURE_S_DEFAULT
    scenarios = {
        "three long baselines": {
            "pair_losses": (0.20, 0.20, 0.20),
            "nus": (0.10, 0.10, 0.10),
            "short": "LLL",
        },
        "one short + two long": {
            "pair_losses": (0.05, 0.20, 0.20),
            "nus": (0.80, 0.10, 0.10),
            "short": "SLL",
        },
    }
    sample_counts = np.array([1.0, N_SAMPLES_REF])
    diameter_ref_m = STATION_DIAMETER_M
    diameters_m = np.linspace(DIAMETER_MIN_M, DIAMETER_MAX_M, 161)

    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.9,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.2,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.95), constrained_layout=True)

    ax = axes[0]
    chi = np.logspace(-2.0, 0.0, 121)
    chi2, chi3 = np.meshgrid(chi, chi, indexing="xy")
    background_split_gain = np.sqrt(
        2.0 * (MODE_FALSE_POSITIVE + PAIR_FALSE_POSITIVE)
        / (MODE_FALSE_POSITIVE + DIRECT_FALSE_POSITIVE)
    )
    reference_nu = 0.3
    gain, intrinsic_gain = gauge_marginalized_gain_grid(chi, nu_ref=reference_nu)
    image = ax.pcolormesh(
        chi2,
        chi3,
        gain,
        shading="auto",
        cmap="magma",
        vmin=float(np.nanmin(gain)),
        vmax=float(np.nanmax(gain)),
    )
    contour_levels = np.linspace(float(np.nanmin(gain)), float(np.nanmax(gain)), 5)[1:-1]
    contours = ax.contour(
        chi2,
        chi3,
        gain,
        levels=contour_levels,
        colors="white",
        linewidths=0.55,
        alpha=0.75,
    )
    ax.clabel(contours, inline=True, fontsize=6.8, fmt="%.2f")
    ax.plot([0.01, 1.0], [0.01, 1.0], color="white", lw=0.8, ls="--", alpha=0.55)
    ax.scatter([1.0], [1.0], color="white", s=18, edgecolor="0.2", linewidth=0.5, zorder=4)
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"SNR gain $G_{\rm split}\,G_{\rm joint}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.01, 1.0)
    ax.set_xlabel(r"$\chi_2=\mathrm{SNR}_{(2)}/\mathrm{SNR}_{(1)}$")
    ax.set_ylabel(r"$\chi_3=\mathrm{SNR}_{(3)}/\mathrm{SNR}_{(1)}$")
    ax.set_title(r"(a) Gauge-marginalized gain")

    ax = axes[1]
    line_styles = {
        ("three long baselines", "split"): ("#bb3e03", "--", "LLL edge-first"),
        ("three long baselines", "direct"): ("#bb3e03", "-", "LLL closure-space"),
        ("one short + two long", "split"): ("#005f73", "--", "SLL edge-first"),
        ("one short + two long", "direct"): ("#005f73", "-", "SLL closure-space"),
    }
    reference_limits_30 = {}
    for name, cfg in scenarios.items():
        for strategy in ["split", "direct"]:
            ref_limit = limiting_magnitude(
                np.array([N_SAMPLES_REF]),
                target_snr=target_snr,
                pair_losses=cfg["pair_losses"],
                nus=cfg["nus"],
                strategy=strategy,
                exposure_s=exposure_s,
                diameter_m=diameter_ref_m,
                lam_min_m=LAMBDA_MIN_M,
                lam_max_m=LAMBDA_MAX_M,
            )[0]
            reference_limits_30[(name, strategy)] = ref_limit
            curve = ref_limit + 5.0 * np.log10(diameters_m / diameter_ref_m)
            color, linestyle, label = line_styles[(name, strategy)]
            ax.plot(diameters_m, curve, color=color, ls=linestyle, lw=2.0, label=label)
            ax.scatter([diameter_ref_m], [ref_limit], color=color, s=18, zorder=3)
            if strategy == "direct":
                ax.text(
                    diameter_ref_m + 0.07,
                    ref_limit + 0.08,
                    f"{ref_limit:.1f}",
                    fontsize=7.0,
                    color=color,
                )
    ax.axhline(AGN_MARKER_MAG, color="0.18", lw=1.0, ls=":", alpha=0.85)
    ax.axhline(OJ287_MARKER_MAG, color="0.18", lw=1.0, ls="-.", alpha=0.85)
    ax.axhline(OJ287_BURST_MARKER_MAG, color="0.18", lw=0.95, ls=(0, (4, 2)), alpha=0.75)
    x_text = float(diameters_m[-1] - 0.18 * (diameters_m[-1] - diameters_m[0]))
    ax.text(x_text, AGN_MARKER_MAG + 0.10, "3C 273", fontsize=7.5, color="0.18")
    ax.text(x_text, OJ287_MARKER_MAG + 0.10, "OJ 287", fontsize=7.5, color="0.18")
    ax.text(x_text, OJ287_BURST_MARKER_MAG + 0.08, "OJ 287 burst", fontsize=7.3, color="0.18")
    ax.set_xlabel(r"station diameter $D$ (m)")
    ax.set_ylabel(r"limiting AB magnitude for SNR $=3$")
    ax.set_title(f"(b) {N_SAMPLES_REF:g} samples, {exposure_label(exposure_s)} each")
    ax.set_xlim(float(diameters_m[0]), float(diameters_m[-1]))
    y_values = []
    for name, cfg in scenarios.items():
        for strategy in ["split", "direct"]:
            y_values.append(reference_limits_30[(name, strategy)] + 5.0 * np.log10(diameters_m / diameter_ref_m))
    y_values = np.concatenate(y_values)
    marker_values = np.asarray([AGN_MARKER_MAG, OJ287_MARKER_MAG, OJ287_BURST_MARKER_MAG])
    y_min = float(min(np.nanmin(y_values), np.nanmin(marker_values)) - 0.35)
    y_max = float(max(np.nanmax(y_values), np.nanmax(marker_values)) + 0.45)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")

    png = OUTDIR / "prl_closure_sensitivity_gauge_marginalized.png"
    pdf = OUTDIR / "prl_closure_sensitivity_gauge_marginalized.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    root_png = ROOT_FIGDIR / "prl_closure_sensitivity_gauge_marginalized.png"
    root_pdf = ROOT_FIGDIR / "prl_closure_sensitivity_gauge_marginalized.pdf"
    fig.savefig(root_png, dpi=260, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")

    stats = {
        "model": "gauge_marginalized_three_station_sld_with_pure_attenuation_plus_independent_false_positive",
        "target_snr": target_snr,
        "station_diameter_m": STATION_DIAMETER_M,
        "capture_efficiency_including_eta_Q": 0.30,
        "eta_Q": 1.0,
        "mode_false_positive": MODE_FALSE_POSITIVE,
        "pair_false_positive": PAIR_FALSE_POSITIVE,
        "direct_false_positive": DIRECT_FALSE_POSITIVE,
        "freq_grid_points": FREQ_GRID_POINTS,
        "exposure_s_per_sample": exposure_s,
        "n_samples_reference": N_SAMPLES_REF,
        "lam_min_m": LAMBDA_MIN_M,
        "lam_max_m": LAMBDA_MAX_M,
        "photons_m0_per_sample_per_station": float(
            photon_count_ab(
                0.0,
                diameter_m=STATION_DIAMETER_M,
                exposure_s=exposure_s,
                lam_min_m=LAMBDA_MIN_M,
                lam_max_m=LAMBDA_MAX_M,
            )
        ),
        "g_split_three_station": float(background_split_gain),
        "heatmap_reference_nu_max": float(reference_nu),
        "heatmap_gain_definition": (
            "total SNR gain over simultaneous split edge-first; "
            "G_joint=G_total/sqrt(2) is the intrinsic three-mode gain over unsplit edge-first"
        ),
        "agn_marker_mag_ab": AGN_MARKER_MAG,
        "oj287_typical_marker_mag_ab": OJ287_MARKER_MAG,
        "oj287_burst_marker_mag_ab": OJ287_BURST_MARKER_MAG,
        "oj287_burst_marker_note": "Optimistic optical high-state benchmark; literature reports very bright OJ 287 high states reaching about 12 mag.",
        "snr_gain_total_min": float(np.nanmin(gain)),
        "snr_gain_total_max": float(np.nanmax(gain)),
        "snr_gain_intrinsic_min": float(np.nanmin(intrinsic_gain)),
        "snr_gain_intrinsic_max": float(np.nanmax(intrinsic_gain)),
        "diameter_curve_scaling": f"weak_source_noise_dominated_limit: m_lim(D)=m_lim({diameter_ref_m:g}m)+5log10(D/{diameter_ref_m:g}m)",
        "diameter_range_m": [float(diameters_m[0]), float(diameters_m[-1])],
        f"reference_limits_{N_SAMPLES_REF:g}x{exposure_label(exposure_s)}_{diameter_ref_m:g}m": {
            f"{name}_{strategy}": float(value)
            for (name, strategy), value in reference_limits_30.items()
        },
        "scenarios": {},
    }
    for name, cfg in scenarios.items():
        scenario_stats = {
            "pair_attenuation_losses_L12_L23_L31": list(cfg["pair_losses"]),
            "station_efficiencies_eta1_eta2_eta3": station_efficiencies_from_pair_attenuation_losses(
                cfg["pair_losses"]
            ).tolist(),
            "station_false_positive_epsilon_i": [MODE_FALSE_POSITIVE] * 3,
            "direct_false_positive_epsilon_dir": [DIRECT_FALSE_POSITIVE] * 3,
            "pair_false_positive_epsilon_pair": PAIR_FALSE_POSITIVE,
            "nus_nu12_nu23_nu31": list(cfg["nus"]),
            "limiting_magnitudes": {},
            "snr_at_3c273": {},
            "snr_at_oj287": {},
            "snr_at_oj287_burst": {},
        }
        for ns in sample_counts:
            key = f"{ns:g}x{exposure_label(exposure_s)}"
            scenario_stats["limiting_magnitudes"][key] = {
                "direct": float(
                    limiting_magnitude(
                        np.array([ns]),
                        target_snr=target_snr,
                        pair_losses=cfg["pair_losses"],
                        nus=cfg["nus"],
                        strategy="direct",
                        exposure_s=exposure_s,
                        diameter_m=diameter_ref_m,
                        lam_min_m=LAMBDA_MIN_M,
                        lam_max_m=LAMBDA_MAX_M,
                    )[0]
                ),
                "split": float(
                    limiting_magnitude(
                        np.array([ns]),
                        target_snr=target_snr,
                        pair_losses=cfg["pair_losses"],
                        nus=cfg["nus"],
                        strategy="split",
                        exposure_s=exposure_s,
                        diameter_m=diameter_ref_m,
                        lam_min_m=LAMBDA_MIN_M,
                        lam_max_m=LAMBDA_MAX_M,
                    )[0]
                ),
            }
            for target_name, mag in [
                ("snr_at_3c273", AGN_MARKER_MAG),
                ("snr_at_oj287", OJ287_MARKER_MAG),
                ("snr_at_oj287_burst", OJ287_BURST_MARKER_MAG),
            ]:
                scenario_stats[target_name][key] = {
                    "direct": float(
                        snr_three_station(
                            mag,
                            ns,
                            pair_losses=cfg["pair_losses"],
                            nus=cfg["nus"],
                            strategy="direct",
                            exposure_s=exposure_s,
                            diameter_m=diameter_ref_m,
                            lam_min_m=LAMBDA_MIN_M,
                            lam_max_m=LAMBDA_MAX_M,
                        )
                    ),
                    "split": float(
                        snr_three_station(
                            mag,
                            ns,
                            pair_losses=cfg["pair_losses"],
                            nus=cfg["nus"],
                            strategy="split",
                            exposure_s=exposure_s,
                            diameter_m=diameter_ref_m,
                            lam_min_m=LAMBDA_MIN_M,
                            lam_max_m=LAMBDA_MAX_M,
                        )
                    ),
                }
        stats["scenarios"][name] = scenario_stats
    stats_path = OUTDIR / "prl_closure_sensitivity_gauge_marginalized_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    (ROOT_FIGDIR / "prl_closure_sensitivity_gauge_marginalized_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
