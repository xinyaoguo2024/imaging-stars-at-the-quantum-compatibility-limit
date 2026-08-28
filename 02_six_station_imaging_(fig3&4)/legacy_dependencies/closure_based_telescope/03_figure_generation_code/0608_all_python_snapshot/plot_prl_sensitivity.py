from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


H_PLANCK = 6.62607015e-34
C_LIGHT = 299_792_458.0
FNU_AB0 = 3631e-26
STATION_DIAMETER_M = 5.0
BASELINE_KM = 20.0
AGN_MARKER_MAG = 12.8
OJ287_MARKER_MAG = 14.8
OJ287_BURST_MARKER_MAG = 12.0
MODE_FALSE_POSITIVE = float(os.environ.get("MODE_FALSE_POSITIVE", "0.05"))
PAIR_FALSE_POSITIVE = float(os.environ.get("PAIR_FALSE_POSITIVE", "0.0"))


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
        matrix = np.array(
            [
                [s1[idx] + s2[idx], -g31[idx], -g23[idx]],
                [-g31[idx], s2[idx] + s3[idx], -g12[idx]],
                [-g23[idx], -g12[idx], s3[idx] + s1[idx]],
            ],
            dtype=float,
        )
        rhs = (2.0 / 3.0) * np.array([g12[idx], g23[idx], g31[idx]], dtype=float)
        x, y, z = np.linalg.solve(matrix, rhs)
        out[idx] = (2.0 / 3.0) * (g12[idx] * x + g23[idx] * y + g31[idx] * z)
    return out


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
    freq = np.linspace(C_LIGHT / lam_max_m, C_LIGHT / lam_min_m, 4096)
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
) -> np.ndarray:
    fisher = fisher_three_station(
        mag_ab,
        n_samples,
        pair_losses=pair_losses,
        nus=nus,
        strategy=strategy,
        exposure_s=exposure_s,
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
            )
            if float(snr) >= target_snr:
                lo = mid
            else:
                hi = mid
        limits.append(0.5 * (lo + hi))
    return np.asarray(limits)


def main() -> None:
    target_snr = 3.0
    exposure_s = 600.0
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
    sample_counts = np.array([1.0, 30.0])
    diameter_ref_m = STATION_DIAMETER_M
    diameters_m = np.linspace(3.0, 7.0, 161)

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
    chi = np.logspace(-2.0, 0.0, 261)
    chi2, chi3 = np.meshgrid(chi, chi, indexing="xy")
    g0_noise_dominated = np.sqrt(2.0)
    asym_factor = np.sqrt((1.0 + chi2**2 + chi3**2) * (1.0 + chi2 ** -2 + chi3 ** -2)) / 3.0
    gain = g0_noise_dominated * asym_factor
    image = ax.pcolormesh(
        chi2,
        chi3,
        gain,
        shading="auto",
        cmap="magma",
        norm=LogNorm(vmin=g0_noise_dominated, vmax=float(np.nanmax(gain))),
    )
    contours = ax.contour(
        chi2,
        chi3,
        gain,
        levels=[1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0],
        colors="white",
        linewidths=0.55,
        alpha=0.75,
    )
    ax.clabel(contours, inline=True, fontsize=6.8, fmt="%.1f")
    ax.plot([0.01, 1.0], [0.01, 1.0], color="white", lw=0.8, ls="--", alpha=0.55)
    ax.scatter([1.0], [1.0], color="white", s=18, edgecolor="0.2", linewidth=0.5, zorder=4)
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"SNR gain $G$")
    cbar.set_ticks([np.sqrt(2.0), 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    cbar.set_ticklabels([r"$\sqrt{2}$", "2", "3", "5", "10", "20", "50"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(0.01, 1.0)
    ax.set_xlabel(r"$\chi_2=\mathrm{SNR}_{(2)}/\mathrm{SNR}_{(1)}$")
    ax.set_ylabel(r"$\chi_3=\mathrm{SNR}_{(3)}/\mathrm{SNR}_{(1)}$")
    ax.set_title(r"(a) Total gain, $G_0=\sqrt{2}$")

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
                np.array([30.0]),
                target_snr=target_snr,
                pair_losses=cfg["pair_losses"],
                nus=cfg["nus"],
                strategy=strategy,
                exposure_s=exposure_s,
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
    ax.text(6.68, AGN_MARKER_MAG + 0.10, "3C 273", fontsize=7.5, color="0.18")
    ax.text(6.72, OJ287_MARKER_MAG + 0.10, "OJ 287", fontsize=7.5, color="0.18")
    ax.text(6.22, OJ287_BURST_MARKER_MAG + 0.08, "OJ 287 burst", fontsize=7.3, color="0.18")
    ax.set_xlabel(r"station diameter $D$ (m)")
    ax.set_ylabel(r"limiting AB magnitude for SNR $=3$")
    ax.set_title("(b) 30 broadband samples")
    ax.set_xlim(3.0, 7.0)
    ax.set_ylim(10.9, 15.9)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", frameon=True, framealpha=0.88, facecolor="white", edgecolor="none")

    png = OUTDIR / "prl_closure_sensitivity.png"
    pdf = OUTDIR / "prl_closure_sensitivity.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    stats = {
        "model": "explicit_three_station_sld_with_pure_attenuation_plus_independent_false_positive",
        "target_snr": target_snr,
        "station_diameter_m": STATION_DIAMETER_M,
        "capture_efficiency_including_eta_Q": 0.30,
        "eta_Q": 1.0,
        "mode_false_positive": MODE_FALSE_POSITIVE,
        "pair_false_positive": PAIR_FALSE_POSITIVE,
        "exposure_s_per_sample": exposure_s,
        "lam_min_m": 400e-9,
        "lam_max_m": 800e-9,
        "photons_m0_per_10min_per_station": float(
            photon_count_ab(0.0, diameter_m=STATION_DIAMETER_M, exposure_s=exposure_s)
        ),
        "g0_noise_dominated_symmetric_snr_gain": float(g0_noise_dominated),
        "agn_marker_mag_ab": AGN_MARKER_MAG,
        "oj287_typical_marker_mag_ab": OJ287_MARKER_MAG,
        "oj287_burst_marker_mag_ab": OJ287_BURST_MARKER_MAG,
        "oj287_burst_marker_note": "Optimistic optical high-state benchmark; literature reports very bright OJ 287 high states reaching about 12 mag.",
        "snr_gain_min": float(np.nanmin(gain)),
        "snr_gain_max": float(np.nanmax(gain)),
        "diameter_curve_scaling": "weak_source_noise_dominated_limit: m_lim(D)=m_lim(5m)+5log10(D/5m)",
        "diameter_range_m": [float(diameters_m[0]), float(diameters_m[-1])],
        "reference_limits_30x10min_5m": {
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
            "nus_nu12_nu23_nu31": list(cfg["nus"]),
            "limiting_magnitudes": {},
            "snr_at_3c273": {},
            "snr_at_oj287": {},
            "snr_at_oj287_burst": {},
        }
        for ns in sample_counts:
            key = f"{int(ns)}x10min"
            scenario_stats["limiting_magnitudes"][key] = {
                "direct": float(
                    limiting_magnitude(
                        np.array([ns]),
                        target_snr=target_snr,
                        pair_losses=cfg["pair_losses"],
                        nus=cfg["nus"],
                        strategy="direct",
                        exposure_s=exposure_s,
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
                        )
                    ),
                }
        stats["scenarios"][name] = scenario_stats
    stats_path = OUTDIR / "prl_closure_sensitivity_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(pdf)
    print(png)
    print(stats_path)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
