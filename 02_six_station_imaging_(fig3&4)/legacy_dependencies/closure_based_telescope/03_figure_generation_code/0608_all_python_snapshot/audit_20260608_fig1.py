from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_20260608_common import OUT, write_csv

import make_eight_station_cfi_qfi_note_tables_gauge_marginalized as cfi
import plot_prl_sensitivity_gauge_marginalized as sens


def edge_closure_fisher_eq13(
    g12: float,
    g23: float,
    g31: float,
    s_edge: float,
    pair_scaled: float,
) -> float:
    f = 0.5
    f12 = 4.0 * (f * g12) ** 2 / (f * (s_edge + s_edge) + pair_scaled)
    f23 = 4.0 * (f * g23) ** 2 / (f * (s_edge + s_edge) + pair_scaled)
    f31 = 4.0 * (f * g31) ** 2 / (f * (s_edge + s_edge) + pair_scaled)
    return float(1.0 / (1.0 / f12 + 1.0 / f23 + 1.0 / f31))


def gain_from_eq13(chi2: float, chi3: float, nu_ref: float) -> tuple[float, float, float]:
    pair_scaled = sens.PAIR_FALSE_POSITIVE / sens.MODE_FALSE_POSITIVE
    direct_scaled = sens.DIRECT_FALSE_POSITIVE / sens.MODE_FALSE_POSITIVE
    s_edge = 1.0
    s_direct = 1.0 + direct_scaled
    g12 = nu_ref
    g23 = nu_ref * chi2
    g31 = nu_ref * chi3
    f_direct = cfi.triangle_direct_fisher(g12, g23, g31, s_direct, s_direct, s_direct)
    f_edge = edge_closure_fisher_eq13(g12, g23, g31, s_edge, pair_scaled)
    return math.sqrt(f_direct / f_edge), f_direct, f_edge


def run_fig1_checks() -> dict[str, object]:
    chi = np.logspace(-2.0, 0.0, 41)
    code_gain, _intrinsic = sens.gauge_marginalized_gain_grid(chi, nu_ref=0.3)
    rows: list[dict[str, object]] = []
    rel_errors = np.zeros_like(code_gain)
    for iy, chi3 in enumerate(chi):
        for ix, chi2 in enumerate(chi):
            manual_gain, f_direct, f_edge = gain_from_eq13(float(chi2), float(chi3), 0.3)
            code_value = float(code_gain[iy, ix])
            rel = abs(code_value - manual_gain) / max(abs(manual_gain), 1e-300)
            rel_errors[iy, ix] = rel
            if ix in {0, len(chi) // 2, len(chi) - 1} and iy in {0, len(chi) // 2, len(chi) - 1}:
                rows.append(
                    {
                        "chi2": float(chi2),
                        "chi3": float(chi3),
                        "gain_code": code_value,
                        "gain_eq13_manual": manual_gain,
                        "rel_error": rel,
                        "direct_fisher": f_direct,
                        "edge_fisher": f_edge,
                    }
                )
    csv_path = OUT / "fig1a_formula_consistency_grid_spots.csv"
    write_csv(csv_path, rows)

    pair_scaled = sens.PAIR_FALSE_POSITIVE / sens.MODE_FALSE_POSITIVE
    direct_scaled = sens.DIRECT_FALSE_POSITIVE / sens.MODE_FALSE_POSITIVE
    eq16_gain = math.sqrt(2.0 * (1.0 + pair_scaled) / (1.0 * (1.0 - 0.3) + direct_scaled))
    code_sym = float(code_gain[-1, -1])

    fig, ax = plt.subplots(figsize=(3.2, 2.55), constrained_layout=True)
    mesh = ax.pcolormesh(chi, chi, np.maximum(rel_errors, 1e-18), shading="auto", cmap="viridis")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\chi_2$")
    ax.set_ylabel(r"$\chi_3$")
    ax.set_title("Fig. 1(a) Eq. (13) check")
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label("relative error")
    png = OUT / "fig1a_formula_consistency_error.png"
    pdf = OUT / "fig1a_formula_consistency_error.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "spot_csv": str(csv_path),
        "error_png": str(png),
        "error_pdf": str(pdf),
        "grid_shape": list(code_gain.shape),
        "max_rel_error_eq13": float(np.max(rel_errors)),
        "symmetric_code_gain": code_sym,
        "symmetric_eq16_gain": eq16_gain,
        "symmetric_rel_error_eq16": abs(code_sym - eq16_gain) / eq16_gain,
        "pair_scaled": float(pair_scaled),
        "direct_scaled": float(direct_scaled),
        "nu_ref": 0.3,
    }
