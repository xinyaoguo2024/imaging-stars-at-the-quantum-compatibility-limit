from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import make_eight_station_cfi_qfi_note_tables_gauge_marginalized as cfi
import plot_prl_sensitivity_gauge_marginalized as sens


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
FIG_DIR = ROOT / "output" / "figures"
PDF_DIR = ROOT / "output" / "pdf"
TABLE_DIR = ROOT / "output" / "tables"
for directory in (FIG_DIR, PDF_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FIG_PATH = FIG_DIR / "three_station_self_lock_fisher.pdf"
FIG_PNG_PATH = FIG_DIR / "three_station_self_lock_fisher.png"
STATS_PATH = TABLE_DIR / "three_station_self_lock_fisher_stats.json"
TEX_PATH = PDF_DIR / "three_station_self_lock_fisher_note.tex"

EPS_MODE = sens.MODE_FALSE_POSITIVE
PAIR_FP_LOCK = sens.PAIR_FALSE_POSITIVE
PAIR_FP_EDGE = sens.PAIR_FALSE_POSITIVE


def station_params(pair_losses: tuple[float, float, float], nus: tuple[float, float, float], u: float) -> dict:
    eta = sens.station_efficiencies_from_pair_attenuation_losses(pair_losses)
    eps = np.full(3, EPS_MODE, dtype=float)
    s = eta * u + eps
    g = np.array(
        [
            u * np.sqrt(eta[0] * eta[1]) * nus[0],
            u * np.sqrt(eta[1] * eta[2]) * nus[1],
            u * np.sqrt(eta[2] * eta[0]) * nus[2],
        ],
        dtype=float,
    )
    return {"eta": eta, "s": s, "g": g}


def edge_qfi_direct(g: np.ndarray, s: np.ndarray) -> np.ndarray:
    bmat, derivs = cfi.triangle_bmat_and_edge_derivatives(g[0], g[1], g[2], s[0], s[1], s[2])
    return cfi.base.qfi_from_bmat_derivatives(bmat, derivs)


def closure_schur_from_edge_fisher(edge_fisher: np.ndarray) -> float:
    cut = cfi.base.edge_cut_basis(cfi.TRIANGLE_EDGES, 3)
    v = cfi.TRIANGLE_CLOSURE_WEIGHTS
    jcc = float(v.T @ edge_fisher @ v)
    jcg = v.T @ edge_fisher @ cut
    jgg = cut.T @ edge_fisher @ cut
    return max(float(jcc - jcg @ np.linalg.pinv(jgg, rcond=1e-12) @ jcg.T), 0.0)


def edge_first_fisher(g: np.ndarray, s: np.ndarray, fraction: float, pair_fp: float = PAIR_FP_EDGE) -> np.ndarray:
    f = float(fraction)
    loads = np.array([s[0] + s[1], s[1] + s[2], s[2] + s[0]], dtype=float)
    return 4.0 * (f * g) ** 2 / np.maximum(f * loads + pair_fp, 1e-300)


def closure_harmonic(edge_fi: np.ndarray) -> float:
    edge_fi = np.maximum(np.asarray(edge_fi, dtype=float), 1e-300)
    return float(1.0 / np.sum(1.0 / edge_fi))


def lock_gauge_prior(edge_fi: np.ndarray) -> np.ndarray:
    """Gauge Fisher from lock data after marginalizing the unknown closure phase.

    A science-light edge lock does not measure station pistons alone.  Its edge
    phases contain both closure and gauge components, so the closure component
    must be profiled out before the lock data can be used as a gauge prior for
    the science readout.
    """
    w = np.diag(np.asarray(edge_fi, dtype=float))
    cut = cfi.base.edge_cut_basis(cfi.TRIANGLE_EDGES, 3)
    v = cfi.TRIANGLE_CLOSURE_WEIGHTS
    j_phiphi = float(v.T @ w @ v)
    j_psiphi = cut.T @ w @ v
    j_psipsi = cut.T @ w @ cut
    if j_phiphi <= 1e-300:
        return np.zeros((2, 2), dtype=float)
    k_prior = j_psipsi - np.outer(j_psiphi, j_psiphi) / j_phiphi
    return 0.5 * (k_prior + k_prior.T)


def self_lock_fisher(
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    u: float,
    lam: float,
    *,
    mode: str,
) -> float:
    """Closure Fisher for a self-lock split.

    ``mode='feedback'`` is the implementation model requested here: lock-arm
    photons are consumed by the fast lock and enter only as a station-gauge
    prior for the remaining closure readout, but the unknown closure component
    of the lock data is marginalized first.  ``mode='all_data'`` is a
    fundamental-information reference that also recycles all closure information
    in the lock-arm baseline data.
    """
    params = station_params(pair_losses, nus, u)
    s = params["s"]
    g = params["g"]
    lam = float(lam)
    read = max(1.0 - lam, 0.0)

    j_read = edge_qfi_direct(read * g, read * s) if read > 0.0 else np.zeros((3, 3), dtype=float)
    f_lock_edges = edge_first_fisher(g, s, lam, pair_fp=PAIR_FP_LOCK) if lam > 0.0 else np.zeros(3)

    if mode == "all_data":
        return closure_schur_from_edge_fisher(j_read + np.diag(f_lock_edges))
    if mode == "feedback":
        cut = cfi.base.edge_cut_basis(cfi.TRIANGLE_EDGES, 3)
        v = cfi.TRIANGLE_CLOSURE_WEIGHTS
        jcc = float(v.T @ j_read @ v)
        jcg = v.T @ j_read @ cut
        jgg = cut.T @ j_read @ cut + lock_gauge_prior(f_lock_edges)
        return max(float(jcc - jcg @ np.linalg.pinv(jgg, rcond=1e-12) @ jcg.T), 0.0)
    raise ValueError(mode)


def optimize_lambda(
    pair_losses: tuple[float, float, float],
    nus: tuple[float, float, float],
    u: float,
    *,
    mode: str,
    n_grid: int = 501,
) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, n_grid)
    values = np.array([self_lock_fisher(pair_losses, nus, u, lam, mode=mode) for lam in grid])
    best = int(np.argmax(values))
    return float(grid[best]), float(values[best])


def baseline_fishers(pair_losses: tuple[float, float, float], nus: tuple[float, float, float], u: float) -> dict:
    params = station_params(pair_losses, nus, u)
    s = params["s"]
    g = params["g"]
    j_direct = edge_qfi_direct(g, s)
    f_schur = closure_schur_from_edge_fisher(j_direct)
    f_cond = float(cfi.TRIANGLE_CLOSURE_WEIGHTS.T @ j_direct @ cfi.TRIANGLE_CLOSURE_WEIGHTS)
    f_edge_split = closure_harmonic(edge_first_fisher(g, s, 0.5))
    return {
        "f_schur": f_schur,
        "f_conditional": f_cond,
        "f_edge_split": f_edge_split,
        "g": g,
        "s": s,
        "eta": params["eta"],
    }


def scan_scenario(name: str, cfg: dict, rho_grid: np.ndarray) -> dict:
    rows = []
    for rho in rho_grid:
        u = float(rho * EPS_MODE)
        base = baseline_fishers(cfg["pair_losses"], cfg["nus"], u)
        lam_all, f_all = optimize_lambda(cfg["pair_losses"], cfg["nus"], u, mode="all_data")
        lam_feedback, f_feedback = optimize_lambda(cfg["pair_losses"], cfg["nus"], u, mode="feedback")
        rows.append(
            {
                "rho_u_over_eps": float(rho),
                "u": u,
                "F_schur": base["f_schur"],
                "F_conditional": base["f_conditional"],
                "F_edge_split": base["f_edge_split"],
                "lambda_all_data": lam_all,
                "F_all_data_opt": f_all,
                "lambda_feedback": lam_feedback,
                "F_feedback_opt": f_feedback,
                "snr_gain_all_over_schur": float(np.sqrt(f_all / max(base["f_schur"], 1e-300))),
                "snr_gain_feedback_over_schur": float(np.sqrt(f_feedback / max(base["f_schur"], 1e-300))),
                "snr_gain_conditional_over_schur": float(
                    np.sqrt(base["f_conditional"] / max(base["f_schur"], 1e-300))
                ),
                "snr_gain_schur_over_edge": float(np.sqrt(base["f_schur"] / max(base["f_edge_split"], 1e-300))),
            }
        )
    return {"name": name, "config": cfg, "rows": rows}


def make_figure(results: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
    colors = {"LLL": "#bb3e03", "SLL": "#005f73"}

    for result in results:
        short = result["config"]["short"]
        rows = result["rows"]
        rho = np.array([r["rho_u_over_eps"] for r in rows])
        lam_all = np.array([r["lambda_all_data"] for r in rows])
        lam_feedback = np.array([r["lambda_feedback"] for r in rows])
        gain_all = np.array([r["snr_gain_all_over_schur"] for r in rows])
        gain_feedback = np.array([r["snr_gain_feedback_over_schur"] for r in rows])
        gain_cond = np.array([r["snr_gain_conditional_over_schur"] for r in rows])
        gain_edge = np.array([r["snr_gain_schur_over_edge"] for r in rows])
        color = colors[short]

        axes[0, 0].plot(rho, lam_feedback, color=color, lw=2.0, label=f"{short}, lock-only")
        axes[0, 0].plot(rho, lam_all, color=color, lw=1.3, ls=":", label=f"{short}, all-data bound")
        axes[0, 1].plot(rho, gain_feedback, color=color, lw=2.0, label=f"{short}, lock-only")
        axes[0, 1].plot(rho, gain_all, color=color, lw=1.3, ls=":", label=f"{short}, all-data bound")
        axes[0, 1].plot(rho, gain_cond, color=color, lw=1.0, ls=":", label=f"{short}, conditional limit")
        axes[1, 0].plot(rho, gain_edge, color=color, lw=2.0, label=short)

    for ax in axes.flat[:3]:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$\rho=u/\epsilon$")
    axes[0, 0].set_ylabel(r"optimized lock tap $\lambda_{\rm lock,\star}$")
    axes[0, 1].set_ylabel(r"SNR gain over Schur limit")
    axes[1, 0].set_ylabel(r"Schur SNR gain over split edge-first")
    axes[0, 0].set_title("(a) self-lock resource split")
    axes[0, 1].set_title("(b) lock-only implementation gain")
    axes[1, 0].set_title("(c) baseline closure-space gain")
    axes[0, 0].legend(ncol=1, frameon=False)
    axes[0, 1].legend(ncol=1, frameon=False)
    axes[1, 0].legend(frameon=False)

    ax = axes[1, 1]
    for result in results:
        rows = result["rows"]
        short = result["config"]["short"]
        color = colors[short]
        rho = np.array([r["rho_u_over_eps"] for r in rows])
        f_schur = np.array([r["F_schur"] for r in rows])
        f_all = np.array([r["F_all_data_opt"] for r in rows])
        ax.plot(rho, f_schur / np.maximum(f_schur[-1], 1e-300), color=color, lw=1.7, ls="--", label=f"{short}, Schur")
        ax.plot(rho, f_all / np.maximum(f_schur[-1], 1e-300), color=color, lw=2.0, label=f"{short}, self-lock")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$\rho=u/\epsilon$")
    ax.set_ylabel(r"relative FI (normalized)")
    ax.set_title("(d) Fisher scaling")
    ax.legend(frameon=False)

    fig.savefig(FIG_PATH, bbox_inches="tight")
    fig.savefig(FIG_PNG_PATH, dpi=260, bbox_inches="tight")


def format_float(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) > 1e3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def make_latex(stats: dict) -> None:
    representative = stats["representative_rows"]
    table_lines = []
    for row in representative:
        table_lines.append(
            " & ".join(
                [
                    row["case"],
                    format_float(row["rho_u_over_eps"]),
                    format_float(row["F_schur"]),
                    format_float(row["lambda_feedback"], 2),
                    format_float(row["snr_gain_feedback_over_schur"], 3),
                    format_float(row["lambda_all_data"], 2),
                    format_float(row["snr_gain_all_over_schur"], 3),
                    format_float(row["snr_gain_conditional_over_schur"], 3),
                ]
            )
            + r" \\"
        )

    tex = rf"""
\documentclass[10pt]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{amsmath,amssymb,bm,graphicx,booktabs,hyperref}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}}
\newcommand{{\cl}}{{\mathrm{{cl}}}}
\newcommand{{\sep}}{{\mathrm{{sep}}}}
\newcommand{{\opt}}{{\mathrm{{opt}}}}
\newcommand{{\lock}}{{\mathrm{{lock}}}}
\begin{{document}}
\begin{{center}}
{{\Large Three-station Fisher information with science self-locking}}\\[3pt]
{{\normalsize Gauge-marginalized closure readout versus adaptive self-locking}}
\end{{center}}

\section*{{Setup}}
We use the same station-local parameterization as the PRL-style manuscript.  The one-photon
matrix at the hub is
\begin{{equation}}
B=
\begin{{pmatrix}}
s_1 & g_{{12}} & g_{{31}}\\
g_{{12}} & s_2 & g_{{23}}\\
g_{{31}} & g_{{23}} & s_3
\end{{pmatrix}},
\qquad
s_i=\eta_i u+\epsilon_i,\qquad
g_{{ij}}=u\sqrt{{\eta_i\eta_j}}\,\nu_{{ij}} .
\end{{equation}}
The coherent attenuation losses $L_{{ij}}$ are defined by
$\sqrt{{\eta_i\eta_j}}=1-L_{{ij}}$; they are not false-positive rates.  In the numerical examples
we take the independent mode-local background $\epsilon_i=\epsilon={EPS_MODE:.2f}$ and no extra
pair-combiner false-positive load.  The two benchmark triangles are the long-long-long case
LLL, $(L_{{12}},L_{{23}},L_{{31}})=(0.2,0.2,0.2)$ and
$(\nu_{{12}},\nu_{{23}},\nu_{{31}})=(0.1,0.1,0.1)$, and the short-long-long case
SLL, $(0.05,0.2,0.2)$ and $(0.8,0.1,0.1)$.

\section*{{Case 1: no gauge information}}
Let the edge-phase vector be
$\bm\phi=v\Phi+C\bm\psi$, with
\begin{{equation}}
v=\frac13(1,1,1)^{{\mathsf T}},\qquad
C=\begin{{pmatrix}}-1&0\\1&-1\\0&1\end{{pmatrix}} .
\end{{equation}}
If $J$ is the $3\times3$ edge-phase QFI matrix of $B$, the closure Fisher information with no
station-gauge information is the Schur complement
\begin{{equation}}
F_{{\rm Schur}}
=v^{{\mathsf T}}Jv
-v^{{\mathsf T}}JC(C^{{\mathsf T}}JC)^+C^{{\mathsf T}}Jv .
\label{{eq:schur}}
\end{{equation}}
This is the correct closure-only benchmark: if one visibility amplitude vanishes, the Fisher
information for the closure vanishes as well.

\section*{{Case 2: lock-only adaptive implementation}}
Now tap a fraction $\lambda$ of each station mode into a fast edge-phase lock and send the
remaining fraction $1-\lambda$ into the direct closure receiver.  The direct readout branch has
$B_{{\rm r}}=(1-\lambda)B$, hence an edge-QFI matrix $J_{{\rm r}}(\lambda)$.  The lock branch gives
classical edge Fisher informations
\begin{{equation}}
F_{{ij}}^{{\lock}}(\lambda)
=
\frac{{4(\lambda g_{{ij}})^2}}
{{\lambda(s_i+s_j)+\epsilon_{{ij}}^{{\rm pair}}}} .
\end{{equation}}
In the implementation model considered here, the lock-arm photons are consumed by the feedback
loop and are not used in the closure estimator.  However, because they are still science photons,
their edge phases contain the unknown closure phase as well as the station gauge.  The gauge prior
from the lock arm is therefore
\begin{{equation}}
K_\psi^{{\lock}}
=C^{{\mathsf T}}W_\lock C
-C^{{\mathsf T}}W_\lock v
\left(v^{{\mathsf T}}W_\lock v\right)^+
v^{{\mathsf T}}W_\lock C,
\qquad
W_\lock=\operatorname{{diag}}(F_{{12}}^{{\lock}},F_{{23}}^{{\lock}},F_{{31}}^{{\lock}}).
\label{{eq:klock}}
\end{{equation}}
This is the lock-arm gauge information after profiling over the unknown closure.  The realized
closure Fisher is
\begin{{equation}}
F_{{\rm fb}}(\lambda)
=J_{{\Phi\Phi}}^{{\rm r}}
-J_{{\Phi\psi}}^{{\rm r}}
\left[J_{{\psi\psi}}^{{\rm r}}+
K_\psi^{{\lock}}
\right]^+
J_{{\psi\Phi}}^{{\rm r}} .
\label{{eq:feedback}}
\end{{equation}}
Equation~(\ref{{eq:feedback}}) is the relevant implementation Fisher if the lock output is used
only to program the receiver basis.  Using $C^{{\mathsf T}}W_\lock C$ instead of
Eq.~(\ref{{eq:klock}}) would be an oracle approximation: it assumes that the closure contribution to
the lock signal is already known.  For reference, a fundamental all-data bound would also recycle
the closure information in the lock-arm baseline data.  In that accounting the total edge Fisher
matrix is
\begin{{equation}}
J_{{\rm tot}}(\lambda)=
J_{{\rm r}}(\lambda)+
\operatorname{{diag}}(F_{{12}}^{{\lock}},F_{{23}}^{{\lock}},F_{{31}}^{{\lock}}),
\end{{equation}}
and the optimized Fisher information is
\begin{{equation}}
F_{{\rm self}}^\star
=\max_{{0\le\lambda\le1}}
\left[
v^{{\mathsf T}}J_{{\rm tot}}v
-v^{{\mathsf T}}J_{{\rm tot}}C(C^{{\mathsf T}}J_{{\rm tot}}C)^+
C^{{\mathsf T}}J_{{\rm tot}}v
\right].
\label{{eq:self_opt}}
\end{{equation}}
The distinction matters.  Equation~(\ref{{eq:feedback}}) answers the experimental question
``how much closure-readout performance is recovered if a fraction of the photons is sacrificed to
hold the gauge frame?''  Equation~(\ref{{eq:self_opt}}) is only a bookkeeping upper bound in which
no information from the lock measurement is discarded.

\section*{{Numerical optimization}}
The scan below uses the dimensionless brightness $\rho=u/\epsilon$ and optimizes $\lambda$ on a
uniform grid of 501 points.  The table reports representative points; gains are SNR gains relative
to the Schur-limit closure Fisher of Case 1.  The ``lock'' columns are the requested implementation
model of Eq.~(\ref{{eq:feedback}}).  The ``all'' columns are shown only as a fundamental-information
reference in which the lock data is also used for closure science.

\begin{{center}}
\begin{{tabular}}{{lrrrrrrr}}
\toprule
case & $\rho$ & $F_{{\rm Schur}}$ &
$\lambda_\star^{{\rm lock}}$ &
$\sqrt{{F_{{\rm lock}}^\star/F_{{\rm Schur}}}}$ &
$\lambda_\star^{{\rm all}}$ &
$\sqrt{{F_{{\rm all}}^\star/F_{{\rm Schur}}}}$ &
$\sqrt{{F_{{\rm cond}}/F_{{\rm Schur}}}}$\\
\midrule
{chr(10).join(table_lines)}
\bottomrule
\end{{tabular}}
\end{{center}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.84\linewidth]{{../figures/three_station_self_lock_fisher.pdf}}
\caption{{Self-lock optimization for the two PRL-style three-station benchmarks.  The lock-only
curve is the requested implementation model: tapped photons are used only to estimate the gauge
frame and are not reused in the closure estimator.  The all-data curve is a fundamental-information
reference in which the lock measurement is also used as science data.}}
\end{{figure}}

\section*{{Takeaway}}
For the requested lock-only implementation, self-locking can improve the realized Fisher
information of a gauge-dependent closure sorter, especially for asymmetric SLL loops where the
conditional known-gauge Fisher is much larger than the Schur-limit value.  This does not invalidate
the Schur complement as a fundamental closure-only bound: the all-data reference still optimizes at
$\lambda_\star=0$ because it does not discard any information.  The lock-only optimum instead
quantifies a practical tradeoff between sacrificing photons for gauge control and preserving photons
for the closure readout.

\end{{document}}
"""
    TEX_PATH.write_text(tex.strip() + "\n")


def main() -> None:
    scenarios = {
        "three long baselines": {
            "short": "LLL",
            "pair_losses": (0.20, 0.20, 0.20),
            "nus": (0.10, 0.10, 0.10),
        },
        "one short + two long": {
            "short": "SLL",
            "pair_losses": (0.05, 0.20, 0.20),
            "nus": (0.80, 0.10, 0.10),
        },
    }
    rho_grid = np.logspace(-4, 2, 121)
    results = [scan_scenario(name, cfg, rho_grid) for name, cfg in scenarios.items()]
    make_figure(results)

    representative_rhos = [1e-3, 1e-1, 1.0, 10.0]
    representative_rows = []
    for result in results:
        rows = result["rows"]
        rho_values = np.array([row["rho_u_over_eps"] for row in rows])
        for rho in representative_rhos:
            row = dict(rows[int(np.argmin(np.abs(np.log10(rho_values) - np.log10(rho))))])
            row["case"] = result["config"]["short"]
            representative_rows.append(row)

    stats = {
        "mode_false_positive_epsilon": EPS_MODE,
        "pair_false_positive_lock": PAIR_FP_LOCK,
        "pair_false_positive_edge": PAIR_FP_EDGE,
        "rho_grid": rho_grid.tolist(),
        "scenarios": results,
        "representative_rows": representative_rows,
        "figure_pdf": str(FIG_PATH.relative_to(ROOT)),
        "figure_png": str(FIG_PNG_PATH.relative_to(ROOT)),
        "tex": str(TEX_PATH.relative_to(ROOT)),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    make_latex(stats)
    print(FIG_PATH)
    print(FIG_PNG_PATH)
    print(STATS_PATH)
    print(TEX_PATH)
    print(
        json.dumps(
            {
                "representative_rows": representative_rows,
                "summary": {
                    result["config"]["short"]: {
                        "lambda_all_data_min": min(row["lambda_all_data"] for row in result["rows"]),
                        "lambda_all_data_max": max(row["lambda_all_data"] for row in result["rows"]),
                        "gain_all_max": max(row["snr_gain_all_over_schur"] for row in result["rows"]),
                        "gain_feedback_max": max(row["snr_gain_feedback_over_schur"] for row in result["rows"]),
                    }
                    for result in results
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
