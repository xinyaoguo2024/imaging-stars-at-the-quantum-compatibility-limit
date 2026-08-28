from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_20260608_common import FIG_DIAG_DIR, OUT, newest_matching, read_csv, write_csv


STRATEGIES = [
    ("all", "all", "#6a4c93"),
    ("edge_uniform", "edge", "#0077b6"),
    ("core4_remote_optimized", "near", "#f77f00"),
    ("nmode_joint_scheduled", "old sched.", "#d00000"),
]


def float_col(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def full_qfi_normalized_rows(
    ratio_rows: list[dict[str, str]],
    direct_edge_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    by_loop = {row["loop"]: row for row in direct_edge_rows}
    out: list[dict[str, object]] = []
    for row in ratio_rows:
        loop = row["loop"]
        audit = by_loop[loop]
        full_over_local = float(audit["direct_full_qfi_over_local_raw_fisher"])
        local_rms_over_full_rms = math.sqrt(full_over_local)
        scheduled_over_local = float(row["scheduled_proxy_over_direct_rms_mean"])
        near_over_local = float(row["near_over_direct_rms_mean"])
        edge_over_local = float(row["edge_over_direct_rms_mean"])
        out.append(
            {
                "loop": loop,
                "full_direct_qfi_over_full_direct_qfi_rms_mean": 1.0,
                "near_over_full_direct_qfi_rms_mean": near_over_local * local_rms_over_full_rms,
                "near_over_full_direct_qfi_rms_std_approx": float(row["near_over_direct_rms_std"]) * local_rms_over_full_rms,
                "edge_over_full_direct_qfi_rms_mean": edge_over_local * local_rms_over_full_rms,
                "edge_over_full_direct_qfi_rms_std_approx": float(row["edge_over_direct_rms_std"]) * local_rms_over_full_rms,
                "old_scheduled_proxy_over_full_direct_qfi_rms_mean": scheduled_over_local * local_rms_over_full_rms,
                "local_triangle_raw_over_full_direct_qfi_rms_mean": local_rms_over_full_rms,
                "direct_full_qfi_over_local_raw_fisher": full_over_local,
                "edge_full_over_harmonic_fisher": float(audit["edge_full_over_harmonic_fisher"]),
            }
        )
    return out


def plot_compact(seed_rows: list[dict[str, str]], norm_rows: list[dict[str, object]]) -> tuple[str, str, str, str]:
    plt.rcParams.update(
        {
            "font.size": 6.0,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.4,
            "legend.fontsize": 5.0,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.55), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(STRATEGIES), dtype=float)
    means = []
    stds = []
    for strategy, _label, _color in STRATEGIES:
        values = float_col(seed_rows, f"{strategy}_blr_corr")
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values, ddof=1)))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=[color for _strategy, _label, color in STRATEGIES],
        alpha=0.72,
        width=0.62,
        edgecolor="black",
        linewidth=0.45,
        error_kw={"lw": 0.45, "capsize": 1.4},
    )
    rng = np.random.default_rng(20260608)
    for idx, (strategy, _label, color) in enumerate(STRATEGIES):
        values = float_col(seed_rows, f"{strategy}_blr_corr")
        ax.scatter(
            np.full(len(values), x[idx]) + rng.uniform(-0.075, 0.075, size=len(values)),
            values,
            s=8,
            facecolor="white",
            edgecolor=color,
            linewidth=0.55,
            zorder=4,
        )
    ax.set_xticks(x, [label for _strategy, label, _color in STRATEGIES], rotation=20, ha="right")
    ax.set_ylabel("BLR correspondence")
    all_blr = np.concatenate([float_col(seed_rows, f"{s}_blr_corr") for s, _l, _c in STRATEGIES])
    ax.set_ylim(max(0.0, float(np.min(all_blr)) - 0.05), min(1.0, float(np.max(all_blr)) + 0.05))
    ax.grid(axis="y", color="0.88", lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title("5 seeds")

    ax = axes[1]
    labels = [str(row["loop"]) for row in norm_rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    edge = np.asarray([float(row["edge_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    near = np.asarray([float(row["near_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    old_sched = np.asarray([float(row["old_scheduled_proxy_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    ax.bar(x - 0.5 * width, edge, width=width, color="#0077b6", alpha=0.78, label="edge", edgecolor="black", linewidth=0.35)
    ax.bar(x + 0.5 * width, near, width=width, color="#f77f00", alpha=0.78, label="near", edgecolor="black", linewidth=0.35)
    ax.axhline(1.0, color="#d00000", lw=0.65, ls=":", label="full QFI")
    ax.plot(x, old_sched, color="0.45", lw=0.65, ls="--", marker="o", ms=2.0, label="old 2/5")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMS / full direct")
    ax.set_ylim(0.0, max(float(np.max(edge)), float(np.max(near)), float(np.max(old_sched)), 1.0) * 1.12)
    ax.grid(axis="y", color="0.88", lw=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderpad=0.1)
    ax.set_title("closure RMS")

    audit_png = OUT / "fig3b_full_qfi_normalized_200ms_minphase.png"
    audit_pdf = OUT / "fig3b_full_qfi_normalized_200ms_minphase.pdf"
    stable_png = FIG_DIAG_DIR / "fig2_compact_seed_loop_diagnostic_singlecol_200ms_minphase_fullqfi.png"
    stable_pdf = FIG_DIAG_DIR / "fig2_compact_seed_loop_diagnostic_singlecol_200ms_minphase_fullqfi.pdf"
    for path in (audit_png, stable_png):
        fig.savefig(path, dpi=320, bbox_inches="tight")
    for path in (audit_pdf, stable_pdf):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(audit_pdf), str(audit_png), str(stable_pdf), str(stable_png)


def run_fig3_full_qfi_checks() -> dict[str, object]:
    seed_csv = newest_matching(
        "exploration/fig2_current_seed_diagnostics/"
        "fig2_current_5seed_correspondence_metrics_*200ms*minphase.csv"
    )
    ratio_csv = newest_matching(
        "exploration/fig2_current_seed_diagnostics/"
        "fig2_loop_123_125_127_compact_ratio_diagnosis_*200ms*minphase.csv"
    )
    direct_edge_csv = OUT / "direct_edge_full_array_scalar_audit_200ms_minphase.csv"
    if not direct_edge_csv.exists():
        from audit_20260608_direct_edge_models import run_direct_edge_model_audit

        run_direct_edge_model_audit()
    seed_rows = read_csv(seed_csv)
    ratio_rows = read_csv(ratio_csv)
    direct_edge_rows = read_csv(direct_edge_csv)
    norm_rows = full_qfi_normalized_rows(ratio_rows, direct_edge_rows)
    norm_csv = OUT / "fig3b_full_qfi_normalized_200ms_minphase.csv"
    write_csv(norm_csv, norm_rows)
    audit_pdf, audit_png, stable_pdf, stable_png = plot_compact(seed_rows, norm_rows)

    near_ratios = np.asarray([float(row["near_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    edge_ratios = np.asarray([float(row["edge_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    old_sched = np.asarray([float(row["old_scheduled_proxy_over_full_direct_qfi_rms_mean"]) for row in norm_rows])
    return {
        "seed_csv": str(seed_csv),
        "ratio_csv_raw_direct_normalized": str(ratio_csv),
        "direct_edge_audit_csv": str(direct_edge_csv),
        "full_qfi_normalized_csv": str(norm_csv),
        "audit_pdf": audit_pdf,
        "audit_png": audit_png,
        "stable_paper_pdf": stable_pdf,
        "stable_paper_png": stable_png,
        "near_over_full_direct_qfi_rms_mean_range": [float(np.min(near_ratios)), float(np.max(near_ratios))],
        "edge_over_full_direct_qfi_rms_mean_range": [float(np.min(edge_ratios)), float(np.max(edge_ratios))],
        "old_scheduled_proxy_over_full_direct_qfi_rms_mean": float(np.mean(old_sched)),
    }


if __name__ == "__main__":
    result = run_fig3_full_qfi_checks()
    print(result["full_qfi_normalized_csv"])
    print(result["audit_pdf"])
    print(f"near_range={result['near_over_full_direct_qfi_rms_mean_range']}")
    print(f"edge_range={result['edge_over_full_direct_qfi_rms_mean_range']}")
