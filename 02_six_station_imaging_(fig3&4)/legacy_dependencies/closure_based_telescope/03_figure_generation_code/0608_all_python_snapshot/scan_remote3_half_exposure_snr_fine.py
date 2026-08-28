from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import scan_remote3_half_exposure_snr_advantage as coarse


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_half_exposure_snr_fine_20260525"
OUT.mkdir(parents=True, exist_ok=True)

FINE_SNR_GRID = [0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]


def plot_fine_scan(rows: list[dict], advantage_rows: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), constrained_layout=True)
    colors = {"all": "#8d99ae", "split": "#0077b6", "direct": "#d00000"}
    for strategy in ("all", "split", "direct"):
        srows = sorted([row for row in rows if row["strategy"] == strategy], key=lambda row: row["snr_boost"])
        x = [row["snr_boost"] for row in srows]
        axes[0, 0].plot(x, [row["blr_corr"] for row in srows], "o-", color=colors[strategy], label=strategy)
        axes[0, 1].plot(x, [row["profile_rmse"] for row in srows], "o-", color=colors[strategy], label=strategy)
        axes[1, 0].plot(x, [row["phase_chi2"] for row in srows], "o-", color=colors[strategy], label=strategy)
    adv = sorted(advantage_rows, key=lambda row: row["snr_boost"])
    x = [row["snr_boost"] for row in adv]
    axes[1, 1].plot(x, [row["direct_minus_split_blr_corr"] for row in adv], "o-", color="#d00000", label=r"$\Delta$ BLR corr")
    axes[1, 1].plot(
        x,
        [row["split_over_direct_profile_rmse"] - 1.0 for row in adv],
        "s-",
        color="#0077b6",
        label="profile RMSE gain - 1",
    )
    axes[1, 1].axhline(0.0, color="0.3", lw=0.8)
    for ax in axes.ravel():
        ax.set_xlabel("global SNR boost")
        ax.grid(alpha=0.25)
    axes[0, 0].set_title("BLR correlation")
    axes[0, 1].set_title("BLR profile RMSE")
    axes[1, 0].set_title("phase reduced chi-square")
    axes[1, 1].set_title("direct advantage over edge-first")
    axes[0, 0].set_ylabel("higher is better")
    axes[0, 1].set_ylabel("lower is better")
    axes[1, 0].set_ylabel("lower is better")
    axes[1, 1].set_ylabel("positive favors direct")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Remote3 fine SNR scan at half per-sample exposure", weight="bold")
    png = OUT / "remote3_half_exposure_snr_fine.png"
    pdf = OUT / "remote3_half_exposure_snr_fine.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    scan_rows: list[dict] = []
    for snr in FINE_SNR_GRID:
        print(f"[fine snr scan] boost={snr:g}", flush=True)
        dataset = coarse.run_one_setting(
            exposure_s=300.0,
            snr_boost=snr,
            adam_iter=1100,
            two_start=False,
            scan_mode="fine_snr_scan_dirty_start",
        )
        scan_rows.extend(dataset["rows"])
    advantage_rows = coarse.summarize_advantage(scan_rows)
    fig_pdf, fig_png = plot_fine_scan(scan_rows, advantage_rows)

    scan_csv = OUT / "remote3_half_exposure_snr_fine_metrics.csv"
    with scan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    advantage_csv = OUT / "remote3_half_exposure_snr_fine_advantage.csv"
    with advantage_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(advantage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(advantage_rows)
    best_blr = max(advantage_rows, key=lambda row: row["direct_minus_split_blr_corr"])
    best_rmse = max(advantage_rows, key=lambda row: row["split_over_direct_profile_rmse"])
    payload = {
        "runtime": {
            "snr_grid": FINE_SNR_GRID,
            "exposure_s": 300.0,
            "adam_iter": 1100,
            "two_start": False,
            "prior": coarse.REMOTE3_PRIOR.prior,
            "tv": coarse.REMOTE3_PRIOR.tv,
            "entropy": coarse.REMOTE3_PRIOR.entropy,
        },
        "note": "Fine trend scan only: dirty-start single-start RML, fixed remote3 optimized prior.",
        "best_direct_minus_split_blr_corr": best_blr,
        "best_split_over_direct_profile_rmse": best_rmse,
        "advantage_rows": advantage_rows,
        "figures": {
            "png": str(fig_png),
            "pdf": str(fig_pdf),
            "metrics_csv": str(scan_csv),
            "advantage_csv": str(advantage_csv),
        },
    }
    json_path = OUT / "remote3_half_exposure_snr_fine_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "# Remote3 fine SNR scan\n\n"
        "Fine scan around the coarse optimum at SNR_BOOST ~0.35-0.5.  This is a dirty-start\n"
        "single-start diagnostic with 1100 Adam steps, fixed remote3 optimized prior, and 5 min samples.\n"
    )
    print(fig_png)
    print(advantage_csv)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
