from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_prl_broadband_blr_optimized as opt
import scan_remote3_half_exposure_snr_advantage as coarse


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_remote3_half_exposure_snr_fine_20260525"
OUT.mkdir(parents=True, exist_ok=True)

STRATEGIES = coarse.STRATEGIES
SNR = 0.35


def plot_dataset(dataset: dict, *, tag: str, title: str):
    axis = dataset["axis_uas"]
    truth = dataset["truth"]
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    result_by = {item["strategy"]: item for item in dataset["results"]}
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

    theta, truth_prof = coarse.angular_profile(truth, axis)
    axes[1, 0].plot(np.rad2deg(theta), truth_prof, color="black", lw=2.2, label="input")
    for strategy, label, _start, color in STRATEGIES:
        _, prof = coarse.angular_profile(result_by[strategy]["best"]["image"], axis)
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

    fig.suptitle(title, weight="bold")
    png = OUT / f"{tag}.png"
    pdf = OUT / f"{tag}.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def rows_from_dataset(dataset: dict, mode: str) -> list[dict]:
    rows = dataset["rows"]
    for row in rows:
        row["mode"] = mode
    return rows


def main() -> None:
    shutil.copyfile(Path(__file__), OUT / Path(__file__).name)
    print("[single-start exact fine-scan-like run]", flush=True)
    single = coarse.run_one_setting(
        exposure_s=300.0,
        snr_boost=SNR,
        adam_iter=1100,
        two_start=False,
        scan_mode="snr035_single_start_image",
    )
    single_pdf, single_png = plot_dataset(
        single,
        tag="remote3_snr035_max_advantage_single_start_reconstruction",
        title="Remote3 maximum-advantage point: SNR boost 0.35, single dirty-start",
    )

    print("[two-start stability check]", flush=True)
    two = coarse.run_one_setting(
        exposure_s=300.0,
        snr_boost=SNR,
        adam_iter=1600,
        two_start=True,
        scan_mode="snr035_two_start_image",
    )
    two_pdf, two_png = plot_dataset(
        two,
        tag="remote3_snr035_max_advantage_two_start_reconstruction",
        title="Remote3 maximum-advantage point: SNR boost 0.35, two-start check",
    )

    rows = rows_from_dataset(single, "single_start") + rows_from_dataset(two, "two_start")
    csv_path = OUT / "remote3_snr035_max_advantage_reconstruction_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "snr_boost": SNR,
        "exposure_s": 300.0,
        "note": (
            "The single-start image matches the fine-scan protocol. The two-start image is a stability check "
            "and may shift small image metrics because the RML problem has local minima."
        ),
        "figures": {
            "single_start_png": str(single_png),
            "single_start_pdf": str(single_pdf),
            "two_start_png": str(two_png),
            "two_start_pdf": str(two_pdf),
            "metrics_csv": str(csv_path),
        },
        "single_start_rows": single["rows"],
        "two_start_rows": two["rows"],
    }
    json_path = OUT / "remote3_snr035_max_advantage_reconstruction_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(single_png)
    print(two_png)
    print(csv_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
