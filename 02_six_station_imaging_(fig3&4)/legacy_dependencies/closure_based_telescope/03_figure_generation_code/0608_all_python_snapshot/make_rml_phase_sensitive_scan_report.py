from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUT = ROOT / "rml_phase_sensitive_source_amp_scan_20260525"


def read_rows() -> list[dict[str, str]]:
    with (OUT / "stress_test_metrics.csv").open() as f:
        return list(csv.DictReader(f))


def add_image_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.imshow(img)
    ax.axis("off")
    fig.suptitle(title, fontsize=15, weight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, rows: list[dict[str, str]]) -> None:
    columns = [
        "config",
        "strategy",
        "global_corr",
        "blr_corr",
        "profile_rmse",
        "phase_chi2",
        "amp_chi2",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["config"],
                row["strategy"],
                f"{float(row['global_corr']):.3f}",
                f"{float(row['blr_corr']):.3f}",
                f"{float(row['profile_rmse']):.3f}",
                f"{float(row['phase_chi2']):.3f}",
                f"{float(row['amp_chi2']):.3f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.axis("off")
    table = ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#edf2f4")
        if col_idx in {2, 3, 4, 5, 6} and row_idx > 0:
            cell.set_facecolor("#fbfbfb")
    ax.set_title("Numerical metrics", fontsize=15, weight="bold", pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = read_rows()
    report = OUT / "rml_phase_sensitive_scan_report.pdf"
    with PdfPages(report) as pdf:
        add_image_page(pdf, OUT / "stress_test_metric_summary.png", "Metric summary")
        add_table_page(pdf, rows)
        for config in ("crescent_ampdom", "crescent_phaseled", "spotted_ampdom", "spotted_phaseled"):
            add_image_page(pdf, OUT / f"{config}_images_residuals.png", f"{config}: images and residuals")
            add_image_page(pdf, OUT / f"{config}_blr_profile.png", f"{config}: BLR annular profile")
    print(report)


if __name__ == "__main__":
    main()
