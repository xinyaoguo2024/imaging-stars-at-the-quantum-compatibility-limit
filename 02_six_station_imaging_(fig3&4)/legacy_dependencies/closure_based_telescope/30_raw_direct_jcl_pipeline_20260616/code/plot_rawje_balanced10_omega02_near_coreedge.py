from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rawje_balanced10_helpers as h


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

STEM = "rawje_balanced10_omega02_near_coreedge"


def loop_class(tri: tuple[int, int, int]) -> str:
    n_remote = sum(station in h.REMOTE for station in tri)
    if n_remote == 0:
        return "core"
    if n_remote == 1:
        return "one remote"
    return "two remote"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_rows(
    bm: h.RawJeBenchmark,
    edge_sigma: np.ndarray,
    direct_sigma: np.ndarray,
    direct_per_sample_sigma: np.ndarray,
    near_sigma: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for idx, tri in enumerate(h.BALANCED10):
        rows.append(
            {
                "loop": h.loop_label(tri),
                "loop_class": loop_class(tri),
                "omega_l": 0.2,
                "rms_edge_rad": float(edge_sigma[idx]),
                "rms_rawdirect_global_schur_rad": float(direct_sigma[idx]),
                "rms_rawdirect_per_sample_schur_rad": float(direct_per_sample_sigma[idx]),
                "rms_near_coreedge_rad": float(near_sigma[idx]),
                "snr_gain_rawdirect_vs_edge": float(edge_sigma[idx] / max(direct_sigma[idx], 1.0e-300)),
                "snr_gain_rawdirect_per_sample_vs_edge": float(
                    edge_sigma[idx] / max(direct_per_sample_sigma[idx], 1.0e-300)
                ),
                "snr_gain_near_coreedge_vs_edge": float(edge_sigma[idx] / max(near_sigma[idx], 1.0e-300)),
                "snr_ratio_near_over_rawdirect": float(direct_sigma[idx] / max(near_sigma[idx], 1.0e-300)),
            }
        )
    return rows


def plot_rows(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    labels = [str(row["loop"]) for row in rows]
    direct = np.asarray([float(row["snr_gain_rawdirect_vs_edge"]) for row in rows])
    near = np.asarray([float(row["snr_gain_near_coreedge_vs_edge"]) for row in rows])
    classes = [str(row["loop_class"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 3.9), constrained_layout=True)
    ax.bar(x - 0.5 * width, direct, width, color="#9d0208", label=r"raw $J_e$ direct, $\omega_\ell=0.2$")
    ax.bar(x + 0.5 * width, near, width, color="#2a9d8f", label="core-direct + remote edge near")
    ax.axhline(1.0, color="0.25", lw=1.0, ls="--")
    for idx, cls in enumerate(classes):
        if idx > 0 and cls != classes[idx - 1]:
            ax.axvline(idx - 0.5, color="0.82", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("SNR gain vs uniform edge-first")
    ax.set_ylim(0.8, max(1.55, 1.08 * float(max(np.max(direct), np.max(near)))))
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{STEM}.{suffix}", dpi=260)
    plt.close(fig)
    return FIGURES / f"{STEM}.pdf", FIGURES / f"{STEM}.png"


def main() -> None:
    bm = h.make_benchmark()
    edge_fisher = h.uniform_edge_fisher(bm)
    direct_fisher = h.rawdirect_balanced10_fisher(bm, weight=0.2, schur_per_sample=False)
    direct_per_sample_fisher = h.rawdirect_balanced10_fisher(bm, weight=0.2, schur_per_sample=True)

    edge_sigma = h.loop_sigmas(bm, edge_fisher)
    direct_sigma = h.loop_sigmas(bm, direct_fisher)
    direct_per_sample_sigma = h.loop_sigmas(bm, direct_per_sample_fisher)

    near_p, near_alpha, near_sigma, near_info = h.optimize_near_coreedge(
        bm,
        edge_sigma,
        direct_sigma,
        seed=20260616,
    )

    rows = make_rows(bm, edge_sigma, direct_sigma, direct_per_sample_sigma, near_sigma)
    csv_path = RESULTS / f"{STEM}_loop_gains.csv"
    write_csv(csv_path, rows)
    pdf_path, png_path = plot_rows(rows)

    direct_station_sums = {
        f"S{i + 1}": float(sum(0.2 for tri in h.BALANCED10 if i in tri))
        for i in range(bm.n)
    }
    payload = {
        "definition": {
            "loop_set": [h.loop_label(tri) for tri in h.BALANCED10],
            "direct_model": "selected balanced 10 triangle raw edge Fisher branches with omega_l=0.2, assembled as raw J_e and Schur-complemented once globally",
            "near_model": "S1-S3 core raw direct block plus pairwise edge-first readout on all non-core-core baselines, optimized against the fixed raw-direct target",
            "snr_gain": "sigma_edge_uniform / sigma_strategy for each displayed loop",
            "exposure_s": float(h.aug.EXPOSURE_S),
            "n_samples": int(len(bm.samples)),
        },
        "direct_station_weight_sums": direct_station_sums,
        "summary": {
            "rawdirect_gain_vs_edge": h.gain_stats(
                np.asarray([float(row["snr_gain_rawdirect_vs_edge"]) for row in rows])
            ),
            "rawdirect_per_sample_gain_vs_edge": h.gain_stats(
                np.asarray([float(row["snr_gain_rawdirect_per_sample_vs_edge"]) for row in rows])
            ),
            "near_gain_vs_edge": h.gain_stats(
                np.asarray([float(row["snr_gain_near_coreedge_vs_edge"]) for row in rows])
            ),
            "near_over_rawdirect_snr_ratio": h.gain_stats(
                np.asarray([float(row["snr_ratio_near_over_rawdirect"]) for row in rows])
            ),
        },
        "near_info": near_info,
        "near_alpha_core": [float(x) for x in near_alpha],
        "near_split_matrix": np.asarray(near_p, dtype=float).tolist(),
        "outputs": {
            "loop_gains_csv": str(csv_path),
            "figure_pdf": str(pdf_path),
            "figure_png": str(png_path),
        },
    }
    json_path = RESULTS / f"{STEM}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    split_path = RESULTS / f"{STEM}_near_strategy.json"
    split_path.write_text(
        json.dumps(
            {
                "alpha_core": [float(x) for x in near_alpha],
                "split_matrix": np.asarray(near_p, dtype=float).tolist(),
                "info": near_info,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(payload["summary"], indent=2))
    print(csv_path)
    print(json_path)
    print(split_path)
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
