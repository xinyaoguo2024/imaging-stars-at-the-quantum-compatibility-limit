from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED_RUNS = RESULTS / "fig3a_seed_runs"
SEEDS = [
    int(item.strip())
    for item in os.environ.get(
        "FIG3_SEEDS",
        "20260529,20260530,20260531,20260532,20260533",
    ).split(",")
    if item.strip()
]
EXPOSURES = [
    item.strip()
    for item in os.environ.get("FIG3_EXPOSURES", "100ms,50ms").split(",")
    if item.strip()
]

STRATEGY_ORDER = [
    ("all", "All visibilities + drift"),
    ("edge_uniform", "Edge-first closure"),
    ("core4_remote_optimized", "Strict near pair-combine"),
    ("nmode_joint_scheduled", "Direct optimized closure"),
]


def run_tag(label: str, seed: int) -> str:
    return f"fig3a_{label}_seed{seed}_paired10loop"


def summary_path(label: str, seed: int) -> Path:
    stem = f"broad_plume_split_objective_nmode_rml_{run_tag(label, seed)}_summary.json"
    return SEED_RUNS / label / f"seed_{seed}" / "rml_outputs" / stem


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for exposure_label in EXPOSURES:
        for seed in SEEDS:
            path = summary_path(exposure_label, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            data = json.loads(path.read_text())
            paired = bool(data.get("stats", {}).get("paired_loop_noise", False))
            exposure_s = float(data.get("stats", {}).get("sample_stress_test", {}).get("effective_exposure_s", 0.0))
            for row in data["rows"]:
                rows.append(
                    {
                        "exposure_label": exposure_label,
                        "exposure_s": exposure_s,
                        "seed": seed,
                        "paired_loop_noise": paired,
                        "strategy": row["strategy"],
                        "label": row["label"],
                        "blr_corr": float(row["blr_corr"]),
                        "global_corr": float(row["global_corr"]),
                        "profile_rmse": float(row["profile_rmse"]),
                        "amp_chi2": float(row["amp_chi2"]),
                        "phase_chi2": float(row["phase_chi2"]),
                        "summary_path": str(path),
                    }
                )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["exposure_label"]), str(row["strategy"]))].append(row)
    out: list[dict[str, object]] = []
    for exposure_label in EXPOSURES:
        for strategy, label in STRATEGY_ORDER:
            vals = grouped[(exposure_label, strategy)]
            if len(vals) != len(SEEDS):
                raise ValueError(f"Missing rows for {exposure_label} {strategy}: found {len(vals)}")
            for metric in ("blr_corr", "global_corr", "profile_rmse", "amp_chi2", "phase_chi2"):
                arr = np.asarray([float(row[metric]) for row in vals], dtype=float)
                out.append(
                    {
                        "exposure_label": exposure_label,
                        "strategy": strategy,
                        "label": label,
                        "metric": metric,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=1)),
                        "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                        "n_seed": len(arr),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                )
    return out


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_csv(RESULTS / "fig3a_5seed_seed_metrics.csv", rows)
    write_csv(RESULTS / "fig3a_5seed_summary.csv", summary)
    (RESULTS / "fig3a_5seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary:
        if row["metric"] in {"blr_corr", "global_corr"}:
            print(
                f"{row['exposure_label']} {row['strategy']} {row['metric']}: "
                f"{row['mean']:.4f} +/- {row['std']:.4f}"
            )


if __name__ == "__main__":
    main()
