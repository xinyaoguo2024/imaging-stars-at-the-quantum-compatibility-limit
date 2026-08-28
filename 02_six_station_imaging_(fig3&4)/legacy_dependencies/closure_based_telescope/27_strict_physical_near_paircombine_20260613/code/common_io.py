from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
WORKSPACE = THIS_DIR.parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOTES = ROOT / "notes"
LOGS = ROOT / "logs"
SOURCE18 = WORKSPACE / "18_balanced_10loop_independent_set_20260611"
SOURCE16 = WORKSPACE / "16_six_station_reduced_from7_20260611"

for folder in (RESULTS, FIGURES, NOTES, LOGS, LOGS / "mplconfig"):
    folder.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_b18():
    module = load_module(
        "balanced10_remote_star",
        SOURCE18 / "code" / "run_remote_star_joint_near_benchmark.py",
    )
    module.variants.configure_six_station_constants()
    bm = module.variants.configure_six_benchmark()
    return module, bm


@contextlib.contextmanager
def good_runtime(b18):
    with b18.variants.fig_run.morph.patched_variant(
        b18.variants.fig_run.GOOD_VARIANT
    ), b18.variants.fig_run.ngc.patched_source(b18.variants.fig_run.GOOD_SOURCE):
        yield


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ratio_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def load_baseline_rows() -> list[dict[str, str]]:
    return list(csv.DictReader((SOURCE18 / "results" / "remote_star_joint_loop_gains.csv").open()))


def load_baselines() -> dict[str, object]:
    rows = load_baseline_rows()
    return {
        "rows": rows,
        "labels": [row["loop"] for row in rows],
        "direct_gain": np.asarray([float(row["snr_gain_direct_optimized_schedule_vs_edge"]) for row in rows]),
        "direct_sigma": np.asarray([float(row["rms_direct_optimized_schedule_rad"]) for row in rows]),
        "old_near_gain": np.asarray([float(row["snr_gain_current_near_vs_edge"]) for row in rows]),
        "old_star_gain": np.asarray([float(row["snr_gain_remote_star_independent_vs_edge"]) for row in rows]),
        "old_near_sigma": np.asarray([float(row["rms_current_near_rad"]) for row in rows]),
        "old_star_sigma": np.asarray([float(row["rms_remote_star_independent_rad"]) for row in rows]),
    }


def load_saved_split_payload() -> dict[str, object]:
    return json.loads((SOURCE18 / "results" / "balanced10_near_split_payload.json").read_text())


def load_saved_summary18() -> dict[str, object]:
    return json.loads((SOURCE18 / "results" / "remote_star_joint_near_summary.json").read_text())["summary"]


def reproduce_saved_columns(b18, bm, baselines: dict[str, object]) -> dict[str, float]:
    payload = load_saved_split_payload()
    p = np.asarray(payload["split_matrix"], dtype=float)
    alpha = np.asarray(payload["alpha_core"], dtype=float)
    gamma = np.asarray(load_saved_summary18()["independent_gamma_vector"], dtype=float)

    current_sigma = b18.root_sigmas(bm, b18.fisher_current_near(bm, p, alpha))
    star_fisher, _ = b18.fisher_remote_star_near(
        bm,
        p,
        alpha,
        gamma,
        core_core_handling="nuisance",
    )
    star_sigma = b18.root_sigmas(bm, star_fisher)
    current_err = float(np.max(np.abs(current_sigma / np.asarray(baselines["old_near_sigma"]) - 1.0)))
    star_err = float(np.max(np.abs(star_sigma / np.asarray(baselines["old_star_sigma"]) - 1.0)))
    if current_err > 1.0e-10 or star_err > 1.0e-10:
        raise RuntimeError(f"Folder-18 reproduction failed: current={current_err:g}, star={star_err:g}")
    return {
        "current_near_max_relative_error": current_err,
        "remote_star_independent_max_relative_error": star_err,
    }
