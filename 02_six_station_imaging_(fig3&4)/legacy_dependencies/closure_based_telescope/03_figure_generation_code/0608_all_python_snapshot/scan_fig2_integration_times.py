from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[2]
ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
PARENT_OUT = ROOT / "rml_remote3_broad_plume_split_objective_20260527"
FIG2_SCRIPT = BUNDLE / "code" / "all_python_snapshot" / "run_broad_plume_split_objective_rml.py"
DEFAULT_OUT = BUNDLE / "exploration" / "fig2_integration_time_scan" / "commonnoise_seed20260529_fit40_i2600_p0p01_tv0p01"


def read_metrics(path: Path, minutes: int, tag: str) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out: dict[str, str | int | float] = {
                "minutes_per_sample": minutes,
                "run_tag": tag,
            }
            for key, value in row.items():
                if key in {"strategy", "label", "best_start"}:
                    out[key] = value
                else:
                    out[key] = float(value)
            rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(minutes: int, out_dir: Path, *, force: bool) -> list[dict[str, str | int | float]]:
    scale = minutes / 10.0
    tag = f"5night_y0p85_seed20260529_fit40_i2600_p0p01_tv0p01_8panel_commonnoise_{minutes:02d}min"
    stem = f"broad_plume_split_objective_nmode_rml_{tag}"
    copied_metrics = out_dir / f"fig2_{minutes:02d}min_metrics.csv"
    copied_summary = out_dir / f"fig2_{minutes:02d}min_summary.json"
    copied_pdf = out_dir / f"fig2_{minutes:02d}min.pdf"
    copied_png = out_dir / f"fig2_{minutes:02d}min.png"
    log_path = out_dir / f"fig2_{minutes:02d}min.log"

    if not force and copied_metrics.exists() and copied_pdf.exists() and copied_png.exists():
        print(f"[skip] {minutes} min already exists", flush=True)
        return read_metrics(copied_metrics, minutes, tag)

    env = dict(os.environ)
    env.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONPATH": f"{BUNDLE / 'code' / 'all_python_snapshot'}:{BUNDLE / 'code' / 'core'}",
            "OBSERVING_DAYS": "5",
            "EPS_STATION": "0.02",
            "EPS_PAIR": "0.01",
            "EPS_DIRECT_EXTRA": "0.01",
            "HAWAII3_REMOTE_X_SCALE": "1.0",
            "HAWAII3_REMOTE_Y_SCALE": "0.85",
            "KEEP_EVERY_TIME_ROW": "1",
            "EXPOSURE_SCALE": f"{scale:.6g}",
            "FIG2_RNG_SEED": "20260529",
            "RML_FIT_N_PIX": "40",
            "RML_ADAM_ITER": "2600",
            "RML_ADAM_LR": "0.010",
            "RML_PRIOR_WEIGHT": "0.01",
            "RML_TV_WEIGHT": "0.01",
            "RML_ENTROPY_WEIGHT": "0.005",
            "PAIR_CORE_DIRECT_NOISE": "1",
            "RUN_TAG": tag,
        }
    )

    print(f"[run] {minutes} min/sample -> {tag}", flush=True)
    with log_path.open("w") as log:
        subprocess.run(
            [sys.executable, str(FIG2_SCRIPT)],
            cwd=BUNDLE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    source_paths = {
        copied_pdf: PARENT_OUT / f"{stem}.pdf",
        copied_png: PARENT_OUT / f"{stem}.png",
        copied_metrics: PARENT_OUT / f"{stem}_metrics.csv",
        copied_summary: PARENT_OUT / f"{stem}_summary.json",
    }
    for dst, src in source_paths.items():
        if not src.exists():
            raise FileNotFoundError(f"missing expected output: {src}")
        shutil.copyfile(src, dst)
    return read_metrics(copied_metrics, minutes, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Fig. 2 per-sample integration times.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="rerun even if copied outputs exist")
    parser.add_argument("--minutes", nargs="*", type=int, default=list(range(3, 11)))
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str | int | float]] = []
    for minutes in args.minutes:
        if minutes < 1:
            raise ValueError(f"invalid minutes_per_sample={minutes}")
        all_rows.extend(run_one(minutes, out_dir, force=args.force))

    summary_csv = out_dir / "fig2_integration_time_scan_metrics.csv"
    summary_json = out_dir / "fig2_integration_time_scan_metrics.json"
    write_csv(summary_csv, all_rows)
    summary_json.write_text(json.dumps(all_rows, indent=2) + "\n")
    print(summary_csv)
    print(summary_json)
    print(out_dir)


if __name__ == "__main__":
    main()
