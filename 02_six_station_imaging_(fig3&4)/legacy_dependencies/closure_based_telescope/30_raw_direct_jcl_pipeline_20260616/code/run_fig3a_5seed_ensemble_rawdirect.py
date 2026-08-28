from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_broad_plume_split_objective_rml_sixstation.py"
RESULTS = ROOT / "results" / "fig3a_seed_runs"
LOGS = ROOT / "logs"
MPLCONFIG = LOGS / "mplconfig"

SEEDS = [20260529, 20260530, 20260531, 20260532, 20260533]
EXPOSURES = [("100ms", 0.100), ("50ms", 0.050)]


def run_tag(label: str, seed: int) -> str:
    return f"fig3a_{label}_seed{seed}_paired10loop"


def summary_path(label: str, seed: int) -> Path:
    output_root = RESULTS / label / f"seed_{seed}"
    stem = f"broad_plume_split_objective_nmode_rml_{run_tag(label, seed)}_summary.json"
    return output_root / "rml_outputs" / stem


def run_one(label: str, exposure_s: float, seed: int) -> None:
    output_root = RESULTS / label / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summary_path(label, seed)
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            if len(data.get("rows", [])) == 4:
                print(f"[skip] {label} seed={seed} already complete", flush=True)
                return
        except Exception:
            pass

    env = os.environ.copy()
    env.update(
        {
            "FIG2_OUTPUT_ROOT": str(output_root),
            "FIG2_EXPOSURE_S": f"{exposure_s:.3f}",
            "FIG2_SAMPLE_CADENCE_S": "900.0",
            "FIG2_RNG_SEED": str(seed),
            "FIG2_PAIRED_LOOP_NOISE": "1",
            "FIG2_SKIP_PLOT": "1",
            "RUN_TAG": run_tag(label, seed),
            "MPLCONFIGDIR": str(MPLCONFIG),
            "PYTHONUNBUFFERED": "1",
        }
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    MPLCONFIG.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"fig3a_{label}_seed{seed}.log"
    print(f"[run] {label} seed={seed} exposure={exposure_s:.3f}s", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{label} seed={seed} failed with code {proc.returncode}; see {log_path}")
    if not summary.exists():
        raise FileNotFoundError(f"Expected summary was not produced: {summary}")
    print(f"[done] {label} seed={seed}", flush=True)


def main() -> None:
    for label, exposure_s in EXPOSURES:
        for seed in SEEDS:
            run_one(label, exposure_s, seed)


if __name__ == "__main__":
    main()
