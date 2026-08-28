from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "run_broad_plume_split_objective_rml_sixstation.py"
RESULTS = ROOT / "results" / "fig3a_seed_runs"
LOGS = ROOT / "logs"
MPLCONFIG = LOGS / "mplconfig"

EXPOSURE_LABEL = "100ms"
EXPOSURE_S = 0.100
DEFAULT_SEEDS = ",".join(str(seed) for seed in range(20260529, 20260544))
SEEDS = [
    int(item.strip())
    for item in os.environ.get("FIG3_EXTRA_SEEDS", DEFAULT_SEEDS).split(",")
    if item.strip()
]
MAX_WORKERS = int(os.environ.get("FIG3_MAX_WORKERS", "2"))


def run_tag(seed: int) -> str:
    return f"fig3a_{EXPOSURE_LABEL}_seed{seed}_paired10loop"


def summary_path(seed: int) -> Path:
    stem = f"broad_plume_split_objective_nmode_rml_{run_tag(seed)}_summary.json"
    return RESULTS / EXPOSURE_LABEL / f"seed_{seed}" / "rml_outputs" / stem


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return False
    rows = data.get("rows", [])
    return len(rows) == 4 and any(row.get("strategy") == "core4_remote_optimized" for row in rows)


def run_one(seed: int) -> tuple[int, str]:
    summary = summary_path(seed)
    if complete(summary):
        return seed, "skip"

    output_root = RESULTS / EXPOSURE_LABEL / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    MPLCONFIG.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"fig3a_{EXPOSURE_LABEL}_seed{seed}_full_paircombine.log"

    env = os.environ.copy()
    env.update(
        {
            "FIG2_OUTPUT_ROOT": str(output_root),
            "FIG2_EXPOSURE_S": f"{EXPOSURE_S:.3f}",
            "FIG2_SAMPLE_CADENCE_S": "900.0",
            "FIG2_RNG_SEED": str(seed),
            "FIG2_PAIRED_LOOP_NOISE": "1",
            "FIG2_SKIP_PLOT": "1",
            "RUN_TAG": run_tag(seed),
            "MPLCONFIGDIR": str(MPLCONFIG),
            "PYTHONUNBUFFERED": "1",
        }
    )
    print(f"[run] {EXPOSURE_LABEL} seed={seed}", flush=True)
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
        return seed, f"failed code={proc.returncode} log={log_path}"
    if not complete(summary):
        return seed, f"incomplete summary={summary}"
    return seed, "done"


def main() -> None:
    print(f"[config] seeds={SEEDS} max_workers={MAX_WORKERS}", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                done_seed, status = future.result()
            except Exception as exc:
                print(f"[error] seed={seed} {exc}", flush=True)
                raise
            print(f"[{status}] seed={done_seed}", flush=True)
            if status.startswith("failed") or status.startswith("incomplete"):
                raise RuntimeError(f"seed={done_seed} {status}")


if __name__ == "__main__":
    main()
