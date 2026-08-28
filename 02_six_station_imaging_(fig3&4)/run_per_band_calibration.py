#!/usr/bin/env python3
"""Run the ten finite-n_s=3 wavelength calibrations with bounded concurrency."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "optimize_finite_ns_povm_per_band.py"
LOG_DIR = HERE / "logs" / "finite_ns_per_band"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = int(os.environ.get("FINITE_NS_MAX_WORKERS", "3"))
RESTARTS = int(os.environ.get("FINITE_NS_RESTARTS", "4"))
STEPS = int(os.environ.get("FINITE_NS_STEPS", "1600"))


def run_band(band_index: int) -> tuple[int, int, Path]:
    log_path = LOG_DIR / f"band_{band_index:02d}.log"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(HERE / "mplconfig")
    command = [
        sys.executable,
        str(SCRIPT),
        "--copies",
        "3",
        "--band-index",
        str(band_index),
        "--restarts",
        str(RESTARTS),
        "--steps",
        str(STEPS),
        "--seed",
        str(20260818 + 100 * band_index),
    ]
    with log_path.open("w") as log:
        process = subprocess.run(
            command,
            cwd=str(HERE.parent),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return band_index, process.returncode, log_path


def main() -> None:
    print(
        f"[finite-ns] bands=10 workers={MAX_WORKERS} "
        f"restarts={RESTARTS} steps={STEPS}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_band, band): band for band in range(10)}
        for future in as_completed(futures):
            band, returncode, log_path = future.result()
            print(
                f"[finite-ns] band={band:02d} returncode={returncode} "
                f"log={log_path}",
                flush=True,
            )
            if returncode != 0:
                raise RuntimeError(f"band {band} failed; see {log_path}")


if __name__ == "__main__":
    main()
