#!/usr/bin/env python3
"""Run all non-expensive reproducibility and completeness checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODULES = (
    "01_principle_and_implementation_schematics_(fig1&2)",
    "02_six_station_imaging_(fig3&4)",
    "03_astrophysical_reach_fig5",
    "04_N4_ns2_illustrative_example",
)

SUPERSEDED_MODULE_NAMES = (
    "01_principle_and_implementation_schematics" + "/",
    "02_six_station_imaging_fig3" + "/",
    "03_astrophysical_reach_fig4" + "/",
)


def main() -> None:
    for module in MODULES:
        print(f"\n=== audit {module} ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / module / "audit.py")],
            cwd=ROOT / module,
            check=True,
        )

    forbidden = ("/Users/" + "xinyaoguo", "Desktop/" + "VLBI")
    bad: list[str] = []
    for suffix in ("*.py", "*.sh"):
        for path in ROOT.rglob(suffix):
            text = path.read_text(errors="replace")
            if any(token in text for token in forbidden):
                bad.append(str(path.relative_to(ROOT)))
    if bad:
        raise RuntimeError(f"non-portable absolute paths remain in source: {bad}")
    stale: list[str] = []
    for suffix in ("*.md", "*.py", "*.sh"):
        for path in ROOT.rglob(suffix):
            text = path.read_text(errors="replace")
            if any(name in text for name in SUPERSEDED_MODULE_NAMES):
                stale.append(str(path.relative_to(ROOT)))
    if stale:
        raise RuntimeError(f"superseded module names remain in active text: {stale}")
    print("\nPASS source portability: no workstation-specific paths in .py/.sh")
    print("PASS directory-name consistency: current Fig. 1--5 module names only")
    print("PASS archive audit")


if __name__ == "__main__":
    main()
