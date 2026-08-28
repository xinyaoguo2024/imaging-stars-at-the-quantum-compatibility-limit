from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


BUNDLE = Path(__file__).resolve().parents[2]
ROOT = BUNDLE.parents[0]
CORE_DIR = BUNDLE / "code" / "core"
SNAPSHOT_DIR = BUNDLE / "code" / "all_python_snapshot"
OUT = BUNDLE / "exploration" / "self_consistency_audit_20260608"
FIG_DIAG_DIR = BUNDLE / "figures" / "diagnostics"

for path in (CORE_DIR, SNAPSHOT_DIR):
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)

OUT.mkdir(parents=True, exist_ok=True)
FIG_DIAG_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def newest_matching(pattern: str) -> Path:
    matches = sorted(BUNDLE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]
