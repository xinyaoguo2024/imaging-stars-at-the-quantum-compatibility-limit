#!/usr/bin/env python3
"""Build the archive file manifest and transport checksums."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"MANIFEST.tsv", "SHA256SUMS.txt"}
EXCLUDED_PARTS = {"__pycache__", "mplconfig"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED:
            continue
        relative = path.relative_to(ROOT)
        if path.name == ".DS_Store" or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        rows.append((str(relative), path.stat().st_size, digest(path)))
    manifest = ["relative_path\tbytes\tsha256"]
    manifest.extend(f"{path}\t{size}\t{sha}" for path, size, sha in rows)
    (ROOT / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha}  {path}" for path, _size, sha in rows) + "\n"
    )
    print(f"indexed {len(rows)} files")


if __name__ == "__main__":
    main()
