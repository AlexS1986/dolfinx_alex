#!/usr/bin/env python3
"""Build a portable file inventory and optional SHA-256 checksum list."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


SKIP_NAMES = {"MANIFEST.csv", "SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category(relative_path: Path) -> str:
    first = relative_path.parts[0]
    if first == "results":
        return "simulation-result"
    if first == "resources":
        return "simulation-input"
    if first == "plots":
        return "generated-plot"
    if first == "code" or relative_path.suffix in {".py", ".sh"}:
        return "source-code"
    if first.startswith("submission") or first == "68c3b8d0b7dca7b64b8b7a93":
        return "manuscript"
    return "documentation-or-provenance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Archive root. Defaults to the parent of archive_tools.",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="Compute SHA-256 for every file and write SHA256SUMS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in SKIP_NAMES
    )
    manifest_path = root / "MANIFEST.csv"
    checksum_lines = []

    with manifest_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["path", "bytes", "category", "sha256"])
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(root)
            checksum = sha256(path) if args.checksums else ""
            writer.writerow([relative.as_posix(), path.stat().st_size, category(relative), checksum])
            if checksum:
                checksum_lines.append(f"{checksum}  {relative.as_posix()}")
            if index % 1000 == 0:
                print(f"Inventoried {index}/{len(files)} files")

    if args.checksums:
        (root / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

    print(f"Wrote {manifest_path} with {len(files)} files")
    if args.checksums:
        print(f"Wrote {root / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()

