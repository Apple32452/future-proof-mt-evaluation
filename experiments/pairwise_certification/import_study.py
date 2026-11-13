#!/usr/bin/env python3
"""Import the audited, locally archived MT pairwise certification study.

Usage (from the Git repository root):
    python experiments/pairwise_certification/import_study.py ~/Downloads/MT_Pairwise_Certification_Study_2026-11-13.zip

This script requires no network access or dependencies beyond Python 3.10.
It imports text code, compact protocols and reproducibility summaries.
Large regenerable trial logs and raw human-label data are deliberately omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_ARCHIVE_SHA256 = (
    "114d4c9a96349791187258022583703a3fd96e56cc706b8af27d9675a6f92afd"
)
ROOT_PREFIX = "mt_pairwise_study/"
SKIP_BASENAMES = {"PACKAGE_MANIFEST.json", "make_package.py"}
SKIP_PATHS = {
    "diagnostics/near_tie_v1/traces.jsonl",
    "diagnostics/near_tie_v1/diagnostic_results.json",
    "validation_v1/trials.jsonl",
    "validation_v2/comparison_trials.jsonl",
    "validation_v2/new_trials.jsonl",
}
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".txt"}
TOP_LEVEL_SKIP = {"README.md"}  # GitHub's landing README documents evidence status.

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True, check=True, capture_output=True,
    )
    return Path(result.stdout.strip()).resolve()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Validated study ZIP")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    actual = sha256(archive)
    if actual != EXPECTED_ARCHIVE_SHA256:
        parser.error(
            f"ZIP SHA-256 mismatch: {actual}; expected {EXPECTED_ARCHIVE_SHA256}"
        )
    root = git_root()
    destroot = root / "experiments" / "pairwise_certification"
    planned = []
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            parser.error(f"Invalid ZIP member: {bad}")
        for info in zf.infolist():
            if info.is_dir() or not info.filename.startswith(ROOT_PREFIX):
                continue
            relative_str = info.filename[len(ROOT_PREFIX):]
            relative = PurePosixPath(relative_str)
            if (not relative.parts or
                    any(part in ("", ".", "..") for part in relative.parts) or
                    relative.is_absolute()):
                parser.error(f"Unsafe archive member: {info.filename}")
            if (len(relative.parts) == 1 and relative.name in TOP_LEVEL_SKIP or
                    relative.name in SKIP_BASENAMES or
                    relative_str in SKIP_PATHS):
                continue
            if relative.suffix not in TEXT_SUFFIXES:
                parser.error(f"Unexpected archive format: {relative_str}")
            if info.file_size > 500_000:
                parser.error(f"Oversized text member: {relative_str}")
            path = destroot.joinpath(*relative.parts)
            payload = zf.read(info)
            payload.decode("utf-8")  # Reject binaries.
            if path.exists() and path.read_bytes() != payload:
                parser.error(f"Refusing overwrite of changed file: {path}")
            planned.append((path, payload))
    if not planned:
        parser.error("No files matched; incorrect archive format")
    for path, payload in planned:
        print(("PLAN" if args.dry_run else "IMPORT"), path.relative_to(root))
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print(f"{len(planned)} validated files prepared; raw dataset/traces omitted.")
    print("Next: inspect git diff, run tests, then commit on the experiment branch.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
