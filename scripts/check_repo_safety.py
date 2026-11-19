"""Defensive repository scanner; never uploads or modifies local files.

Usage: python scripts/check_repo_safety.py [--staged]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE_LIMIT = 90 * 1024 * 1024
DENIED_DIRS = {".venv", "venv", "__pycache__", "node_modules", "private", "public_external"}
DENIED_EXTS = {".zip", ".7z", ".rar", ".tar", ".gz", ".h5", ".hdf5", ".mat", ".npy", ".npz", ".onnx", ".pt", ".pth", ".mp4", ".avi", ".mov", ".mkv", ".task", ".pem", ".key"}
APPROVED_DEMOS = {Path("assets/demo/GazeGuard_Hand_Tracking_Full.mp4"), Path("assets/demo/GazeGuard_Gaze_Detection_1_5x.mp4")}
DEMO_MAX_BYTES = 30 * 1024 * 1024
DENIED_NAMES = {"kaggle.json", "credentials.json", ".env", "trained_gaze.json"}


def suspicious(path: Path) -> bool:
    parts = path.parts
    if path in APPROVED_DEMOS:
        return False  # Explicitly approved public demo; checked separately for size.
    if parts in (("datasets", "private", "README.txt"), ("datasets", "public_external", "README.txt")):
        return False
    if any(p in DENIED_DIRS for p in parts):
        return True
    if path.suffix.lower() in DENIED_EXTS:
        return True
    if path.name.lower() in DENIED_NAMES or path.name.lower().startswith(".env."):
        return True
    if parts and parts[0] == "models" and path.name != "README.txt":
        return True
    if parts and parts[0] == "datasets" and path.suffix.lower() == ".csv" and path.name != "CSV_SCHEMA.csv":
        return True
    return False


def tracked_paths() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "-z"], stderr=subprocess.DEVNULL,
        )
        return [Path(p.decode("utf-8", errors="replace")) for p in out.split(b"\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return []


def staged_paths() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z"], stderr=subprocess.DEVNULL,
        )
        return [Path(p.decode("utf-8", errors="replace")) for p in out.split(b"\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return []


def source_paths() -> list[Path]:
    """Approximate what 'git add .' would stage, honoring .gitignore."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
        return [Path(p.decode("utf-8", errors="replace")) for p in out.split(b"\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true", help="Inspect staged paths only")
    args = parser.parse_args()
    paths = staged_paths() if args.staged else source_paths()
    found = []
    for path in paths:
        file = ROOT / path
        if suspicious(path):
            found.append(f"Sensitive/binary path: {path}")
        if path in APPROVED_DEMOS and file.is_file() and file.stat().st_size > DEMO_MAX_BYTES:
            found.append(f"Demo must be under {DEMO_MAX_BYTES // 1024**2} MiB: {path}")
        if file.is_file() and file.stat().st_size >= SIZE_LIMIT:
            found.append(f"Large file ({file.stat().st_size / 1024**2:.1f} MiB): {path}")
    for item in found:
        print("BLOCK:", item)
    if found:
        print("Review and remove these files from the Git index before publishing.")
        return 1
    print(f"PASS: inspected {len(paths)} {'staged' if args.staged else 'candidate'} files; no blocked paths detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
