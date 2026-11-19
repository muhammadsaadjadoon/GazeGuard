"""CLI for live monitoring, diagnostics and offline history/report operations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .analytics import aggregate, load_history, save_html
from .config import DEFAULT_HOME, Settings, load_settings, write_default_settings


def diagnostics(data_dir: Path, camera_index: int | None = None, probe: bool = False) -> dict:
    import platform
    import sys
    from importlib import metadata
    details = {"python": sys.version.split()[0], "platform": platform.platform(),
               "data_directory": str(data_dir), "packages": {}, "camera": "not tested"}
    for package in ("numpy", "opencv-contrib-python", "mediapipe"):
        try:
            details["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            details["packages"][package] = "NOT INSTALLED"
    if probe:
        try:
            import cv2
            index = camera_index if camera_index is not None else 0
            cap = cv2.VideoCapture(index)
            try:
                ok, frame = cap.read() if cap.isOpened() else (False, None)
                details["camera"] = f"index={index}, frame={frame.shape if ok else 'unavailable'}"
            finally:
                cap.release()
        except Exception as exc:
            details["camera"] = f"error: {exc}"
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gazeguard", description="GazeGuard 8 | local dark browser dashboard")
    parser.add_argument("command", nargs="?", choices=("run", "report", "history", "doctor", "init"), default="run")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-camera", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    root = args.data_dir.expanduser()
    if args.command == "run":
        from .webapp import main as app_main
        forwarded = ["--data-dir", str(root)]
        if args.camera is not None: forwarded += ["--camera", str(args.camera)]
        return app_main(forwarded)
    if args.command == "doctor":
        print(json.dumps(diagnostics(root, args.camera, args.probe_camera), indent=2))
        return 0
    if args.command == "init":
        write_default_settings(root / "settings.json")
        print("Settings: " + str(root / "settings.json"))
        return 0
    try:
        sessions, warnings = load_history(root / "sessions", args.limit)
        for msg in warnings: print("WARNING:", msg)
        if args.command == "history":
            print(json.dumps({"aggregate": aggregate(sessions),
                              "sessions": [s.__dict__ for s in sessions]}, indent=2))
            return 0
        target = load_settings(root / "settings.json").focus_target_percent
        destination = args.output or (root / "analytics.html")
        save_html(destination, sessions, target)
        print("Report: " + str(destination))
        print("Sessions: " + str(len(sessions)))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print("ERROR:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
