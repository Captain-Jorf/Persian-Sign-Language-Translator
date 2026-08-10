from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a one-file desktop CLI executable.")
    parser.add_argument("--name", default="PersianSignLanguageTranslator")
    parser.add_argument("--release-dir", default="release")
    args = parser.parse_args()

    pyinstaller = shutil.which("pyinstaller")
    command = [pyinstaller] if pyinstaller else [sys.executable, "-m", "PyInstaller"]

    extra_args: list[str] = []
    if importlib.util.find_spec("mediapipe") is not None:
        extra_args += ["--collect-data", "mediapipe"]
    if importlib.util.find_spec("cv2") is not None:
        extra_args += ["--collect-submodules", "cv2"]

    subprocess.check_call(
        command
        + [
            "--clean",
            "--onefile",
            "--name",
            args.name,
        ]
        + extra_args
        + ["psl_translator/__main__.py"],
        cwd=ROOT,
    )

    suffix = ".exe" if platform.system().lower() == "windows" else ""
    built = ROOT / "dist" / f"{args.name}{suffix}"
    if not built.exists():
        # On Linux/macOS PyInstaller ignores .exe conventions. Keep this explicit
        # because fake Windows binaries are worse than no binary.
        built = ROOT / "dist" / args.name
    release_dir = ROOT / args.release_dir
    release_dir.mkdir(exist_ok=True)
    target_name = f"{args.name}.exe" if platform.system().lower() == "windows" else f"{args.name}-{platform.system().lower()}"
    target = release_dir / target_name
    shutil.copy2(built, target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
