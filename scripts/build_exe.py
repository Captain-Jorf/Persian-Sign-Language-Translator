from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the desktop executable.")
    parser.add_argument("--name", default="PersianSignLanguageTranslator")
    parser.add_argument("--release-dir", default="release")
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="gui is for double-click users; cli is for terminal automation.",
    )
    args = parser.parse_args()

    command = [sys.executable, "-m", "PyInstaller"]

    sep = ";" if platform.system().lower() == "windows" else ":"

    extra_args: list[str] = []
    if args.mode == "gui":
        extra_args.append("--windowed")

    for folder in ["data/demo", "models", "reports/demo"]:
        source = ROOT / folder
        if source.exists():
            extra_args += ["--add-data", f"{source}{sep}{folder}"]

    entrypoint = "psl_translator/gui_entry.py" if args.mode == "gui" else "psl_translator/__main__.py"
    exe_name = args.name if args.mode == "gui" else f"{args.name}CLI"

    subprocess.check_call(
        command
        + [
            "--clean",
            "--noconfirm",
            "--onefile",
            "--name",
            exe_name,
        ]
        + extra_args
        + [entrypoint],
        cwd=ROOT,
    )

    suffix = ".exe" if platform.system().lower() == "windows" else ""
    built = ROOT / "dist" / f"{exe_name}{suffix}"

    release_dir = ROOT / args.release_dir
    release_dir.mkdir(exist_ok=True)

    target = release_dir / f"{exe_name}{suffix}"
    shutil.copy2(built, target)

    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
