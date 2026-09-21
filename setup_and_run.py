"""One-command setup and runner for a fresh checkout of this project."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def create_environment() -> Path:
    python = venv_python()
    if not python.exists():
        uv = shutil.which("uv")
        if uv:
            subprocess.run(
                [uv, "venv", "--python", "3.11", str(VENV)],
                cwd=ROOT,
                check=True,
            )
        elif sys.version_info >= (3, 11):
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV)],
                cwd=ROOT,
                check=True,
            )
        else:
            raise SystemExit(
                "Python 3.11+ is required. Install it from https://python.org "
                "or install uv from https://docs.astral.sh/uv/ and rerun this command."
            )

    marker = VENV / ".ner-landslide-installed"
    project_file = ROOT / "pyproject.toml"
    if not marker.exists() or marker.stat().st_mtime < project_file.stat().st_mtime:
        uv = shutil.which("uv")
        if uv:
            install = [uv, "pip", "install", "--python", str(python), "-e", "."]
        else:
            install = [str(python), "-m", "pip", "install", "-e", "."]
        subprocess.run(install, cwd=ROOT, check=True)
        marker.touch()
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Prepare predictions and launch the real-time GIS dashboard.",
    )
    args, pipeline_args = parser.parse_known_args()
    python = create_environment()

    runner = [str(python), str(ROOT / "run_python.py"), *pipeline_args]
    subprocess.run(runner, cwd=ROOT, check=True)
    if args.dashboard:
        subprocess.run(
            [
                str(python),
                "-m",
                "streamlit",
                "run",
                str(ROOT / "dashboard.py"),
                "--server.headless=false",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
