#!/usr/bin/env python3
"""
Build the Month-End Trend dashboard into a standalone Windows bundle.

This script expects to run on Windows with Python, PyInstaller, and the project's
dependencies installed in the current interpreter. It produces a ZIP archive
containing an `exe` alongside the app's workspace so it can run without admin
rights.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_NAME = 'MonthEndTrend'
DIST_DIR = ROOT / 'dist' / DIST_NAME
BUILD_DIR = ROOT / 'build'
SPEC_FILE = ROOT / 'run_app.spec'
OUTPUT_ZIP = ROOT / f'{DIST_NAME}.zip'


def run_pyinstaller() -> None:
    """Invoke PyInstaller using the provided spec file."""
    if not SPEC_FILE.exists():
        raise FileNotFoundError(f'Spec file not found: {SPEC_FILE}')

    cmd = [sys.executable, '-m', 'PyInstaller', str(SPEC_FILE.name)]
    subprocess.check_call(cmd, cwd=ROOT)


def create_zip_bundle() -> None:
    """Package the dist directory into a ZIP archive."""
    if not DIST_DIR.exists():
        raise FileNotFoundError(f'Expected build output missing: {DIST_DIR}')

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for item in DIST_DIR.rglob('*'):
            archive_name = f'{DIST_NAME}/{item.relative_to(DIST_DIR)}'
            zf.write(item, archive_name)


def clean_previous_builds() -> None:
    """Remove previous build artifacts to ensure a clean bundle."""
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()


def main() -> None:
    clean_previous_builds()
    run_pyinstaller()
    create_zip_bundle()
    print(f'Created bundle at {OUTPUT_ZIP}')


if __name__ == '__main__':
    main()
