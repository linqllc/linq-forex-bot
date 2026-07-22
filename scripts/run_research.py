#!/usr/bin/env python3
"""Stable repository entry point for the native research runner."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    ROOT / "run_native_research.py",
    ROOT / "scripts" / "native_research_impl.py",
]

for candidate in CANDIDATES:
    if candidate.exists():
        runpy.run_path(str(candidate), run_name="__main__")
        break
else:
    raise SystemExit(
        "Native research implementation was not found. "
        "Copy the existing run_native_research.py into the repository root."
    )
