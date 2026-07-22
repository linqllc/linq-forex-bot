"""
Native market-memory intelligence.

This module preserves the validated legacy implementation while exposing it
through the active LINQ platform package. It is an intermediate migration step:
behavior stays frozen while tests and callers move to the native namespace.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


LEGACY_SOURCE = Path(__file__).resolve().parents[2] / "run_market_memory_v1.py"


def legacy_source_path() -> Path:
    """Return the validated legacy market-memory source path."""
    return LEGACY_SOURCE


def load_legacy_namespace() -> dict[str, Any]:
    """
    Load the legacy market-memory module without executing it as __main__.

    The returned namespace exposes its functions and classes for parity tests
    and incremental native refactoring.
    """
    if not LEGACY_SOURCE.exists():
        raise FileNotFoundError(f"Legacy market-memory source not found: {LEGACY_SOURCE}")

    return runpy.run_path(
        str(LEGACY_SOURCE),
        run_name="linq_legacy_market_memory",
    )


def available_symbols() -> tuple[str, ...]:
    """Return public functions and classes exposed by the legacy engine."""
    namespace = load_legacy_namespace()

    return tuple(
        sorted(
            name
            for name, value in namespace.items()
            if not name.startswith("_") and callable(value)
        )
    )
