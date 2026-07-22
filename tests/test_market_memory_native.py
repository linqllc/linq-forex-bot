from pathlib import Path

from linq_platform.intelligence.market_memory import (
    available_symbols,
    legacy_source_path,
    load_legacy_namespace,
)


def test_legacy_market_memory_source_exists() -> None:
    path = legacy_source_path()

    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == "run_market_memory_v1.py"


def test_legacy_namespace_loads() -> None:
    namespace = load_legacy_namespace()

    assert isinstance(namespace, dict)
    assert namespace


def test_market_memory_exposes_callable_symbols() -> None:
    symbols = available_symbols()

    assert isinstance(symbols, tuple)
    assert len(symbols) > 0
