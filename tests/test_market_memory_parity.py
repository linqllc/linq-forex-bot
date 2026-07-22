from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from linq_platform.intelligence import market_memory as native


@pytest.fixture(scope="module")
def legacy() -> dict[str, object]:
    return native.load_legacy_namespace()


@pytest.fixture
def candles() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC")
    base = np.linspace(1.1000, 1.1040, len(times))
    return pd.DataFrame(
        {
            "time": times,
            "open": base,
            "high": base + 0.0008,
            "low": base - 0.0007,
            "close": base + np.sin(np.arange(len(times))) * 0.0002,
            "volume": np.arange(len(times)) + 100,
        }
    )


def test_normalize_name_parity(legacy: dict[str, object]) -> None:
    legacy_fn = legacy["normalize_name"]
    samples = ["Mid Open", "bid-close", " TIME ", "tick.volume"]
    assert [native.normalize_name(value) for value in samples] == [
        legacy_fn(value)
        for value in samples  # type: ignore[operator]
    ]


def test_load_candles_parity(
    tmp_path: Path, legacy: dict[str, object], candles: pd.DataFrame
) -> None:
    path = tmp_path / "candles.csv"
    raw = candles.rename(
        columns={
            "time": "Timestamp",
            "open": "Mid Open",
            "high": "Mid-High",
            "low": "bid_low",
            "close": "C",
            "volume": "Tick Volume",
        }
    )
    raw.to_csv(path, index=False)
    expected = legacy["load_candles"](path)  # type: ignore[operator]
    actual = native.load_candles(path)
    pdt.assert_frame_equal(actual, expected)


def test_atr_and_feature_parity(legacy: dict[str, object], candles: pd.DataFrame) -> None:
    expected_atr = legacy["calculate_atr"](candles, 14)  # type: ignore[operator]
    actual_atr = native.calculate_atr(candles, 14)
    pdt.assert_series_equal(actual_atr, expected_atr)

    config = native.Config()
    expected_features = legacy["add_features"](candles, legacy["Config"]())  # type: ignore[operator]
    actual_features = native.add_features(candles, config)
    pdt.assert_frame_equal(actual_features, expected_features)


def test_pivot_parity(legacy: dict[str, object], candles: pd.DataFrame) -> None:
    expected_low, expected_high = legacy["calculate_pivots"](candles, 3, 3)  # type: ignore[operator]
    actual_low, actual_high = native.calculate_pivots(candles, 3, 3)
    np.testing.assert_array_equal(actual_low, expected_low)
    np.testing.assert_array_equal(actual_high, expected_high)


def test_threshold_parity(legacy: dict[str, object], candles: pd.DataFrame) -> None:
    args = (candles, 0, len(candles) - 1, 1.1025, 1.0985)
    assert native.first_threshold_hit(*args) == legacy["first_threshold_hit"](*args)  # type: ignore[operator]


def test_index_helper_parity(legacy: dict[str, object], candles: pd.DataFrame) -> None:
    target = candles.at[10, "time"] + pd.Timedelta(minutes=2)
    assert native.nearest_index_at_or_after(candles, target) == legacy["nearest_index_at_or_after"](
        candles, target
    )  # type: ignore[operator]
    assert native.nearest_index_before(candles, target) == legacy["nearest_index_before"](
        candles, target
    )  # type: ignore[operator]


def test_summary_helpers_parity(legacy: dict[str, object]) -> None:
    values = pd.Series([1.0, -0.5, -1.0, 2.0, -0.25])
    assert native.maximum_drawdown_r(values) == legacy["maximum_drawdown_r"](values)  # type: ignore[operator]

    zone = pd.Series({"zone_low": 1.1, "zone_high": 1.2, "direction": "demand"})
    assert native.classify_zone_status(zone, 1.25) == legacy["classify_zone_status"](zone, 1.25)  # type: ignore[operator]
    assert native.format_price(1.234567) == legacy["format_price"](1.234567)  # type: ignore[operator]
