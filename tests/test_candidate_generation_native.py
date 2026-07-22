from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from linq_platform.intelligence.candidate_generation import (
    build_zones,
    calculate_pivots,
    candle_touches_zone,
    cluster_direction_reactions,
    detect_reactions,
    find_first_zone_touch,
    first_threshold_hit,
)
from linq_platform.intelligence.market_memory import (
    Config,
    add_features,
)


LEGACY_PATH = Path(__file__).resolve().parents[1] / "run_market_memory_v1.py"


def legacy_namespace() -> dict:
    return runpy.run_path(
        str(LEGACY_PATH),
        run_name="linq_candidate_generation_reference",
    )


def synthetic_candles() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 420

    time = pd.date_range(
        "2025-01-01",
        periods=size,
        freq="5min",
        tz="UTC",
    )

    movement = rng.normal(
        loc=0.0,
        scale=0.00018,
        size=size,
    )

    close = 1.1000 + np.cumsum(movement)
    open_price = np.concatenate([[close[0]], close[:-1]])

    wick = rng.uniform(
        0.00005,
        0.00022,
        size=size,
    )

    high = np.maximum(open_price, close) + wick
    low = np.minimum(open_price, close) - wick

    # Add several objective swing reactions so the
    # detector and zone builder receive meaningful input.
    for index in [80, 150, 225, 310]:
        low[index] -= 0.0015

        for offset in range(1, 12):
            close[index + offset] += offset * 0.00018
            high[index + offset] += offset * 0.00018

    for index in [115, 190, 270, 350]:
        high[index] += 0.0015

        for offset in range(1, 12):
            close[index + offset] -= offset * 0.00018
            low[index + offset] -= offset * 0.00018

    frame = pd.DataFrame(
        {
            "time": time,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100,
        }
    )

    return add_features(frame, Config())


def test_pivot_parity() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()

    native_low, native_high = calculate_pivots(
        candles,
        left=3,
        right=3,
    )

    legacy_low, legacy_high = legacy["calculate_pivots"](
        candles,
        left=3,
        right=3,
    )

    np.testing.assert_array_equal(
        native_low,
        legacy_low,
    )
    np.testing.assert_array_equal(
        native_high,
        legacy_high,
    )


def test_threshold_hit_parity() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()

    native = first_threshold_hit(
        candles,
        start_index=20,
        end_index=100,
        upper_price=float(candles.at[20, "close"]) + 0.0005,
        lower_price=float(candles.at[20, "close"]) - 0.0005,
    )

    reference = legacy["first_threshold_hit"](
        candles,
        start_index=20,
        end_index=100,
        upper_price=float(candles.at[20, "close"]) + 0.0005,
        lower_price=float(candles.at[20, "close"]) - 0.0005,
    )

    assert native == reference


def test_reaction_detection_parity() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()
    config = Config()

    native = detect_reactions(
        candles,
        config,
    )

    reference = legacy["detect_reactions"](
        candles,
        config,
    )

    assert_frame_equal(
        native.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_clustering_parity() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()
    config = Config()

    reactions = detect_reactions(
        candles,
        config,
    )

    for direction in ("demand", "supply"):
        native_clusters = cluster_direction_reactions(
            reactions,
            direction,
            config,
        )

        legacy_clusters = legacy["cluster_direction_reactions"](
            reactions,
            direction,
            config,
        )

        assert len(native_clusters) == len(legacy_clusters)

        for native, reference in zip(
            native_clusters,
            legacy_clusters,
            strict=True,
        ):
            assert_frame_equal(
                native.reset_index(drop=True),
                reference.reset_index(drop=True),
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )


def test_zone_building_parity() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()
    config = Config()

    reactions = detect_reactions(
        candles,
        config,
    )

    as_of_time = candles["time"].iloc[-1]

    native = build_zones(
        reactions,
        as_of_time,
        config,
    )

    reference = legacy["build_zones"](
        reactions,
        as_of_time,
        config,
    )

    assert_frame_equal(
        native.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_zone_touch_helpers_match_legacy() -> None:
    legacy = legacy_namespace()
    candles = synthetic_candles()

    native_touch = candle_touches_zone(
        candle_low=1.0990,
        candle_high=1.1010,
        zone_low=1.1000,
        zone_high=1.1020,
    )

    reference_touch = legacy["candle_touches_zone"](
        candle_low=1.0990,
        candle_high=1.1010,
        zone_low=1.1000,
        zone_high=1.1020,
    )

    assert native_touch == reference_touch

    zone_low = float(candles["low"].iloc[50:80].quantile(0.40))
    zone_high = float(candles["high"].iloc[50:80].quantile(0.60))

    native_index = find_first_zone_touch(
        candles,
        start_index=50,
        end_index=120,
        zone_low=zone_low,
        zone_high=zone_high,
    )

    reference_index = legacy["find_first_zone_touch"](
        candles,
        start_index=50,
        end_index=120,
        zone_low=zone_low,
        zone_high=zone_high,
    )

    assert native_index == reference_index
