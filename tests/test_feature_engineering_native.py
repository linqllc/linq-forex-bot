from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from linq_platform.intelligence.feature_engineering import (
    LEAKAGE,
    build_dataset,
    clean_features,
    evaluate,
    feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]

CORE_CANDIDATES = [
    ROOT / "linq_engine" / "core.py",
    ROOT / "legacy" / "phase4_reference" / "linq_engine" / "core.py",
    ROOT / "linq_v9_phase1" / "linq_engine" / "core.py",
]


def reference_core() -> ModuleType:
    core_path = next(
        (path for path in CORE_CANDIDATES if path.exists()),
        None,
    )

    if core_path is None:
        raise FileNotFoundError("Could not locate the Phase 4 reference core.py")

    package_parent = core_path.parent.parent

    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    sys.modules.pop("linq_engine.core", None)
    sys.modules.pop("linq_engine.config", None)
    sys.modules.pop("linq_engine", None)

    return importlib.import_module("linq_engine.core")


def phase4_config():
    core = reference_core()
    return core.Phase4Config(
        probability_threshold=0.65,
        stop_atr=2.0,
        minimum_stop_pips=8.0,
        target_r=1.0,
        forward_bars=72,
        holdout_trades=20,
        minimum_training_rows=10,
        slippage_pips_each_side=0.10,
        maximum_spread_pips=2.5,
        maximum_spread_to_stop_ratio=0.20,
        pip_size=0.0001,
    )


def synthetic_candles() -> pd.DataFrame:
    rng = np.random.default_rng(1701)
    size = 520

    timestamp = pd.date_range(
        "2024-01-01",
        periods=size,
        freq="5min",
        tz="UTC",
    )

    movement = rng.normal(
        loc=0.0,
        scale=0.00012,
        size=size,
    )

    close = 1.0850 + np.cumsum(movement)
    open_price = np.concatenate(([close[0]], close[:-1]))

    wick_up = rng.uniform(
        0.00004,
        0.00018,
        size=size,
    )

    wick_down = rng.uniform(
        0.00004,
        0.00018,
        size=size,
    )

    high = np.maximum(open_price, close) + wick_up
    low = np.minimum(open_price, close) - wick_down

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )

    previous_close = frame["close"].shift()

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    frame["atr"] = true_range.rolling(
        14,
        min_periods=14,
    ).mean()

    frame["ema20"] = (
        frame["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    frame["ema50"] = (
        frame["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    frame["ema200"] = (
        frame["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    difference = frame["close"].diff()

    gain = (
        difference.clip(lower=0)
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    loss = (
        (-difference.clip(upper=0))
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    frame["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    frame["ret3"] = frame["close"].pct_change(3)
    frame["ret12"] = frame["close"].pct_change(12)

    frame["range_atr"] = (frame["high"] - frame["low"]) / frame["atr"]

    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.dayofweek

    frame["spread_pips"] = rng.uniform(
        0.5,
        1.4,
        size=size,
    )

    return frame


def synthetic_inputs():
    candles = synthetic_candles()

    positions = [
        210,
        245,
        280,
        315,
        350,
        385,
        420,
    ]

    setups = pd.DataFrame(
        {
            "setup_id": [f"setup_{index:03d}" for index in range(len(positions))],
            "timestamp": [candles.at[position, "timestamp"] for position in positions],
            "direction": [
                "long",
                "short",
                "long",
                "short",
                "long",
                "short",
                "long",
            ],
            "entry": [candles.at[position, "close"] for position in positions],
        }
    )

    phase1 = pd.DataFrame(
        {
            "setup_id": setups["setup_id"],
            "zone_width_atr": [
                0.50,
                0.70,
                0.45,
                0.80,
                0.60,
                0.55,
                0.65,
            ],
            "reaction_count": [
                2,
                3,
                1,
                4,
                2,
                3,
                2,
            ],
            "zone_type": [
                "demand",
                "supply",
                "demand",
                "supply",
                "demand",
                "supply",
                "demand",
            ],
            "probability_old": [
                0.99,
                0.98,
                0.97,
                0.96,
                0.95,
                0.94,
                0.93,
            ],
            "result_r": [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
            ],
            "hit_target": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "bars_to_target": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    return candles, setups, phase1


def test_leakage_definition_matches_reference() -> None:
    reference = reference_core()

    assert LEAKAGE == reference.LEAKAGE


def test_clean_features_removes_leakage() -> None:
    reference = reference_core()

    row = {
        "setup_id": "setup_001",
        "zone_width_atr": 0.55,
        "reaction_count": 3,
        "result_r": 1.0,
        "gross_result_r": 1.0,
        "probability_1r": 0.82,
        "probability_old": 0.75,
        "hit_target": 1,
        "bars_to_target": 7,
        "market_regime": "trend",
    }

    native = clean_features(row)
    expected = reference.clean_features(row)

    assert native == expected
    assert "zone_width_atr" in native
    assert "reaction_count" in native
    assert "market_regime" in native
    assert "result_r" not in native
    assert "gross_result_r" not in native
    assert "probability_1r" not in native
    assert "probability_old" not in native
    assert "hit_target" not in native
    assert "bars_to_target" not in native


def test_trade_evaluation_matches_reference() -> None:
    reference = reference_core()
    config = phase4_config()
    candles = synthetic_candles()

    future = candles.iloc[260:340].copy()

    cases = [
        (
            "long",
            float(candles.at[259, "close"]),
            float(candles.at[259, "close"] - 0.0010),
        ),
        (
            "short",
            float(candles.at[259, "close"]),
            float(candles.at[259, "close"] + 0.0010),
        ),
    ]

    for direction, entry, stop in cases:
        native = evaluate(
            direction,
            entry,
            stop,
            future,
            0.9,
            config,
        )

        expected = reference.evaluate(
            direction,
            entry,
            stop,
            future,
            0.9,
            config,
        )

        assert native is not None
        assert expected is not None
        assert native.keys() == expected.keys()

        for key in native:
            if isinstance(native[key], float):
                assert np.isclose(
                    native[key],
                    expected[key],
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            else:
                assert native[key] == expected[key]


def test_dataset_construction_matches_reference() -> None:
    reference = reference_core()
    config = phase4_config()

    candles, setups, phase1 = synthetic_inputs()

    native = build_dataset(
        candles,
        setups,
        phase1,
        config,
    )

    expected = reference.build_dataset(
        candles,
        setups,
        phase1,
        config,
    )

    assert_frame_equal(
        native,
        expected,
        check_dtype=True,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    assert len(native) > 0
    assert native["setup_id"].is_unique
    assert native["timestamp"].is_monotonic_increasing

    forbidden = {
        "probability_old",
        "result_r",
        "hit_target",
        "bars_to_target",
    }

    assert forbidden.isdisjoint(native.columns)


def test_feature_column_selection_matches_reference() -> None:
    reference = reference_core()
    config = phase4_config()

    candles, setups, phase1 = synthetic_inputs()

    dataset = build_dataset(
        candles,
        setups,
        phase1,
        config,
    )

    native = feature_columns(dataset)
    expected = reference.feature_columns(dataset)

    assert native == expected

    forbidden = LEAKAGE | {
        "setup_id",
        "strategy",
    }

    assert forbidden.isdisjoint(native)
    assert "direction" in native
    assert "zone_width_atr" in native
    assert "market_rsi" in native
