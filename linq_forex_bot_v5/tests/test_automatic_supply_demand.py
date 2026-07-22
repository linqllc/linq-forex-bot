from __future__ import annotations

import pandas as pd

from src.automatic_supply_demand import (
    add_detection_features,
    detect_automatic_zones,
    generate_automatic_setups,
)


def sample_candles() -> pd.DataFrame:
    rows = []
    price = 1.1000

    for index in range(220):
        open_price = price

        if 80 <= index <= 83:
            close_price = open_price + 0.0012
        elif 120 <= index <= 128:
            close_price = open_price - 0.00015
        elif index == 129:
            close_price = open_price + 0.0005
        else:
            close_price = open_price + 0.00002

        high = max(open_price, close_price) + 0.00005
        low = min(open_price, close_price) - 0.00005

        rows.append(
            {
                "time": pd.Timestamp(
                    "2026-01-01",
                    tz="UTC",
                )
                + pd.Timedelta(minutes=index * 5),
                "mid_open": open_price,
                "mid_high": high,
                "mid_low": low,
                "mid_close": close_price,
            }
        )

        price = close_price

    return pd.DataFrame(rows)


def test_detection_features_exist():
    features = add_detection_features(sample_candles())

    expected = {
        "atr",
        "average_body",
        "bullish_bos",
        "bearish_bos",
        "bullish_fvg",
        "bearish_fvg",
        "discount",
        "premium",
        "trend_bullish",
        "trend_bearish",
    }

    assert expected.issubset(features.columns)


def test_zone_detection_returns_dataframes():
    features, zones = detect_automatic_zones(
        sample_candles(),
        instrument="EUR_USD",
        impulse_atr_multiplier=1.0,
    )

    assert isinstance(features, pd.DataFrame)
    assert isinstance(zones, pd.DataFrame)


def test_setup_generator_handles_empty_zones():
    features = add_detection_features(sample_candles())
    setups = generate_automatic_setups(
        features,
        pd.DataFrame(),
    )

    assert isinstance(setups, pd.DataFrame)
    assert setups.empty
