from __future__ import annotations

import pandas as pd

from src.multi_timeframe import add_multi_timeframe_context


def _candles(rows: int = 1200) -> pd.DataFrame:
    times = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="5min",
        tz="UTC",
    )

    close = pd.Series(
        [1.1000 + (index * 0.00001) for index in range(rows)]
    )

    return pd.DataFrame(
        {
            "time": times,
            "mid_open": close - 0.00002,
            "mid_high": close + 0.00005,
            "mid_low": close - 0.00005,
            "mid_close": close,
        }
    )


def test_multi_timeframe_columns_are_created():
    result = add_multi_timeframe_context(_candles())

    expected = {
        "h1_trend",
        "h4_trend",
        "mtf_bullish_alignment",
        "mtf_bearish_alignment",
        "h1_h4_agree",
    }

    assert expected.issubset(result.columns)
    assert len(result) == 1200


def test_uptrend_eventually_aligns_bullish():
    result = add_multi_timeframe_context(_candles())
    mature = result.iloc[-1]

    assert mature["h1_trend"] == "bullish"
    assert mature["h4_trend"] == "bullish"
    assert mature["mtf_bullish_alignment"] == 1.0
    assert mature["h1_h4_agree"] == 1
