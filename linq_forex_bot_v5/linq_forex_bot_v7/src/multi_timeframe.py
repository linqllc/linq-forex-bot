from __future__ import annotations

import pandas as pd


def _timeframe_context(
    candles: pd.DataFrame,
    rule: str,
    prefix: str,
    fast_ema: int,
    slow_ema: int,
) -> pd.DataFrame:
    """
    Build higher-timeframe context from M5 candles.

    Bar labels use the closing timestamp. merge_asof later uses only completed
    higher-timeframe bars, preventing future-data leakage.
    """
    source = candles.copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    source = source.sort_values("time").set_index("time")

    higher = (
        source.resample(rule, label="right", closed="right")
        .agg(
            mid_open=("mid_open", "first"),
            mid_high=("mid_high", "max"),
            mid_low=("mid_low", "min"),
            mid_close=("mid_close", "last"),
        )
        .dropna()
        .reset_index()
    )

    higher[f"{prefix}_ema_fast"] = (
        higher["mid_close"].ewm(span=fast_ema, adjust=False).mean()
    )
    higher[f"{prefix}_ema_slow"] = (
        higher["mid_close"].ewm(span=slow_ema, adjust=False).mean()
    )

    higher[f"{prefix}_trend"] = "neutral"
    higher.loc[
        higher[f"{prefix}_ema_fast"] > higher[f"{prefix}_ema_slow"],
        f"{prefix}_trend",
    ] = "bullish"
    higher.loc[
        higher[f"{prefix}_ema_fast"] < higher[f"{prefix}_ema_slow"],
        f"{prefix}_trend",
    ] = "bearish"

    higher[f"{prefix}_ema_separation"] = (
        higher[f"{prefix}_ema_fast"] - higher[f"{prefix}_ema_slow"]
    ).abs()

    return higher[
        [
            "time",
            f"{prefix}_ema_fast",
            f"{prefix}_ema_slow",
            f"{prefix}_ema_separation",
            f"{prefix}_trend",
        ]
    ]


def add_multi_timeframe_context(
    candles: pd.DataFrame,
    fast_ema: int = 20,
    slow_ema: int = 50,
) -> pd.DataFrame:
    """
    Attach completed H1 and H4 trend context to each entry-timeframe candle.
    """
    if candles.empty:
        return candles.copy()

    result = candles.copy()
    result["time"] = pd.to_datetime(result["time"], utc=True)
    result = result.sort_values("time").reset_index(drop=True)

    h1 = _timeframe_context(result, "1h", "h1", fast_ema, slow_ema)
    h4 = _timeframe_context(result, "4h", "h4", fast_ema, slow_ema)

    result = pd.merge_asof(
        result,
        h1.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    result = pd.merge_asof(
        result,
        h4.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    for column in ("h1_trend", "h4_trend"):
        result[column] = result[column].fillna("neutral")

    result["mtf_bullish_alignment"] = (
        (result["h1_trend"] == "bullish").astype(int)
        + (result["h4_trend"] == "bullish").astype(int)
    ) / 2.0

    result["mtf_bearish_alignment"] = (
        (result["h1_trend"] == "bearish").astype(int)
        + (result["h4_trend"] == "bearish").astype(int)
    ) / 2.0

    result["h1_h4_agree"] = (
        (result["h1_trend"] == result["h4_trend"])
        & (result["h1_trend"] != "neutral")
    ).astype(int)

    return result
