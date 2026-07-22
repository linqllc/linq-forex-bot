from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _confirmed_pivots(
    series: pd.Series,
    left: int,
    right: int,
    high: bool,
) -> pd.Series:
    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    values = series.to_numpy(dtype=float)

    for pivot_index in range(left, len(values) - right):
        window = values[
            pivot_index - left:
            pivot_index + right + 1
        ]

        center = values[pivot_index]

        if high:
            is_pivot = center == np.max(window)
        else:
            is_pivot = center == np.min(window)

        if is_pivot:
            confirmation_index = pivot_index + right
            output.iloc[confirmation_index] = center

    return output


def _last_two_confirmed(
    confirmed_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Preserve the latest two distinct confirmed pivot values.

    The newest pivot only becomes available on its confirmation bar.
    Both values then remain available until another pivot confirms.
    """
    last_values: list[float] = []
    previous_values: list[float] = []

    latest = np.nan
    previous = np.nan

    for value in confirmed_series:
        if pd.notna(value):
            previous = latest
            latest = float(value)

        last_values.append(latest)
        previous_values.append(previous)

    last = pd.Series(
        last_values,
        index=confirmed_series.index,
        dtype=float,
    )

    previous_series = pd.Series(
        previous_values,
        index=confirmed_series.index,
        dtype=float,
    )

    return last, previous_series


def add_features(
    m5: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    df = m5.copy()

    df["atr"] = _atr(df, cfg.atr_period)
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = (
        df["high"] - df["low"]
    ).clip(lower=1e-12)

    df["body_range"] = (
        df["body"] / df["range"]
    )

    # Completed H1 candles only.
    h1 = (
        df.set_index("time")
        .resample(
            "1h",
            label="right",
            closed="right",
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        .dropna()
        .reset_index()
    )

    h1["ema_fast"] = h1["close"].ewm(
        span=cfg.h1_ema_fast,
        adjust=False,
    ).mean()

    h1["ema_slow"] = h1["close"].ewm(
        span=cfg.h1_ema_slow,
        adjust=False,
    ).mean()

    h1["pivot_high_known"] = _confirmed_pivots(
        h1["high"],
        cfg.h1_pivot_left,
        cfg.h1_pivot_right,
        high=True,
    )

    h1["pivot_low_known"] = _confirmed_pivots(
        h1["low"],
        cfg.h1_pivot_left,
        cfg.h1_pivot_right,
        high=False,
    )

    (
        h1["last_high"],
        h1["prev_high"],
    ) = _last_two_confirmed(
        h1["pivot_high_known"]
    )

    (
        h1["last_low"],
        h1["prev_low"],
    ) = _last_two_confirmed(
        h1["pivot_low_known"]
    )

    h1["bull_structure"] = (
        h1["last_high"].notna()
        & h1["prev_high"].notna()
        & h1["last_low"].notna()
        & h1["prev_low"].notna()
        & (h1["last_high"] > h1["prev_high"])
        & (h1["last_low"] > h1["prev_low"])
    )

    h1["bear_structure"] = (
        h1["last_high"].notna()
        & h1["prev_high"].notna()
        & h1["last_low"].notna()
        & h1["prev_low"].notna()
        & (h1["last_high"] < h1["prev_high"])
        & (h1["last_low"] < h1["prev_low"])
    )

    h1["h1_direction"] = np.select(
        [
            (
                h1["bull_structure"]
                & (h1["ema_fast"] > h1["ema_slow"])
            ),
            (
                h1["bear_structure"]
                & (h1["ema_fast"] < h1["ema_slow"])
            ),
        ],
        [1, -1],
        default=0,
    ).astype(int)

    context = h1[
        [
            "time",
            "h1_direction",
            "last_high",
            "last_low",
            "bull_structure",
            "bear_structure",
        ]
    ].copy()

    df = pd.merge_asof(
        df.sort_values("time"),
        context.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )

    df["h1_direction"] = (
        df["h1_direction"]
        .fillna(0)
        .astype(int)
    )

    df["bull_structure"] = (
        df["bull_structure"]
        .fillna(False)
        .astype(bool)
    )

    df["bear_structure"] = (
        df["bear_structure"]
        .fillna(False)
        .astype(bool)
    )

    return df
