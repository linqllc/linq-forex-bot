"""Native market-memory primitives for the LINQ Trading Platform.

The functions in this module were extracted verbatim from the validated
``run_market_memory_v1.py`` reference implementation.  Keeping the calculations
unchanged allows native callers to move away from the script while regression
tests prove behavioral parity.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LEGACY_SOURCE = Path(__file__).resolve().parents[2] / "run_market_memory_v1.py"


def legacy_source_path() -> Path:
    """Return the frozen reference implementation path."""
    return LEGACY_SOURCE


def load_legacy_namespace() -> dict[str, Any]:
    """Load the frozen reference module without running its CLI entrypoint."""
    if not LEGACY_SOURCE.exists():
        raise FileNotFoundError(f"Legacy market-memory source not found: {LEGACY_SOURCE}")
    return runpy.run_path(str(LEGACY_SOURCE), run_name="linq_legacy_market_memory")


def available_symbols() -> tuple[str, ...]:
    """Return public callable symbols exposed by the reference module."""
    namespace = load_legacy_namespace()
    return tuple(
        sorted(
            name
            for name, value in namespace.items()
            if not name.startswith("_") and callable(value)
        )
    )


COLUMN_ALIASES = {
    "time": [
        "time",
        "timestamp",
        "datetime",
        "date",
    ],
    "open": [
        "open",
        "mid_open",
        "bid_open",
        "o",
    ],
    "high": [
        "high",
        "mid_high",
        "bid_high",
        "h",
    ],
    "low": [
        "low",
        "mid_low",
        "bid_low",
        "l",
    ],
    "close": [
        "close",
        "mid_close",
        "bid_close",
        "c",
    ],
    "volume": [
        "volume",
        "tick_volume",
        "vol",
    ],
}


class Config:
    # ATR
    atr_period: int = 14

    # Reaction detection
    reaction_atr: float = 1.0
    reaction_lookahead_bars: int = 24
    pivot_left_bars: int = 3
    pivot_right_bars: int = 3

    # Zone discovery
    zone_lookback_days: int = 14
    cluster_radius_atr: float = 0.50
    minimum_reactions: int = 2
    maximum_zone_width_atr: float = 2.0

    # Zone trade simulation
    stop_buffer_atr: float = 0.15
    maximum_trade_bars: int = 96
    targets_r: tuple[float, ...] = (
        1.0,
        1.25,
        1.5,
        2.0,
        2.5,
        3.0,
    )

    # Walk-forward testing
    research_days: int = 90
    train_days: int = 14
    test_days: int = 7
    step_days: int = 7

    # Reporting
    current_zone_lookback_days: int = 14
    top_zones_to_print: int = 10


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def find_column(
    normalized_columns: dict[str, str],
    aliases: Iterable[str],
) -> str | None:
    for alias in aliases:
        key = normalize_name(alias)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def load_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candle file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The candle CSV is empty.")

    normalized_columns = {normalize_name(column): column for column in df.columns}

    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        source = find_column(normalized_columns, aliases)

        if source is not None:
            rename_map[source] = canonical

    df = df.rename(columns=rename_map)

    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required candle columns: "
            + ", ".join(sorted(missing))
            + f"\nAvailable columns: {list(df.columns)}"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    if invalid.any():
        bad_count = int(invalid.sum())
        raise ValueError(f"Found {bad_count} candles with invalid OHLC relationships.")

    return df


def calculate_atr(
    df: pd.DataFrame,
    period: int,
) -> pd.Series:
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
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_features(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    result = df.copy()

    result["atr"] = calculate_atr(
        result,
        config.atr_period,
    )

    result["body_high"] = result[["open", "close"]].max(axis=1)
    result["body_low"] = result[["open", "close"]].min(axis=1)

    return result


def calculate_pivots(
    df: pd.DataFrame,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)

    pivot_low = np.zeros(len(df), dtype=bool)
    pivot_high = np.zeros(len(df), dtype=bool)

    for index in range(left, len(df) - right):
        low_window = lows[index - left : index + right + 1]
        high_window = highs[index - left : index + right + 1]

        pivot_low[index] = lows[index] <= np.min(low_window)
        pivot_high[index] = highs[index] >= np.max(high_window)

    return pivot_low, pivot_high


def first_threshold_hit(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    upper_price: float,
    lower_price: float,
) -> tuple[str | None, int | None]:
    """
    Determines which threshold is reached first.

    Conservative same-candle rule:
    If both levels are touched in the same candle, return "both".
    """

    for index in range(start_index, end_index + 1):
        candle_high = float(df.at[index, "high"])
        candle_low = float(df.at[index, "low"])

        upper_hit = candle_high >= upper_price
        lower_hit = candle_low <= lower_price

        if upper_hit and lower_hit:
            return "both", index

        if upper_hit:
            return "upper", index

        if lower_hit:
            return "lower", index

    return None, None


def candle_touches_zone(
    candle_low: float,
    candle_high: float,
    zone_low: float,
    zone_high: float,
) -> bool:
    return candle_high >= zone_low and candle_low <= zone_high


def find_first_zone_touch(
    df: pd.DataFrame,
    start_index: int,
    end_index: int,
    zone_low: float,
    zone_high: float,
) -> int | None:
    for index in range(start_index, end_index + 1):
        if candle_touches_zone(
            float(df.at[index, "low"]),
            float(df.at[index, "high"]),
            zone_low,
            zone_high,
        ):
            return index

    return None


def nearest_index_at_or_after(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> int | None:
    """Return the first candle index at or after timestamp."""
    if df.empty:
        return None

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    position = int(df["time"].searchsorted(timestamp, side="left"))

    if position >= len(df):
        return None

    return position


def nearest_index_before(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
) -> int | None:
    """Return the final candle index strictly before timestamp."""
    if df.empty:
        return None

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    position = int(df["time"].searchsorted(timestamp, side="left")) - 1

    if position < 0:
        return None

    return position


def maximum_drawdown_r(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = values.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak

    return float(drawdown.min())


def summarize_trades(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    summaries: list[dict] = []

    group_columns = [
        "zone_direction",
        "target_r",
    ]

    for keys, group in trades.groupby(group_columns):
        direction, target_r = keys

        wins = group["realized_r"] > 0
        losses = group["realized_r"] < 0

        gross_profit = float(group.loc[wins, "realized_r"].sum())

        gross_loss = abs(float(group.loc[losses, "realized_r"].sum()))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        window_results = group.groupby("window")["realized_r"].sum()

        positive_windows = float((window_results > 0).mean()) if not window_results.empty else 0.0

        summaries.append(
            {
                "direction": direction,
                "target_r": float(target_r),
                "trades": int(len(group)),
                "wins": int(wins.sum()),
                "losses": int(losses.sum()),
                "win_rate": float(wins.mean()),
                "expectancy_r": float(group["realized_r"].mean()),
                "net_r": float(group["realized_r"].sum()),
                "profit_factor": profit_factor,
                "maximum_drawdown_r": maximum_drawdown_r(
                    group.sort_values("entry_time")["realized_r"]
                ),
                "walk_forward_windows": int(group["window"].nunique()),
                "positive_window_rate": positive_windows,
            }
        )

    summary = pd.DataFrame(summaries)

    return summary.sort_values(
        [
            "expectancy_r",
            "positive_window_rate",
            "profit_factor",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def classify_zone_status(
    zone: pd.Series,
    current_price: float,
) -> str:
    zone_low = float(zone["zone_low"])
    zone_high = float(zone["zone_high"])

    if zone_low <= current_price <= zone_high:
        return "PRICE INSIDE ZONE"

    if zone["direction"] == "demand":
        if current_price > zone_high:
            return "BELOW PRICE — POTENTIAL BUY AREA"
        return "BROKEN OR ABOVE PRICE"

    if current_price < zone_low:
        return "ABOVE PRICE — POTENTIAL SELL AREA"

    return "BROKEN OR BELOW PRICE"


def format_price(price: float) -> str:
    return f"{price:.5f}"
