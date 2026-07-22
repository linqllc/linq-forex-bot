from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


AmbiguousPolicy = Literal["stop_first", "target_first"]


@dataclass(frozen=True)
class OutcomeConfig:
    """Controls how far and how conservatively each setup is evaluated."""

    max_bars: int = 2016
    target_r_values: tuple[float, ...] = (1.0, 2.0, 3.0)
    ambiguous_policy: AmbiguousPolicy = "stop_first"

    def __post_init__(self) -> None:
        if self.max_bars <= 0:
            raise ValueError("max_bars must be greater than zero.")

        if not self.target_r_values:
            raise ValueError("target_r_values cannot be empty.")

        if any(value <= 0 for value in self.target_r_values):
            raise ValueError("All target R values must be greater than zero.")

        if self.ambiguous_policy not in {"stop_first", "target_first"}:
            raise ValueError(
                "ambiguous_policy must be 'stop_first' or 'target_first'."
            )


def _find_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    required: bool = True,
) -> str | None:
    lookup = {str(column).lower(): str(column) for column in frame.columns}

    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match

    if required:
        raise ValueError(
            f"Missing required column. Expected one of: {', '.join(candidates)}"
        )

    return None


def _normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    if candles.empty:
        raise ValueError("Candle data is empty.")

    result = candles.copy()

    time_column = _find_column(
        result,
        ("time", "timestamp", "datetime", "date"),
    )
    open_column = _find_column(result, ("open", "o"))
    high_column = _find_column(result, ("high", "h"))
    low_column = _find_column(result, ("low", "l"))
    close_column = _find_column(result, ("close", "c"))

    result = result.rename(
        columns={
            time_column: "time",
            open_column: "open",
            high_column: "high",
            low_column: "low",
            close_column: "close",
        }
    )

    result["time"] = pd.to_datetime(result["time"], utc=True, errors="coerce")

    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = (
        result.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    if result.empty:
        raise ValueError("No valid candles remained after normalization.")

    return result


def _normalize_setups(setups: pd.DataFrame) -> pd.DataFrame:
    if setups.empty:
        return setups.copy()

    result = setups.copy()

    timestamp_column = _find_column(
        result,
        ("timestamp", "time", "entry_time", "datetime"),
    )
    direction_column = _find_column(
        result,
        ("direction", "side", "trade_direction"),
    )
    entry_column = _find_column(
        result,
        ("entry", "entry_price"),
    )
    stop_column = _find_column(
        result,
        ("stop", "stop_loss", "stop_price"),
    )

    rename_map = {
        timestamp_column: "timestamp",
        direction_column: "direction",
        entry_column: "entry",
        stop_column: "stop",
    }

    result = result.rename(columns=rename_map)

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )
    result["direction"] = (
        result["direction"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"buy": "long", "sell": "short"})
    )
    result["entry"] = pd.to_numeric(result["entry"], errors="coerce")
    result["stop"] = pd.to_numeric(result["stop"], errors="coerce")

    result = result.dropna(
        subset=["timestamp", "direction", "entry", "stop"]
    ).reset_index(drop=True)

    invalid_directions = ~result["direction"].isin(["long", "short"])
    if invalid_directions.any():
        invalid = sorted(result.loc[invalid_directions, "direction"].unique())
        raise ValueError(f"Unsupported setup directions: {invalid}")

    return result


def _target_price(
    direction: str,
    entry: float,
    risk: float,
    target_r: float,
) -> float:
    if direction == "long":
        return entry + risk * target_r

    return entry - risk * target_r


def _evaluate_setup(
    setup: pd.Series,
    candles: pd.DataFrame,
    config: OutcomeConfig,
) -> dict:
    direction = str(setup["direction"])
    entry = float(setup["entry"])
    stop = float(setup["stop"])
    setup_time = setup["timestamp"]

    risk = abs(entry - stop)

    if risk <= 0:
        raise ValueError(
            f"Setup at {setup_time} has identical entry and stop prices."
        )

    future = candles.loc[candles["time"] > setup_time].head(config.max_bars)

    result = setup.to_dict()
    result["risk_distance"] = risk
    result["bars_evaluated"] = 0
    result["stop_hit"] = False
    result["stop_time"] = pd.NaT
    result["bars_to_stop"] = pd.NA
    result["mfe_r"] = 0.0
    result["mae_r"] = 0.0
    result["outcome"] = "open"

    targets = {
        target_r: _target_price(direction, entry, risk, target_r)
        for target_r in config.target_r_values
    }

    hit_targets = {target_r: False for target_r in config.target_r_values}
    target_times = {target_r: pd.NaT for target_r in config.target_r_values}
    bars_to_target = {target_r: pd.NA for target_r in config.target_r_values}

    max_favorable = 0.0
    max_adverse = 0.0
    ambiguous_stop = False

    for bar_number, candle in enumerate(
        future.itertuples(index=False),
        start=1,
    ):
        result["bars_evaluated"] = bar_number
        high = float(candle.high)
        low = float(candle.low)
        candle_time = candle.time

        if direction == "long":
            favorable = max(0.0, high - entry)
            adverse = max(0.0, entry - low)
            stop_touched = low <= stop

            targets_touched = [
                target_r
                for target_r, price in targets.items()
                if high >= price and not hit_targets[target_r]
            ]
        else:
            favorable = max(0.0, entry - low)
            adverse = max(0.0, high - entry)
            stop_touched = high >= stop

            targets_touched = [
                target_r
                for target_r, price in targets.items()
                if low <= price and not hit_targets[target_r]
            ]

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        same_bar_conflict = stop_touched and bool(targets_touched)

        if same_bar_conflict and config.ambiguous_policy == "stop_first":
            result["stop_hit"] = True
            result["stop_time"] = candle_time
            result["bars_to_stop"] = bar_number
            ambiguous_stop = True
            break

        for target_r in sorted(targets_touched):
            hit_targets[target_r] = True
            target_times[target_r] = candle_time
            bars_to_target[target_r] = bar_number

        highest_configured_target = max(config.target_r_values)

        if hit_targets[highest_configured_target]:
            break

        if stop_touched:
            result["stop_hit"] = True
            result["stop_time"] = candle_time
            result["bars_to_stop"] = bar_number
            break

    result["mfe_r"] = round(max_favorable / risk, 4)
    result["mae_r"] = round(max_adverse / risk, 4)

    for target_r in config.target_r_values:
        label = f"{target_r:g}r"
        hit = hit_targets[target_r]

        result[f"target_{label}"] = targets[target_r]
        result[f"hit_{label}"] = hit
        result[f"time_to_{label}"] = target_times[target_r]
        result[f"bars_to_{label}"] = bars_to_target[target_r]

        if hit:
            target_result = "win"
        elif ambiguous_stop:
            target_result = "ambiguous_loss"
        elif result["stop_hit"]:
            target_result = "loss"
        else:
            target_result = "open"

        result[f"{label}_result"] = target_result

    highest_target = max(
        (
            target_r
            for target_r, hit in hit_targets.items()
            if hit
        ),
        default=0.0,
    )

    if ambiguous_stop:
        result["outcome"] = "ambiguous_loss"
    elif result["stop_hit"] and highest_target == 0:
        result["outcome"] = "stopped"
    elif highest_target > 0:
        result["outcome"] = f"hit_{highest_target:g}r"
    elif future.empty:
        result["outcome"] = "no_future_data"
    else:
        result["outcome"] = "open"

    return result


def analyze_outcomes(
    candles: pd.DataFrame,
    setups: pd.DataFrame,
    config: OutcomeConfig | None = None,
) -> pd.DataFrame:
    """
    Evaluate setup performance against future candle highs and lows.

    Returns one row per setup with stop, target, MFE, MAE, and timing data.
    """
    active_config = config or OutcomeConfig()

    normalized_candles = _normalize_candles(candles)
    normalized_setups = _normalize_setups(setups)

    if normalized_setups.empty:
        return normalized_setups.copy()

    results = [
        _evaluate_setup(setup, normalized_candles, active_config)
        for _, setup in normalized_setups.iterrows()
    ]

    return pd.DataFrame(results)
