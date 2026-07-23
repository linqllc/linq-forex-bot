"""Deterministic EMA crossover strategy with ATR-based stops."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from linq_platform.backtesting.models import (
    Candle,
    TradeDirection,
    TradeSignal,
)


@dataclass(frozen=True)
class EmaAtrStrategyConfig:
    """Configuration for the EMA/ATR baseline strategy."""

    fast_period: int = 10
    slow_period: int = 30
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    minimum_stop_pips: float = 5.0
    pip_size: float = 0.0001
    allow_long: bool = True
    allow_short: bool = True

    def __post_init__(self) -> None:
        if self.fast_period <= 0:
            raise ValueError("fast_period must be greater than zero.")

        if self.slow_period <= self.fast_period:
            raise ValueError("slow_period must be greater than fast_period.")

        if self.atr_period <= 0:
            raise ValueError("atr_period must be greater than zero.")

        if self.atr_stop_multiplier <= 0:
            raise ValueError("atr_stop_multiplier must be greater than zero.")

        if self.minimum_stop_pips <= 0:
            raise ValueError("minimum_stop_pips must be greater than zero.")

        if self.pip_size <= 0:
            raise ValueError("pip_size must be greater than zero.")

        if not self.allow_long and not self.allow_short:
            raise ValueError("At least one trade direction must be enabled.")


def calculate_ema(
    values: Sequence[float],
    period: int,
) -> list[float | None]:
    """Calculate an EMA aligned to the supplied values."""

    if period <= 0:
        raise ValueError("period must be greater than zero.")

    if not values:
        return []

    for value in values:
        if not isfinite(value):
            raise ValueError("EMA values must all be finite.")

    output: list[float | None] = [None for _ in values]

    if len(values) < period:
        return output

    initial_average = sum(values[:period]) / period
    output[period - 1] = initial_average

    multiplier = 2.0 / (period + 1.0)
    previous_ema = initial_average

    for index in range(period, len(values)):
        current_ema = (values[index] - previous_ema) * multiplier + previous_ema

        output[index] = current_ema
        previous_ema = current_ema

    return output


def calculate_true_ranges(
    candles: Sequence[Candle],
) -> list[float]:
    """Calculate true range for each candle."""

    if not candles:
        return []

    true_ranges: list[float] = []

    for index, candle in enumerate(candles):
        if index == 0:
            true_range = candle.high - candle.low
        else:
            previous_close = candles[index - 1].close

            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )

        true_ranges.append(float(true_range))

    return true_ranges


def calculate_atr(
    candles: Sequence[Candle],
    period: int,
) -> list[float | None]:
    """Calculate Wilder's average true range."""

    if period <= 0:
        raise ValueError("period must be greater than zero.")

    if not candles:
        return []

    true_ranges = calculate_true_ranges(candles)

    output: list[float | None] = [None for _ in candles]

    if len(candles) < period:
        return output

    initial_atr = sum(true_ranges[:period]) / period

    output[period - 1] = initial_atr
    previous_atr = initial_atr

    for index in range(period, len(candles)):
        current_atr = ((previous_atr * (period - 1)) + true_ranges[index]) / period

        output[index] = current_atr
        previous_atr = current_atr

    return output


def _crossed_above(
    previous_fast: float,
    previous_slow: float,
    current_fast: float,
    current_slow: float,
) -> bool:
    return previous_fast <= previous_slow and current_fast > current_slow


def _crossed_below(
    previous_fast: float,
    previous_slow: float,
    current_fast: float,
    current_slow: float,
) -> bool:
    return previous_fast >= previous_slow and current_fast < current_slow


def generate_ema_atr_signals(
    candles: Sequence[Candle],
    config: EmaAtrStrategyConfig | None = None,
) -> list[TradeSignal]:
    """
    Generate signals from completed candle data.

    A signal occurs only when the fast EMA crosses the slow EMA.
    Entry is placed at the signal candle's close. The historical
    simulator begins evaluating the position on the next candle.
    """

    resolved = config or EmaAtrStrategyConfig()

    minimum_candles = (
        max(
            resolved.slow_period,
            resolved.atr_period,
        )
        + 1
    )

    if len(candles) < minimum_candles:
        return []

    closes = [candle.close for candle in candles]

    fast_ema = calculate_ema(
        closes,
        resolved.fast_period,
    )
    slow_ema = calculate_ema(
        closes,
        resolved.slow_period,
    )
    atr_values = calculate_atr(
        candles,
        resolved.atr_period,
    )

    signals: list[TradeSignal] = []
    minimum_stop_distance = resolved.minimum_stop_pips * resolved.pip_size

    for index in range(1, len(candles)):
        previous_fast = fast_ema[index - 1]
        previous_slow = slow_ema[index - 1]
        current_fast = fast_ema[index]
        current_slow = slow_ema[index]
        current_atr = atr_values[index]

        if any(
            value is None
            for value in (
                previous_fast,
                previous_slow,
                current_fast,
                current_slow,
                current_atr,
            )
        ):
            continue

        assert previous_fast is not None
        assert previous_slow is not None
        assert current_fast is not None
        assert current_slow is not None
        assert current_atr is not None

        candle = candles[index]
        stop_distance = max(
            current_atr * resolved.atr_stop_multiplier,
            minimum_stop_distance,
        )

        if resolved.allow_long and _crossed_above(
            previous_fast,
            previous_slow,
            current_fast,
            current_slow,
        ):
            signals.append(
                TradeSignal(
                    signal_id=(f"ema-atr-long-{candle.time.isoformat()}"),
                    time=candle.time,
                    direction=TradeDirection.LONG,
                    entry_price=candle.close,
                    stop_price=(candle.close - stop_distance),
                    target_price=None,
                    metadata={
                        "strategy": "ema_atr_crossover",
                        "fast_period": (resolved.fast_period),
                        "slow_period": (resolved.slow_period),
                        "atr_period": (resolved.atr_period),
                        "atr": current_atr,
                        "fast_ema": current_fast,
                        "slow_ema": current_slow,
                    },
                )
            )

        elif resolved.allow_short and _crossed_below(
            previous_fast,
            previous_slow,
            current_fast,
            current_slow,
        ):
            signals.append(
                TradeSignal(
                    signal_id=(f"ema-atr-short-{candle.time.isoformat()}"),
                    time=candle.time,
                    direction=TradeDirection.SHORT,
                    entry_price=candle.close,
                    stop_price=(candle.close + stop_distance),
                    target_price=None,
                    metadata={
                        "strategy": "ema_atr_crossover",
                        "fast_period": (resolved.fast_period),
                        "slow_period": (resolved.slow_period),
                        "atr_period": (resolved.atr_period),
                        "atr": current_atr,
                        "fast_ema": current_fast,
                        "slow_ema": current_slow,
                    },
                )
            )

    return signals
