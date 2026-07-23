from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

import pytest

from linq_platform.backtesting import (
    Candle,
    TradeDirection,
)
from linq_platform.strategies import (
    EmaAtrStrategyConfig,
    calculate_atr,
    calculate_ema,
    calculate_true_ranges,
    generate_ema_atr_signals,
)


START = datetime(2026, 1, 1)


def candle(
    index: int,
    close: float,
) -> Candle:
    open_price = close

    return Candle(
        time=START + timedelta(hours=index),
        open=open_price,
        high=close + 0.0010,
        low=close - 0.0010,
        close=close,
        volume=1000.0,
    )


def test_invalid_strategy_periods_raise() -> None:
    with pytest.raises(
        ValueError,
        match="slow_period",
    ):
        EmaAtrStrategyConfig(
            fast_period=5,
            slow_period=5,
        )


def test_calculate_ema() -> None:
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    result = calculate_ema(
        values,
        period=3,
    )

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_true_range_uses_previous_close() -> None:
    candles = [
        Candle(
            time=START,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1000,
        ),
        Candle(
            time=START + timedelta(hours=1),
            open=1.1050,
            high=1.1060,
            low=1.1040,
            close=1.1050,
        ),
    ]

    ranges = calculate_true_ranges(candles)

    assert ranges[0] == pytest.approx(0.0020)
    assert ranges[1] == pytest.approx(0.0060)


def test_calculate_atr_is_aligned() -> None:
    candles = [candle(index, 1.1000) for index in range(5)]

    result = calculate_atr(
        candles,
        period=3,
    )

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(0.0020)
    assert result[4] == pytest.approx(0.0020)


def test_generates_long_crossover_signal() -> None:
    closes = [
        1.1000,
        1.0990,
        1.0980,
        1.0970,
        1.0960,
        1.1000,
        1.1050,
        1.1100,
    ]

    candles = [candle(index, close) for index, close in enumerate(closes)]

    signals = generate_ema_atr_signals(
        candles,
        EmaAtrStrategyConfig(
            fast_period=2,
            slow_period=4,
            atr_period=2,
            atr_stop_multiplier=1.0,
        ),
    )

    assert any(signal.direction is TradeDirection.LONG for signal in signals)


def test_generates_short_crossover_signal() -> None:
    closes = [
        1.1000,
        1.1010,
        1.1020,
        1.1030,
        1.1040,
        1.1000,
        1.0950,
        1.0900,
    ]

    candles = [candle(index, close) for index, close in enumerate(closes)]

    signals = generate_ema_atr_signals(
        candles,
        EmaAtrStrategyConfig(
            fast_period=2,
            slow_period=4,
            atr_period=2,
            atr_stop_multiplier=1.0,
        ),
    )

    assert any(signal.direction is TradeDirection.SHORT for signal in signals)


def test_direction_filter_disables_shorts() -> None:
    closes = [
        1.1000,
        1.1010,
        1.1020,
        1.1030,
        1.1040,
        1.1000,
        1.0950,
        1.0900,
    ]

    candles = [candle(index, close) for index, close in enumerate(closes)]

    signals = generate_ema_atr_signals(
        candles,
        EmaAtrStrategyConfig(
            fast_period=2,
            slow_period=4,
            atr_period=2,
            allow_long=True,
            allow_short=False,
        ),
    )

    assert all(signal.direction is TradeDirection.LONG for signal in signals)


def test_insufficient_candles_returns_empty() -> None:
    signals = generate_ema_atr_signals(
        [
            candle(0, 1.1000),
            candle(1, 1.1010),
        ],
        EmaAtrStrategyConfig(
            fast_period=2,
            slow_period=4,
            atr_period=2,
        ),
    )

    assert signals == []


def test_cli_runs_without_signal_csv(
    tmp_path: Path,
) -> None:
    candle_path = tmp_path / "candles.csv"
    output_path = tmp_path / "results"

    closes = [
        1.1000,
        1.0990,
        1.0980,
        1.0970,
        1.0960,
        1.1000,
        1.1050,
        1.1100,
        1.1080,
        1.1060,
        1.1040,
    ]

    lines = ["time,open,high,low,close,volume"]

    for index, close in enumerate(closes):
        timestamp = (START + timedelta(hours=index)).strftime("%Y-%m-%d %H:%M:%S")

        lines.append(f"{timestamp},{close},{close + 0.0010},{close - 0.0010},{close},1000")

    candle_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "run_strategy_backtest.py",
            "--candles",
            str(candle_path),
            "--output",
            str(output_path),
            "--fast-period",
            "2",
            "--slow-period",
            "4",
            "--atr-period",
            "2",
            "--spread-pips",
            "0",
            "--slippage-pips",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Strategy backtest complete." in completed.stdout
    assert (output_path / "metrics.json").exists()
    assert (output_path / "trades.csv").exists()
