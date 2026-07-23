from __future__ import annotations

from datetime import datetime, timedelta
from math import isinf

import pytest

from linq_platform.backtesting import (
    BacktestConfig,
    Candle,
    ExitReason,
    RiskConfig,
    StrategyConfig,
    TradeDirection,
    TradeSignal,
    build_equity_curve_r,
    calculate_backtest_metrics,
    run_backtest,
)


START = datetime(2026, 1, 1)


def candle(
    index: int,
    *,
    open_price: float = 1.1000,
    high: float | None = None,
    low: float | None = None,
    close: float = 1.1000,
) -> Candle:
    if high is None:
        high = max(open_price, close) + 0.0010

    if low is None:
        low = min(open_price, close) - 0.0010

    return Candle(
        time=START + timedelta(hours=index),
        open=open_price,
        high=high,
        low=low,
        close=close,
    )


def long_signal(
    index: int,
    signal_id: str,
) -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        time=START + timedelta(hours=index),
        direction=TradeDirection.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
    )


def short_signal(
    index: int,
    signal_id: str,
) -> TradeSignal:
    return TradeSignal(
        signal_id=signal_id,
        time=START + timedelta(hours=index),
        direction=TradeDirection.SHORT,
        entry_price=1.1000,
        stop_price=1.1050,
    )


def test_run_backtest_executes_multiple_signals() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
        candle(2),
        candle(
            3,
            high=1.1010,
            low=1.0890,
            close=1.0900,
        ),
    ]

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "long"),
            short_signal(2, "short"),
        ],
    )

    assert result.signals_received == 2
    assert result.signals_executed == 2
    assert result.signals_skipped == 0
    assert len(result.trades) == 2
    assert result.metrics.net_r == pytest.approx(4.0)


def test_overlapping_signal_is_skipped() -> None:
    candles = [
        candle(0),
        candle(1),
        candle(2),
        candle(
            3,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    config = BacktestConfig(
        strategy=StrategyConfig(maximum_holding_bars=20),
        risk=RiskConfig(maximum_open_positions=1),
    )

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "first"),
            long_signal(1, "overlap"),
        ],
        config=config,
    )

    assert result.signals_executed == 1
    assert result.signals_skipped == 1


def test_multiple_positions_can_overlap() -> None:
    candles = [
        candle(0),
        candle(1),
        candle(
            2,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    config = BacktestConfig(risk=RiskConfig(maximum_open_positions=2))

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "first"),
            long_signal(1, "second"),
        ],
        config=config,
    )

    assert result.signals_executed == 2
    assert result.signals_skipped == 0


def test_signal_without_matching_candle_is_skipped() -> None:
    candles = [
        candle(0),
        candle(1),
    ]
    signal = TradeSignal(
        signal_id="missing-time",
        time=START + timedelta(minutes=30),
        direction=TradeDirection.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
    )

    result = run_backtest(
        candles=candles,
        signals=[signal],
    )

    assert result.signals_executed == 0
    assert result.signals_skipped == 1


def test_signal_on_final_candle_is_skipped() -> None:
    result = run_backtest(
        candles=[
            candle(0),
            candle(1),
        ],
        signals=[
            long_signal(1, "final"),
        ],
    )

    assert result.signals_executed == 0
    assert result.signals_skipped == 1


def test_unsorted_candles_raise() -> None:
    with pytest.raises(
        ValueError,
        match="strictly ordered",
    ):
        run_backtest(
            candles=[
                candle(1),
                candle(0),
            ],
            signals=[],
        )


def test_duplicate_candle_times_raise() -> None:
    duplicate = candle(0)

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        run_backtest(
            candles=[
                duplicate,
                duplicate,
            ],
            signals=[],
        )


def test_empty_candles_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_backtest(
            candles=[],
            signals=[],
        )


def test_empty_trade_metrics() -> None:
    metrics = calculate_backtest_metrics([])

    assert metrics.trade_count == 0
    assert metrics.net_r == 0
    assert metrics.maximum_drawdown_r == 0


def test_metrics_for_wins_and_losses() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
        candle(2),
        candle(
            3,
            high=1.1060,
            low=1.0990,
            close=1.1040,
        ),
    ]

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "winner"),
            short_signal(2, "loser"),
        ],
    )

    metrics = result.metrics

    assert metrics.trade_count == 2
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.net_r == pytest.approx(1.0)
    assert metrics.expectancy_r == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.maximum_drawdown_r == pytest.approx(1.0)


def test_all_winners_have_infinite_profit_factor() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "winner"),
        ],
    )

    assert isinf(result.metrics.profit_factor)


def test_equity_curve_begins_at_zero() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "winner"),
        ],
    )

    curve = build_equity_curve_r(result.trades)

    assert curve == pytest.approx([0.0, 2.0])


def test_result_serializes() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    result = run_backtest(
        candles=candles,
        signals=[
            long_signal(0, "winner"),
        ],
    )

    payload = result.to_dict()

    assert payload["signals_executed"] == 1
    assert payload["metrics"]["trade_count"] == 1
    assert len(payload["trades"]) == 1
    assert payload["trades"][0]["exit_reason"] == ExitReason.TAKE_PROFIT.value
