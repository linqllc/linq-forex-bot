from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from linq_platform.backtesting import (
    BacktestConfig,
    Candle,
    ExecutionConfig,
    ExitReason,
    PositionStatus,
    StrategyConfig,
    TradeDirection,
    TradeSignal,
    calculate_entry_fill,
    calculate_exit_fill,
    open_position,
    simulate_position,
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


def long_signal() -> TradeSignal:
    return TradeSignal(
        signal_id="long-1",
        time=START,
        direction=TradeDirection.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
    )


def short_signal() -> TradeSignal:
    return TradeSignal(
        signal_id="short-1",
        time=START,
        direction=TradeDirection.SHORT,
        entry_price=1.1000,
        stop_price=1.1050,
    )


def test_long_entry_fill_applies_adverse_costs() -> None:
    config = BacktestConfig(
        execution=ExecutionConfig(
            spread_pips=2.0,
            slippage_pips=1.0,
        )
    )

    price = calculate_entry_fill(
        direction=TradeDirection.LONG,
        requested_price=1.1000,
        config=config,
    )

    assert price == pytest.approx(1.1002)


def test_short_entry_fill_applies_adverse_costs() -> None:
    config = BacktestConfig(
        execution=ExecutionConfig(
            spread_pips=2.0,
            slippage_pips=1.0,
        )
    )

    price = calculate_entry_fill(
        direction=TradeDirection.SHORT,
        requested_price=1.1000,
        config=config,
    )

    assert price == pytest.approx(1.0998)


def test_long_exit_fill_applies_adverse_costs() -> None:
    config = BacktestConfig(
        execution=ExecutionConfig(
            spread_pips=2.0,
            slippage_pips=1.0,
        )
    )

    price = calculate_exit_fill(
        direction=TradeDirection.LONG,
        requested_price=1.1100,
        config=config,
    )

    assert price == pytest.approx(1.1098)


def test_open_position_generates_target() -> None:
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    assert position.entry_price == pytest.approx(1.1000)
    assert position.stop_price == pytest.approx(1.0950)
    assert position.target_price == pytest.approx(1.1100)
    assert position.status is PositionStatus.OPEN


def test_long_take_profit() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1090,
        ),
    ]

    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=BacktestConfig(),
    )

    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.gross_r == pytest.approx(2.0)
    assert trade.net_r == pytest.approx(2.0)


def test_long_stop_loss() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1010,
            low=1.0940,
            close=1.0960,
        ),
    ]

    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=BacktestConfig(),
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.net_r == pytest.approx(-1.0)


def test_short_take_profit() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1010,
            low=1.0890,
            close=1.0910,
        ),
    ]

    position = open_position(
        signal=short_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=BacktestConfig(),
    )

    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.gross_r == pytest.approx(2.0)


def test_short_stop_loss() -> None:
    candles = [
        candle(0),
        candle(
            1,
            open_price=1.1000,
            high=1.1060,
            low=1.0990,
            close=1.1040,
        ),
    ]

    position = open_position(
        signal=short_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=BacktestConfig(),
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.net_r == pytest.approx(-1.0)


def test_same_bar_defaults_to_stop_first() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0940,
            close=1.1000,
        ),
    ]

    config = BacktestConfig(execution=ExecutionConfig(same_bar_exit_policy="stop_first"))
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=config,
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=config,
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS


def test_same_bar_can_use_target_first() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0940,
            close=1.1000,
        ),
    ]

    config = BacktestConfig(execution=ExecutionConfig(same_bar_exit_policy="target_first"))
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=config,
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=config,
    )

    assert trade.exit_reason is ExitReason.TAKE_PROFIT


def test_maximum_holding_bars_exit() -> None:
    candles = [
        candle(0),
        candle(1, close=1.1010),
        candle(2, close=1.1020),
        candle(3, close=1.1030),
    ]

    config = BacktestConfig(
        strategy=StrategyConfig(
            target_r=2.0,
            maximum_holding_bars=2,
        )
    )
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=config,
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=config,
    )

    assert trade.exit_reason is ExitReason.MAX_HOLDING_BARS
    assert trade.exit_index == 2
    assert trade.bars_held == 2


def test_end_of_data_exit() -> None:
    candles = [
        candle(0),
        candle(1, close=1.1010),
        candle(2, close=1.1020),
    ]

    config = BacktestConfig(strategy=StrategyConfig(maximum_holding_bars=20))
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=config,
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=config,
    )

    assert trade.exit_reason is ExitReason.END_OF_DATA
    assert trade.exit_price == pytest.approx(1.1020)


def test_commission_reduces_net_r() -> None:
    candles = [
        candle(0),
        candle(
            1,
            high=1.1110,
            low=1.0990,
            close=1.1100,
        ),
    ]

    config = BacktestConfig(execution=ExecutionConfig(commission_r=0.10))
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=config,
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=config,
    )

    assert trade.gross_r == pytest.approx(2.0)
    assert trade.costs_r == pytest.approx(0.10)
    assert trade.net_r == pytest.approx(1.90)


def test_gap_through_long_stop_uses_open() -> None:
    candles = [
        candle(0),
        candle(
            1,
            open_price=1.0930,
            high=1.0960,
            low=1.0920,
            close=1.0940,
        ),
    ]

    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    trade = simulate_position(
        candles=candles,
        position=position,
        config=BacktestConfig(),
    )

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == pytest.approx(1.0930)
    assert trade.net_r == pytest.approx(-1.4)


def test_disabled_long_trade_raises() -> None:
    config = BacktestConfig(
        strategy=StrategyConfig(
            allow_long=False,
            allow_short=True,
        )
    )

    with pytest.raises(ValueError, match="disabled"):
        open_position(
            signal=long_signal(),
            entry_index=0,
            config=config,
        )


def test_empty_candle_sequence_raises() -> None:
    position = open_position(
        signal=long_signal(),
        entry_index=0,
        config=BacktestConfig(),
    )

    with pytest.raises(ValueError, match="empty"):
        simulate_position(
            candles=[],
            position=position,
            config=BacktestConfig(),
        )
