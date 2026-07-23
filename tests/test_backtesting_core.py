from __future__ import annotations

from datetime import datetime

import pytest

from linq_platform.backtesting import (
    BacktestConfig,
    Candle,
    ExecutionConfig,
    RiskConfig,
    StrategyConfig,
    TradeDirection,
    TradeSignal,
    calculate_position_size,
    calculate_target_price,
)


def test_default_backtest_config_is_valid() -> None:
    config = BacktestConfig()

    assert config.strategy.target_r == 2.0
    assert config.execution.pip_size == pytest.approx(0.0001)
    assert config.risk.account_balance == pytest.approx(10_000.0)


def test_valid_candle() -> None:
    candle = Candle(
        time=datetime(2026, 1, 1),
        open=1.1000,
        high=1.1050,
        low=1.0950,
        close=1.1020,
        volume=1000,
    )

    assert candle.high == pytest.approx(1.1050)


def test_invalid_candle_range_raises() -> None:
    with pytest.raises(ValueError):
        Candle(
            time=datetime(2026, 1, 1),
            open=1.1000,
            high=1.0900,
            low=1.0950,
            close=1.1020,
        )


def test_long_signal_requires_stop_below_entry() -> None:
    with pytest.raises(
        ValueError,
        match="long stop",
    ):
        TradeSignal(
            signal_id="signal-1",
            time=datetime(2026, 1, 1),
            direction=TradeDirection.LONG,
            entry_price=1.1000,
            stop_price=1.1010,
        )


def test_short_signal_requires_stop_above_entry() -> None:
    with pytest.raises(
        ValueError,
        match="short stop",
    ):
        TradeSignal(
            signal_id="signal-2",
            time=datetime(2026, 1, 1),
            direction=TradeDirection.SHORT,
            entry_price=1.1000,
            stop_price=1.0990,
        )


def test_long_target_price() -> None:
    target = calculate_target_price(
        direction=TradeDirection.LONG,
        entry_price=1.1000,
        stop_price=1.0950,
        target_r=2.0,
    )

    assert target == pytest.approx(1.1100)


def test_short_target_price() -> None:
    target = calculate_target_price(
        direction=TradeDirection.SHORT,
        entry_price=1.1000,
        stop_price=1.1050,
        target_r=2.0,
    )

    assert target == pytest.approx(1.0900)


def test_position_size_uses_account_risk() -> None:
    quantity, risk_amount = calculate_position_size(
        entry_price=1.1000,
        stop_price=1.0950,
        risk_config=RiskConfig(
            account_balance=10_000.0,
            risk_per_trade=0.01,
        ),
    )

    assert risk_amount == pytest.approx(100.0)
    assert quantity == pytest.approx(20_000.0)


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: ExecutionConfig(spread_pips=-1),
        lambda: ExecutionConfig(slippage_pips=-1),
        lambda: ExecutionConfig(pip_size=0),
        lambda: ExecutionConfig(same_bar_exit_policy="invalid"),
        lambda: RiskConfig(account_balance=0),
        lambda: RiskConfig(risk_per_trade=0),
        lambda: RiskConfig(risk_per_trade=1.1),
        lambda: RiskConfig(maximum_open_positions=0),
        lambda: StrategyConfig(target_r=0),
        lambda: StrategyConfig(maximum_holding_bars=0),
        lambda: StrategyConfig(
            allow_long=False,
            allow_short=False,
        ),
    ],
)
def test_invalid_config_raises(
    config_factory: object,
) -> None:
    with pytest.raises(ValueError):
        config_factory()
