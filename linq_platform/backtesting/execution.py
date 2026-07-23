"""Position creation and simulated fill-price calculations."""

from __future__ import annotations

from linq_platform.backtesting.config import BacktestConfig
from linq_platform.backtesting.models import (
    Position,
    PositionStatus,
    TradeDirection,
    TradeSignal,
)
from linq_platform.backtesting.risk import (
    calculate_position_size,
    calculate_target_price,
)


def _half_spread(config: BacktestConfig) -> float:
    """Return one side of the configured bid-ask spread."""

    return config.execution.spread_pips * config.execution.pip_size / 2.0


def _slippage(config: BacktestConfig) -> float:
    """Return configured slippage expressed in price units."""

    return config.execution.slippage_pips * config.execution.pip_size


def calculate_entry_fill(
    *,
    direction: TradeDirection,
    requested_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse spread and slippage to an entry price."""

    adjustment = _half_spread(config) + _slippage(config)

    if direction is TradeDirection.LONG:
        return requested_price + adjustment

    return requested_price - adjustment


def calculate_exit_fill(
    *,
    direction: TradeDirection,
    requested_price: float,
    config: BacktestConfig,
) -> float:
    """Apply adverse spread and slippage to an exit price."""

    adjustment = _half_spread(config) + _slippage(config)

    if direction is TradeDirection.LONG:
        return requested_price - adjustment

    return requested_price + adjustment


def open_position(
    *,
    signal: TradeSignal,
    entry_index: int,
    config: BacktestConfig,
) -> Position:
    """Create an open position from a validated trade signal."""

    if entry_index < 0:
        raise ValueError("entry_index cannot be negative.")

    if signal.direction is TradeDirection.LONG and not config.strategy.allow_long:
        raise ValueError("Long trades are disabled by the strategy configuration.")

    if signal.direction is TradeDirection.SHORT and not config.strategy.allow_short:
        raise ValueError("Short trades are disabled by the strategy configuration.")

    entry_fill = calculate_entry_fill(
        direction=signal.direction,
        requested_price=signal.entry_price,
        config=config,
    )

    if signal.direction is TradeDirection.LONG and signal.stop_price >= entry_fill:
        raise ValueError("The executed long entry must remain above the stop price.")

    if signal.direction is TradeDirection.SHORT and signal.stop_price <= entry_fill:
        raise ValueError("The executed short entry must remain below the stop price.")

    target_price = signal.target_price

    if target_price is None:
        target_price = calculate_target_price(
            direction=signal.direction,
            entry_price=entry_fill,
            stop_price=signal.stop_price,
            target_r=config.strategy.target_r,
        )

    if signal.direction is TradeDirection.LONG and target_price <= entry_fill:
        raise ValueError("The executed long target must be above entry.")

    if signal.direction is TradeDirection.SHORT and target_price >= entry_fill:
        raise ValueError("The executed short target must be below entry.")

    quantity, risk_amount = calculate_position_size(
        entry_price=entry_fill,
        stop_price=signal.stop_price,
        risk_config=config.risk,
    )

    return Position(
        position_id=f"position-{signal.signal_id}",
        signal_id=signal.signal_id,
        direction=signal.direction,
        entry_time=signal.time,
        entry_index=entry_index,
        entry_price=entry_fill,
        stop_price=signal.stop_price,
        target_price=target_price,
        initial_risk_price=abs(entry_fill - signal.stop_price),
        quantity=quantity,
        risk_amount=risk_amount,
        status=PositionStatus.OPEN,
    )
