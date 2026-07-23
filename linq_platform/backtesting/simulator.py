"""Candle-by-candle position simulation."""

from __future__ import annotations

from collections.abc import Sequence

from linq_platform.backtesting.config import BacktestConfig
from linq_platform.backtesting.execution import (
    calculate_exit_fill,
)
from linq_platform.backtesting.models import (
    Candle,
    ExitReason,
    Position,
    PositionStatus,
    Trade,
    TradeDirection,
)


def _stop_was_hit(
    position: Position,
    candle: Candle,
) -> bool:
    if position.direction is TradeDirection.LONG:
        return candle.low <= position.stop_price

    return candle.high >= position.stop_price


def _target_was_hit(
    position: Position,
    candle: Candle,
) -> bool:
    if position.direction is TradeDirection.LONG:
        return candle.high >= position.target_price

    return candle.low <= position.target_price


def _stop_exit_price(
    position: Position,
    candle: Candle,
) -> float:
    """Return a gap-aware stop execution price."""

    if position.direction is TradeDirection.LONG:
        if candle.open <= position.stop_price:
            return candle.open

        return position.stop_price

    if candle.open >= position.stop_price:
        return candle.open

    return position.stop_price


def _target_exit_price(
    position: Position,
    candle: Candle,
) -> float:
    """Return a gap-aware target execution price."""

    if position.direction is TradeDirection.LONG:
        if candle.open >= position.target_price:
            return candle.open

        return position.target_price

    if candle.open <= position.target_price:
        return candle.open

    return position.target_price


def calculate_gross_r(
    *,
    position: Position,
    exit_price: float,
) -> float:
    """Calculate trade result before commission in R multiples."""

    if position.initial_risk_price <= 0:
        raise ValueError("Position initial risk must be greater than zero.")

    if position.direction is TradeDirection.LONG:
        price_result = exit_price - position.entry_price
    else:
        price_result = position.entry_price - exit_price

    return price_result / position.initial_risk_price


def close_position(
    *,
    position: Position,
    candle: Candle,
    exit_index: int,
    requested_exit_price: float,
    exit_reason: ExitReason,
    config: BacktestConfig,
) -> Trade:
    """Close an open position and create its immutable trade record."""

    if position.status is not PositionStatus.OPEN:
        raise ValueError("Only an open position can be closed.")

    if exit_index < position.entry_index:
        raise ValueError("exit_index cannot precede entry_index.")

    exit_fill = calculate_exit_fill(
        direction=position.direction,
        requested_price=requested_exit_price,
        config=config,
    )

    gross_r = calculate_gross_r(
        position=position,
        exit_price=exit_fill,
    )
    costs_r = config.execution.commission_r
    net_r = gross_r - costs_r
    bars_held = exit_index - position.entry_index

    position.status = PositionStatus.CLOSED
    position.bars_held = bars_held

    return Trade(
        trade_id=f"trade-{position.position_id}",
        signal_id=position.signal_id,
        direction=position.direction,
        entry_time=position.entry_time,
        exit_time=candle.time,
        entry_index=position.entry_index,
        exit_index=exit_index,
        entry_price=position.entry_price,
        exit_price=exit_fill,
        stop_price=position.stop_price,
        target_price=position.target_price,
        quantity=position.quantity,
        risk_amount=position.risk_amount,
        gross_r=gross_r,
        costs_r=costs_r,
        net_r=net_r,
        pnl_amount=net_r * position.risk_amount,
        bars_held=bars_held,
        exit_reason=exit_reason,
    )


def evaluate_position_on_candle(
    *,
    position: Position,
    candle: Candle,
    candle_index: int,
    config: BacktestConfig,
) -> Trade | None:
    """
    Evaluate one open position against one completed candle.

    Stop and target exits are checked before maximum holding time.
    """

    if position.status is not PositionStatus.OPEN:
        raise ValueError("Only open positions can be evaluated.")

    if candle_index < position.entry_index:
        raise ValueError("candle_index cannot precede entry_index.")

    stop_hit = _stop_was_hit(position, candle)
    target_hit = _target_was_hit(position, candle)

    if stop_hit and target_hit:
        if config.execution.same_bar_exit_policy == "target_first":
            return close_position(
                position=position,
                candle=candle,
                exit_index=candle_index,
                requested_exit_price=_target_exit_price(
                    position,
                    candle,
                ),
                exit_reason=ExitReason.TAKE_PROFIT,
                config=config,
            )

        return close_position(
            position=position,
            candle=candle,
            exit_index=candle_index,
            requested_exit_price=_stop_exit_price(
                position,
                candle,
            ),
            exit_reason=ExitReason.STOP_LOSS,
            config=config,
        )

    if stop_hit:
        return close_position(
            position=position,
            candle=candle,
            exit_index=candle_index,
            requested_exit_price=_stop_exit_price(
                position,
                candle,
            ),
            exit_reason=ExitReason.STOP_LOSS,
            config=config,
        )

    if target_hit:
        return close_position(
            position=position,
            candle=candle,
            exit_index=candle_index,
            requested_exit_price=_target_exit_price(
                position,
                candle,
            ),
            exit_reason=ExitReason.TAKE_PROFIT,
            config=config,
        )

    bars_held = candle_index - position.entry_index
    position.bars_held = bars_held

    if bars_held >= config.strategy.maximum_holding_bars:
        return close_position(
            position=position,
            candle=candle,
            exit_index=candle_index,
            requested_exit_price=candle.close,
            exit_reason=ExitReason.MAX_HOLDING_BARS,
            config=config,
        )

    return None


def close_at_end_of_data(
    *,
    position: Position,
    candle: Candle,
    candle_index: int,
    config: BacktestConfig,
) -> Trade:
    """Close an open position using the final candle close."""

    return close_position(
        position=position,
        candle=candle,
        exit_index=candle_index,
        requested_exit_price=candle.close,
        exit_reason=ExitReason.END_OF_DATA,
        config=config,
    )


def simulate_position(
    *,
    candles: Sequence[Candle],
    position: Position,
    config: BacktestConfig,
) -> Trade:
    """
    Replay candles until the position exits.

    Simulation begins on the candle after entry because the signal entry
    is treated as occurring after the entry candle has completed.
    """

    if not candles:
        raise ValueError("candles cannot be empty.")

    if position.entry_index >= len(candles):
        raise ValueError("entry_index must reference an available candle.")

    first_evaluation_index = position.entry_index + 1

    for candle_index in range(
        first_evaluation_index,
        len(candles),
    ):
        trade = evaluate_position_on_candle(
            position=position,
            candle=candles[candle_index],
            candle_index=candle_index,
            config=config,
        )

        if trade is not None:
            return trade

    final_index = len(candles) - 1

    return close_at_end_of_data(
        position=position,
        candle=candles[final_index],
        candle_index=final_index,
        config=config,
    )
