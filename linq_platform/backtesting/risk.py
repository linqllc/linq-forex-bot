"""Risk and position-sizing helpers."""

from __future__ import annotations

from linq_platform.backtesting.config import RiskConfig
from linq_platform.backtesting.models import TradeDirection


def calculate_target_price(
    *,
    direction: TradeDirection,
    entry_price: float,
    stop_price: float,
    target_r: float,
) -> float:
    """Calculate the take-profit price from a fixed R multiple."""

    if target_r <= 0:
        raise ValueError("target_r must be greater than zero.")

    risk_distance = abs(entry_price - stop_price)

    if risk_distance <= 0:
        raise ValueError("Entry price and stop price must differ.")

    if direction is TradeDirection.LONG:
        return entry_price + risk_distance * target_r

    return entry_price - risk_distance * target_r


def calculate_position_size(
    *,
    entry_price: float,
    stop_price: float,
    risk_config: RiskConfig,
) -> tuple[float, float]:
    """
    Return position quantity and currency risk.

    Quantity is calculated using price distance. Instrument-specific
    contract multipliers can be added in a later milestone.
    """

    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        raise ValueError("Entry price and stop price must differ.")

    risk_amount = risk_config.account_balance * risk_config.risk_per_trade

    quantity = risk_amount / stop_distance

    return float(quantity), float(risk_amount)
