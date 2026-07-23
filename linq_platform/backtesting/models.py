"""Core data models for the native backtesting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class TradeDirection(str, Enum):
    """Supported trade directions."""

    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Supported entry order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class PositionStatus(str, Enum):
    """Lifecycle status for a simulated position."""

    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class ExitReason(str, Enum):
    """Reason a simulated trade was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLDING_BARS = "max_holding_bars"
    END_OF_DATA = "end_of_data"
    MANUAL = "manual"


@dataclass(frozen=True)
class Candle:
    """One OHLCV market-data bar."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        prices = (
            self.open,
            self.high,
            self.low,
            self.close,
        )

        if any(price <= 0 for price in prices):
            raise ValueError("Candle prices must be greater than zero.")

        if self.high < self.low:
            raise ValueError("Candle high cannot be below candle low.")

        if self.high < max(self.open, self.close):
            raise ValueError("Candle high must be at least the open and close.")

        if self.low > min(self.open, self.close):
            raise ValueError("Candle low must be no greater than the open and close.")

        if self.volume < 0:
            raise ValueError("Candle volume cannot be negative.")


@dataclass(frozen=True)
class TradeSignal:
    """Strategy instruction requesting a new position."""

    signal_id: str
    time: datetime
    direction: TradeDirection
    entry_price: float
    stop_price: float
    target_price: float | None = None
    order_type: OrderType = OrderType.MARKET
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.signal_id.strip():
            raise ValueError("signal_id cannot be empty.")

        if self.entry_price <= 0:
            raise ValueError("entry_price must be greater than zero.")

        if self.stop_price <= 0:
            raise ValueError("stop_price must be greater than zero.")

        if self.target_price is not None and self.target_price <= 0:
            raise ValueError("target_price must be greater than zero.")

        if self.direction is TradeDirection.LONG and self.stop_price >= self.entry_price:
            raise ValueError("A long stop must be below the entry price.")

        if self.direction is TradeDirection.SHORT and self.stop_price <= self.entry_price:
            raise ValueError("A short stop must be above the entry price.")

        if (
            self.target_price is not None
            and self.direction is TradeDirection.LONG
            and self.target_price <= self.entry_price
        ):
            raise ValueError("A long target must be above the entry price.")

        if (
            self.target_price is not None
            and self.direction is TradeDirection.SHORT
            and self.target_price >= self.entry_price
        ):
            raise ValueError("A short target must be below the entry price.")


@dataclass
class Position:
    """Mutable position state while a trade is being simulated."""

    position_id: str
    signal_id: str
    direction: TradeDirection
    entry_time: datetime
    entry_index: int
    entry_price: float
    stop_price: float
    target_price: float
    initial_risk_price: float
    quantity: float
    risk_amount: float
    status: PositionStatus = PositionStatus.OPEN
    bars_held: int = 0

    def __post_init__(self) -> None:
        if not self.position_id.strip():
            raise ValueError("position_id cannot be empty.")

        if self.entry_index < 0:
            raise ValueError("entry_index cannot be negative.")

        if self.entry_price <= 0:
            raise ValueError("entry_price must be greater than zero.")

        if self.initial_risk_price <= 0:
            raise ValueError("initial_risk_price must be greater than zero.")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")

        if self.risk_amount <= 0:
            raise ValueError("risk_amount must be greater than zero.")


@dataclass(frozen=True)
class Trade:
    """Final immutable record for a completed simulated trade."""

    trade_id: str
    signal_id: str
    direction: TradeDirection
    entry_time: datetime
    exit_time: datetime
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    quantity: float
    risk_amount: float
    gross_r: float
    costs_r: float
    net_r: float
    pnl_amount: float
    bars_held: int
    exit_reason: ExitReason
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.trade_id.strip():
            raise ValueError("trade_id cannot be empty.")

        if self.exit_index < self.entry_index:
            raise ValueError("exit_index cannot precede entry_index.")

        if self.bars_held < 0:
            raise ValueError("bars_held cannot be negative.")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero.")

        if self.risk_amount <= 0:
            raise ValueError("risk_amount must be greater than zero.")

    @property
    def won(self) -> bool:
        """Whether the trade finished with positive net R."""

        return self.net_r > 0

    @property
    def lost(self) -> bool:
        """Whether the trade finished with negative net R."""

        return self.net_r < 0

    @property
    def breakeven(self) -> bool:
        """Whether the trade finished at exactly zero net R."""

        return self.net_r == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable trade record."""

        result = asdict(self)
        result["direction"] = self.direction.value
        result["exit_reason"] = self.exit_reason.value
        result["entry_time"] = self.entry_time.isoformat()
        result["exit_time"] = self.exit_time.isoformat()
        return result
