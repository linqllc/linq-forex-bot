from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import load_candles
from .features import add_market_features


@dataclass(frozen=True)
class PendingLimitConfig:
    """
    Execution assumptions for causal pending-limit entries.

    Input OHLC data is treated as mid-price data.
    """

    retrace_fraction: float = 0.25
    required_session: str = "london_open"

    spread_pips: float = 0.8
    limit_slippage_pips: float = 0.0
    exit_slippage_pips: float = 0.2

    max_open_trades: int = 1
    max_pending_orders: int = 20

    one_trade_per_impulse: bool = True
    one_trade_per_session: bool = False

    cancel_outside_required_session: bool = False

    minimum_stop_pips: float = 2.0

    account_starting_balance: float = 10_000.0
    risk_percentages: tuple[float, ...] = (
        0.25,
        0.50,
        1.00,
        2.00,
    )


@dataclass
class PendingLimitOrder:
    setup_id: str
    instrument: str
    direction: str

    impulse_index: int
    impulse_timestamp: pd.Timestamp

    created_index: int
    created_timestamp: pd.Timestamp
    expires_after_index: int

    impulse_low: float
    impulse_high: float
    impulse_size: float

    atr: float
    body_ratio: float

    retrace_fraction: float
    mid_limit_price: float
    executable_limit_price: float

    session: str


@dataclass
class LimitTrade:
    setup_id: str
    instrument: str
    direction: str

    impulse_index: int
    order_created_index: int
    entry_index: int

    impulse_timestamp: pd.Timestamp
    order_created_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp

    mid_limit_price: float
    executable_limit_price: float
    filled_entry_price: float

    stop_price: float
    target_price: float
    risk_price: float

    spread_pips: float
    limit_slippage_pips: float
    exit_slippage_pips: float

    retrace_fraction: float
    session: str

    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def _profit_factor(values: pd.Series) -> float | None:
    positive = float(values[values > 0].sum())
    negative = abs(float(values[values < 0].sum()))

    if negative == 0:
        if positive == 0:
            return None
        return float("inf")

    return positive / negative


def _max_drawdown_r(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = values.cumsum()
    high_watermark = equity.cummax()
    drawdown = equity - high_watermark

    return float(drawdown.min())


def performance_summary(trades: pd.DataFrame) -> dict:
    if trades.empty or "net_r" not in trades.columns:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "win_rate": None,
            "expectancy_r": None,
            "median_r": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "total_r": None,
            "average_win_r": None,
            "average_loss_r": None,
        }

    values = pd.to_numeric(
        trades["net_r"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "flats": 0,
            "win_rate": None,
            "expectancy_r": None,
            "median_r": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "total_r": None,
            "average_win_r": None,
            "average_loss_r": None,
        }

    wins = values[values > 0]
    losses = values[values < 0]
    flats = values[values == 0]

    return {
        "trades": int(len(values)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "flats": int(len(flats)),
        "win_rate": float((values > 0).mean()),
        "expectancy_r": float(values.mean()),
        "median_r": float(values.median()),
        "profit_factor": _profit_factor(values),
        "max_drawdown_r": _max_drawdown_r(values),
        "total_r": float(values.sum()),
        "average_win_r": (
            float(wins.mean())
            if not wins.empty
            else None
        ),
        "average_loss_r": (
            float(losses.mean())
            if not losses.empty
            else None
        ),
    }


def streak_summary(values: pd.Series) -> dict:
    longest_win_streak = 0
    longest_loss_streak = 0

    current_win_streak = 0
    current_loss_streak = 0

    for value in values:
        if value > 0:
            current_win_streak += 1
            current_loss_streak = 0

            longest_win_streak = max(
                longest_win_streak,
                current_win_streak,
            )

        elif value < 0:
            current_loss_streak += 1
            current_win_streak = 0

            longest_loss_streak = max(
                longest_loss_streak,
                current_loss_streak,
            )

        else:
            current_win_streak = 0
            current_loss_streak = 0

    return {
        "longest_win_streak": int(
            longest_win_streak
        ),
        "longest_loss_streak": int(
            longest_loss_streak
        ),
    }


class PendingLimitReplay:
    """
    Causal pending-limit replay.

    An order is created only after a displacement candle closes.
    The order can fill only on a later candle.
    """

    def __init__(
        self,
        research_config: ResearchConfig,
        limit_config: PendingLimitConfig,
    ):
        self.cfg = research_config
        self.execution = limit_config

        if not 0 < self.execution.retrace_fraction < 1:
            raise ValueError(
                "retrace_fraction must be between 0 and 1."
            )

        if self.execution.spread_pips < 0:
            raise ValueError(
                "spread_pips cannot be negative."
            )

        if self.execution.limit_slippage_pips < 0:
            raise ValueError(
                "limit_slippage_pips cannot be negative."
            )

        if self.execution.exit_slippage_pips < 0:
            raise ValueError(
                "exit_slippage_pips cannot be negative."
            )

        if self.execution.max_open_trades < 1:
            raise ValueError(
                "max_open_trades must be at least 1."
            )

    @property
    def half_spread_price(self) -> float:
        return (
            self.execution.spread_pips
            * self.cfg.pip_size
            / 2.0
        )

    @property
    def limit_slippage_price(self) -> float:
        return (
            self.execution.limit_slippage_pips
            * self.cfg.pip_size
        )

    @property
    def exit_slippage_price(self) -> float:
        return (
            self.execution.exit_slippage_pips
            * self.cfg.pip_size
        )

    def _session_key(
        self,
        timestamp: pd.Timestamp,
        session: str,
    ) -> str:
        timestamp = pd.Timestamp(timestamp)

        return (
            f"{timestamp.date().isoformat()}::{session}"
        )

    def _is_displacement(
        self,
        candle: pd.Series,
    ) -> bool:
        atr = float(candle.get("atr", np.nan))
        range_atr = float(
            candle.get("range_atr", np.nan)
        )
        body_ratio = float(
            candle.get("body_ratio", np.nan)
        )

        if not np.isfinite(atr) or atr <= 0:
            return False

        return bool(
            range_atr
            >= self.cfg.displacement_atr_min
            and body_ratio
            >= self.cfg.displacement_body_ratio_min
            and float(candle["close"])
            != float(candle["open"])
        )

    def _create_pending_order(
        self,
        index: int,
        candle: pd.Series,
    ) -> PendingLimitOrder | None:
        if not self._is_displacement(candle):
            return None

        candle_open = float(candle["open"])
        candle_close = float(candle["close"])
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        impulse_size = candle_high - candle_low

        if impulse_size <= 0:
            return None

        if candle_close > candle_open:
            direction = "long"

            mid_limit_price = (
                candle_high
                - self.execution.retrace_fraction
                * impulse_size
            )

            executable_limit_price = (
                mid_limit_price
                + self.half_spread_price
            )

        else:
            direction = "short"

            mid_limit_price = (
                candle_low
                + self.execution.retrace_fraction
                * impulse_size
            )

            executable_limit_price = (
                mid_limit_price
                - self.half_spread_price
            )

        timestamp = pd.Timestamp(
            candle["timestamp"]
        )

        return PendingLimitOrder(
            setup_id=(
                f"{self.cfg.instrument}-"
                f"{direction}-"
                f"{timestamp.isoformat()}"
            ),
            instrument=self.cfg.instrument,
            direction=direction,
            impulse_index=index,
            impulse_timestamp=timestamp,
            created_index=index,
            created_timestamp=timestamp,
            expires_after_index=(
                index + self.cfg.max_pullback_bars
            ),
            impulse_low=candle_low,
            impulse_high=candle_high,
            impulse_size=impulse_size,
            atr=float(candle["atr"]),
            body_ratio=float(candle["body_ratio"]),
            retrace_fraction=(
                self.execution.retrace_fraction
            ),
            mid_limit_price=float(mid_limit_price),
            executable_limit_price=float(
                executable_limit_price
            ),
            session=str(candle["session"]),
        )

    def _order_invalidated(
        self,
        order: PendingLimitOrder,
        candle: pd.Series,
    ) -> bool:
        if order.direction == "long":
            return (
                float(candle["low"])
                < order.impulse_low
            )

        return (
            float(candle["high"])
            > order.impulse_high
        )

    def _order_touched(
        self,
        order: PendingLimitOrder,
        candle: pd.Series,
    ) -> bool:
        """
        Mid candles are converted to executable bid/ask prices.

        Long orders fill when the candle's ask low reaches the
        buy-limit price.

        Short orders fill when the candle's bid high reaches the
        sell-limit price.
        """
        mid_low = float(candle["low"])
        mid_high = float(candle["high"])

        ask_low = (
            mid_low + self.half_spread_price
        )
        bid_high = (
            mid_high - self.half_spread_price
        )

        if order.direction == "long":
            return (
                ask_low
                <= order.executable_limit_price
            )

        return (
            bid_high
            >= order.executable_limit_price
        )

    def _fill_order(
        self,
        order: PendingLimitOrder,
        index: int,
        candle: pd.Series,
    ) -> LimitTrade:
        if order.direction == "long":
            filled_entry_price = (
                order.executable_limit_price
                + self.limit_slippage_price
            )
        else:
            filled_entry_price = (
                order.executable_limit_price
                - self.limit_slippage_price
            )

        minimum_risk_price = (
            self.execution.minimum_stop_pips
            * self.cfg.pip_size
        )

        risk_price = max(
            self.cfg.stop_atr * order.atr,
            minimum_risk_price,
        )

        if order.direction == "long":
            stop_price = (
                filled_entry_price - risk_price
            )
            target_price = (
                filled_entry_price
                + self.cfg.target_r * risk_price
            )
        else:
            stop_price = (
                filled_entry_price + risk_price
            )
            target_price = (
                filled_entry_price
                - self.cfg.target_r * risk_price
            )

        return LimitTrade(
            setup_id=order.setup_id,
            instrument=order.instrument,
            direction=order.direction,
            impulse_index=order.impulse_index,
            order_created_index=order.created_index,
            entry_index=index,
            impulse_timestamp=(
                order.impulse_timestamp
            ),
            order_created_timestamp=(
                order.created_timestamp
            ),
            entry_timestamp=pd.Timestamp(
                candle["timestamp"]
            ),
            mid_limit_price=order.mid_limit_price,
            executable_limit_price=(
                order.executable_limit_price
            ),
            filled_entry_price=float(
                filled_entry_price
            ),
            stop_price=float(stop_price),
            target_price=float(target_price),
            risk_price=float(risk_price),
            spread_pips=(
                self.execution.spread_pips
            ),
            limit_slippage_pips=(
                self.execution.limit_slippage_pips
            ),
            exit_slippage_pips=(
                self.execution.exit_slippage_pips
            ),
            retrace_fraction=(
                order.retrace_fraction
            ),
            session=str(candle["session"]),
        )

    def _tradable_extremes(
        self,
        candle: pd.Series,
    ) -> dict[str, float]:
        mid_high = float(candle["high"])
        mid_low = float(candle["low"])

        return {
            "bid_high": (
                mid_high - self.half_spread_price
            ),
            "bid_low": (
                mid_low - self.half_spread_price
            ),
            "ask_high": (
                mid_high + self.half_spread_price
            ),
            "ask_low": (
                mid_low + self.half_spread_price
            ),
        }

    def _update_trade(
        self,
        trade: LimitTrade,
        candle: pd.Series,
    ) -> dict | None:
        extremes = self._tradable_extremes(candle)

        trade.bars_held += 1

        if trade.direction == "long":
            favorable_r = (
                extremes["bid_high"]
                - trade.filled_entry_price
            ) / trade.risk_price

            adverse_r = (
                extremes["bid_low"]
                - trade.filled_entry_price
            ) / trade.risk_price

            stop_hit = (
                extremes["bid_low"]
                <= trade.stop_price
            )
            target_hit = (
                extremes["bid_high"]
                >= trade.target_price
            )

        else:
            favorable_r = (
                trade.filled_entry_price
                - extremes["ask_low"]
            ) / trade.risk_price

            adverse_r = (
                trade.filled_entry_price
                - extremes["ask_high"]
            ) / trade.risk_price

            stop_hit = (
                extremes["ask_high"]
                >= trade.stop_price
            )
            target_hit = (
                extremes["ask_low"]
                <= trade.target_price
            )

        trade.mfe_r = max(
            trade.mfe_r,
            float(favorable_r),
        )
        trade.mae_r = min(
            trade.mae_r,
            float(adverse_r),
        )

        exit_reason = None
        trigger_price = None
        gross_r = None

        if stop_hit and target_hit:
            if (
                self.cfg.intrabar_policy
                == "target_first"
            ):
                exit_reason = "target"
                trigger_price = trade.target_price
                gross_r = self.cfg.target_r
            else:
                exit_reason = "stop"
                trigger_price = trade.stop_price
                gross_r = -1.0

        elif stop_hit:
            exit_reason = "stop"
            trigger_price = trade.stop_price
            gross_r = -1.0

        elif target_hit:
            exit_reason = "target"
            trigger_price = trade.target_price
            gross_r = self.cfg.target_r

        elif (
            trade.bars_held
            >= self.cfg.max_holding_bars
        ):
            exit_reason = "time"
            trigger_price = float(candle["close"])

        if exit_reason is None:
            return None

        if exit_reason == "time":
            if trade.direction == "long":
                filled_exit_price = (
                    trigger_price
                    - self.half_spread_price
                    - self.exit_slippage_price
                )

                net_r = (
                    filled_exit_price
                    - trade.filled_entry_price
                ) / trade.risk_price

            else:
                filled_exit_price = (
                    trigger_price
                    + self.half_spread_price
                    + self.exit_slippage_price
                )

                net_r = (
                    trade.filled_entry_price
                    - filled_exit_price
                ) / trade.risk_price

            gross_r = net_r

        else:
            slippage_r = (
                self.exit_slippage_price
                / trade.risk_price
            )

            net_r = float(
                gross_r - slippage_r
            )

            if trade.direction == "long":
                filled_exit_price = (
                    trigger_price
                    - self.exit_slippage_price
                )
            else:
                filled_exit_price = (
                    trigger_price
                    + self.exit_slippage_price
                )

        return {
            **asdict(trade),
            "exit_timestamp": pd.Timestamp(
                candle["timestamp"]
            ),
            "trigger_exit_price": float(
                trigger_price
            ),
            "filled_exit_price": float(
                filled_exit_price
            ),
            "gross_r": float(gross_r),
            "net_r": float(net_r),
            "r_multiple": float(net_r),
            "outcome": (
                "win"
                if net_r > 0
                else "loss"
                if net_r < 0
                else "flat"
            ),
            "exit_reason": exit_reason,
            "hit_1r": bool(
                trade.mfe_r >= 1.0
            ),
            "hit_target": bool(
                trade.mfe_r
                >= self.cfg.target_r
            ),
        }

    def _period_summary(
        self,
        trades: pd.DataFrame,
        period_type: str,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        out = trades.copy()

        out["entry_timestamp"] = pd.to_datetime(
            out["entry_timestamp"],
            utc=True,
        )

        if period_type == "month":
            out["period"] = (
                out["entry_timestamp"]
                .dt.tz_localize(None)
                .dt.to_period("M")
                .astype(str)
            )
        elif period_type == "year":
            out["period"] = (
                out["entry_timestamp"]
                .dt.year
                .astype(str)
            )
        else:
            raise ValueError(period_type)

        rows = []

        for period, group in out.groupby(
            "period",
            sort=True,
        ):
            rows.append(
                {
                    "period": period,
                    **performance_summary(group),
                }
            )

        return pd.DataFrame(rows)

    def _risk_sizing(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        values = pd.to_numeric(
            trades["net_r"],
            errors="coerce",
        ).dropna()

        rows = []

        for risk_percent in (
            self.execution.risk_percentages
        ):
            balance = (
                self.execution
                .account_starting_balance
            )
            peak = balance
            max_drawdown_percent = 0.0

            for r_value in values:
                amount_at_risk = (
                    balance
                    * risk_percent
                    / 100.0
                )

                balance += (
                    amount_at_risk * r_value
                )

                peak = max(peak, balance)

                if peak > 0:
                    drawdown_percent = (
                        balance - peak
                    ) / peak * 100.0

                    max_drawdown_percent = min(
                        max_drawdown_percent,
                        drawdown_percent,
                    )

            rows.append(
                {
                    "risk_percent_per_trade": (
                        risk_percent
                    ),
                    "starting_balance": (
                        self.execution
                        .account_starting_balance
                    ),
                    "ending_balance": float(balance),
                    "net_profit": float(
                        balance
                        - self.execution
                        .account_starting_balance
                    ),
                    "return_percent": float(
                        (
                            balance
                            / self.execution
                            .account_starting_balance
                            - 1.0
                        )
                        * 100.0
                    ),
                    "max_drawdown_percent": float(
                        max_drawdown_percent
                    ),
                }
            )

        return pd.DataFrame(rows)

    def run(
        self,
        csv_path: str | Path,
        report_dir: str | Path,
    ) -> dict:
        report_dir = Path(report_dir)
        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candles = load_candles(
            csv_path,
            self.cfg.timestamp_column,
        )

        featured = add_market_features(
            candles,
            atr_period=self.cfg.atr_period,
            ema_fast=self.cfg.ema_fast,
            ema_slow=self.cfg.ema_slow,
        ).reset_index(drop=True)

        pending_orders: list[
            PendingLimitOrder
        ] = []

        open_trades: list[LimitTrade] = []

        created_order_rows: list[dict] = []
        canceled_order_rows: list[dict] = []
        filled_order_rows: list[dict] = []
        completed_trade_rows: list[dict] = []

        traded_impulses: set[int] = set()
        traded_sessions: set[str] = set()

        warmup = max(
            self.cfg.ema_slow + 12,
            250,
        )

        for index in range(len(featured)):
            candle = featured.iloc[index]
            timestamp = pd.Timestamp(
                candle["timestamp"]
            )

            still_open: list[LimitTrade] = []

            for trade in open_trades:
                # The fill candle cannot also determine the exit.
                if index <= trade.entry_index:
                    still_open.append(trade)
                    continue

                result = self._update_trade(
                    trade,
                    candle,
                )

                if result is None:
                    still_open.append(trade)
                else:
                    completed_trade_rows.append(
                        result
                    )

            open_trades = still_open

            surviving_orders: list[
                PendingLimitOrder
            ] = []

            for order in pending_orders:
                if index <= order.created_index:
                    surviving_orders.append(order)
                    continue

                if index > order.expires_after_index:
                    canceled_order_rows.append(
                        {
                            **asdict(order),
                            "cancel_index": index,
                            "cancel_timestamp": timestamp,
                            "cancel_reason": "expired",
                        }
                    )
                    continue

                if self._order_invalidated(
                    order,
                    candle,
                ):
                    canceled_order_rows.append(
                        {
                            **asdict(order),
                            "cancel_index": index,
                            "cancel_timestamp": timestamp,
                            "cancel_reason": (
                                "impulse_invalidated"
                            ),
                        }
                    )
                    continue

                if (
                    self.execution
                    .cancel_outside_required_session
                    and str(candle["session"])
                    != self.execution.required_session
                ):
                    canceled_order_rows.append(
                        {
                            **asdict(order),
                            "cancel_index": index,
                            "cancel_timestamp": timestamp,
                            "cancel_reason": (
                                "outside_required_session"
                            ),
                        }
                    )
                    continue

                if not self._order_touched(
                    order,
                    candle,
                ):
                    surviving_orders.append(order)
                    continue

                session_key = self._session_key(
                    timestamp,
                    str(candle["session"]),
                )

                reject_reason = None

                if (
                    len(open_trades)
                    >= self.execution.max_open_trades
                ):
                    reject_reason = (
                        "max_open_trades"
                    )

                if (
                    reject_reason is None
                    and self.execution
                    .one_trade_per_impulse
                    and order.impulse_index
                    in traded_impulses
                ):
                    reject_reason = (
                        "impulse_already_traded"
                    )

                if (
                    reject_reason is None
                    and self.execution
                    .one_trade_per_session
                    and session_key
                    in traded_sessions
                ):
                    reject_reason = (
                        "session_already_traded"
                    )

                if (
                    reject_reason is None
                    and str(candle["session"])
                    != self.execution.required_session
                ):
                    reject_reason = (
                        "fill_outside_required_session"
                    )

                if reject_reason is not None:
                    canceled_order_rows.append(
                        {
                            **asdict(order),
                            "cancel_index": index,
                            "cancel_timestamp": timestamp,
                            "cancel_reason": (
                                reject_reason
                            ),
                        }
                    )
                    continue

                trade = self._fill_order(
                    order,
                    index,
                    candle,
                )

                open_trades.append(trade)

                traded_impulses.add(
                    order.impulse_index
                )
                traded_sessions.add(
                    session_key
                )

                filled_order_rows.append(
                    {
                        **asdict(order),
                        "fill_index": index,
                        "fill_timestamp": timestamp,
                        "filled_entry_price": (
                            trade.filled_entry_price
                        ),
                        "risk_price": (
                            trade.risk_price
                        ),
                        "stop_price": (
                            trade.stop_price
                        ),
                        "target_price": (
                            trade.target_price
                        ),
                    }
                )

            pending_orders = surviving_orders

            if index < warmup:
                continue

            new_order = self._create_pending_order(
                index=index,
                candle=candle,
            )

            if new_order is None:
                continue

            if (
                len(pending_orders)
                >= self.execution.max_pending_orders
            ):
                canceled_order_rows.append(
                    {
                        **asdict(new_order),
                        "cancel_index": index,
                        "cancel_timestamp": timestamp,
                        "cancel_reason": (
                            "max_pending_orders"
                        ),
                    }
                )
                continue

            pending_orders.append(new_order)

            created_order_rows.append(
                asdict(new_order)
            )

        if len(featured):
            final_candle = featured.iloc[-1]
            final_timestamp = pd.Timestamp(
                final_candle["timestamp"]
            )
            final_mid_close = float(
                final_candle["close"]
            )

            for order in pending_orders:
                canceled_order_rows.append(
                    {
                        **asdict(order),
                        "cancel_index": (
                            len(featured) - 1
                        ),
                        "cancel_timestamp": (
                            final_timestamp
                        ),
                        "cancel_reason": (
                            "end_of_data"
                        ),
                    }
                )

            for trade in open_trades:
                if trade.direction == "long":
                    filled_exit_price = (
                        final_mid_close
                        - self.half_spread_price
                        - self.exit_slippage_price
                    )

                    net_r = (
                        filled_exit_price
                        - trade.filled_entry_price
                    ) / trade.risk_price

                else:
                    filled_exit_price = (
                        final_mid_close
                        + self.half_spread_price
                        + self.exit_slippage_price
                    )

                    net_r = (
                        trade.filled_entry_price
                        - filled_exit_price
                    ) / trade.risk_price

                completed_trade_rows.append(
                    {
                        **asdict(trade),
                        "exit_timestamp": (
                            final_timestamp
                        ),
                        "trigger_exit_price": (
                            final_mid_close
                        ),
                        "filled_exit_price": float(
                            filled_exit_price
                        ),
                        "gross_r": float(net_r),
                        "net_r": float(net_r),
                        "r_multiple": float(net_r),
                        "outcome": (
                            "win"
                            if net_r > 0
                            else "loss"
                            if net_r < 0
                            else "flat"
                        ),
                        "exit_reason": (
                            "end_of_data"
                        ),
                        "hit_1r": bool(
                            trade.mfe_r >= 1.0
                        ),
                        "hit_target": bool(
                            trade.mfe_r
                            >= self.cfg.target_r
                        ),
                    }
                )

        created_df = pd.DataFrame(
            created_order_rows
        )
        canceled_df = pd.DataFrame(
            canceled_order_rows
        )
        filled_df = pd.DataFrame(
            filled_order_rows
        )
        trades_df = pd.DataFrame(
            completed_trade_rows
        )

        if not trades_df.empty:
            trades_df = trades_df.sort_values(
                "entry_timestamp"
            ).reset_index(drop=True)

        monthly_df = self._period_summary(
            trades_df,
            "month",
        )

        yearly_df = self._period_summary(
            trades_df,
            "year",
        )

        risk_df = self._risk_sizing(
            trades_df
        )

        performance = performance_summary(
            trades_df
        )

        streaks = (
            streak_summary(
                trades_df["net_r"]
            )
            if not trades_df.empty
            else {
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
            }
        )

        cancellation_reasons = {}

        if (
            not canceled_df.empty
            and "cancel_reason"
            in canceled_df.columns
        ):
            cancellation_reasons = (
                canceled_df[
                    "cancel_reason"
                ]
                .value_counts(dropna=False)
                .to_dict()
            )

        prefix = (
            f"{self.cfg.instrument}"
            f"_v11_3_pending_limit"
        )

        created_path = (
            report_dir
            / f"{prefix}_created_orders.csv"
        )
        canceled_path = (
            report_dir
            / f"{prefix}_canceled_orders.csv"
        )
        filled_path = (
            report_dir
            / f"{prefix}_filled_orders.csv"
        )
        trades_path = (
            report_dir
            / f"{prefix}_trades.csv"
        )
        monthly_path = (
            report_dir
            / f"{prefix}_monthly.csv"
        )
        yearly_path = (
            report_dir
            / f"{prefix}_yearly.csv"
        )
        risk_path = (
            report_dir
            / f"{prefix}_risk_sizing.csv"
        )
        summary_path = (
            report_dir
            / f"{prefix}_summary.json"
        )

        created_df.to_csv(
            created_path,
            index=False,
        )
        canceled_df.to_csv(
            canceled_path,
            index=False,
        )
        filled_df.to_csv(
            filled_path,
            index=False,
        )
        trades_df.to_csv(
            trades_path,
            index=False,
        )
        monthly_df.to_csv(
            monthly_path,
            index=False,
        )
        yearly_df.to_csv(
            yearly_path,
            index=False,
        )
        risk_df.to_csv(
            risk_path,
            index=False,
        )

        summary = {
            "engine_version": (
                "11.3-causal-pending-limit"
            ),
            "instrument": self.cfg.instrument,
            "candles_processed": int(
                len(featured)
            ),
            "strategy": {
                "retrace_fraction": (
                    self.execution
                    .retrace_fraction
                ),
                "required_fill_session": (
                    self.execution
                    .required_session
                ),
                "stop_atr": (
                    self.cfg.stop_atr
                ),
                "target_r": (
                    self.cfg.target_r
                ),
                "max_pullback_bars": (
                    self.cfg.max_pullback_bars
                ),
                "max_holding_bars": (
                    self.cfg.max_holding_bars
                ),
            },
            "execution_assumptions": (
                asdict(self.execution)
            ),
            "orders": {
                "created": int(
                    len(created_df)
                ),
                "filled": int(
                    len(filled_df)
                ),
                "canceled_or_rejected": int(
                    len(canceled_df)
                ),
                "fill_rate": (
                    float(
                        len(filled_df)
                        / len(created_df)
                    )
                    if len(created_df)
                    else None
                ),
                "cancellation_reasons": (
                    cancellation_reasons
                ),
            },
            "performance_after_execution_costs": (
                performance
            ),
            "streaks": streaks,
            "causality": {
                "bar_by_bar_processing": True,
                "order_created_after_impulse_close": True,
                "same_impulse_candle_fill_allowed": False,
                "future_candles_used_for_signal": False,
                "fill_candle_used_for_exit": False,
                "signal_order_and_trade_management_separated": True,
            },
            "reports": {
                "created_orders": str(
                    created_path
                ),
                "canceled_orders": str(
                    canceled_path
                ),
                "filled_orders": str(
                    filled_path
                ),
                "trades": str(
                    trades_path
                ),
                "monthly": str(
                    monthly_path
                ),
                "yearly": str(
                    yearly_path
                ),
                "risk_sizing": str(
                    risk_path
                ),
                "summary": str(
                    summary_path
                ),
            },
            "warning": (
                "Bid and ask prices are approximated "
                "from mid-price OHLC candles. Intrabar "
                "path and actual broker queue priority "
                "are unknown."
            ),
        }

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                default=str,
            )
        )

        return summary
