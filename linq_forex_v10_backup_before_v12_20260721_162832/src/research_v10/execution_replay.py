from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import load_candles
from .features import add_market_features
from .signal_engine import CausalSignalEngine


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Realistic execution assumptions for mid-price OHLC data.

    The historical candle file is assumed to contain mid prices. Bid and ask
    are approximated by applying half the configured spread around the mid.
    """

    spread_pips: float = 0.8
    entry_slippage_pips: float = 0.2
    exit_slippage_pips: float = 0.2
    entry_delay_bars: int = 1

    max_open_trades: int = 1
    one_trade_per_impulse: bool = True
    one_trade_per_session: bool = False
    minimum_minutes_between_entries: int = 0

    reject_if_entry_gap_atr: float | None = 0.50
    minimum_stop_pips: float = 2.0

    account_starting_balance: float = 10_000.0
    risk_percentages: tuple[float, ...] = (
        0.25,
        0.50,
        1.00,
        2.00,
    )

    monte_carlo_runs: int = 2_000
    monte_carlo_seed: int = 42


@dataclass
class PendingOrder:
    signal: dict
    signal_index: int
    execution_index: int


@dataclass
class ExecutedTrade:
    setup_id: str
    instrument: str
    direction: str
    impulse_index: int
    signal_index: int
    entry_index: int
    impulse_timestamp: pd.Timestamp
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp

    raw_signal_entry_price: float
    mid_entry_price: float
    filled_entry_price: float
    stop_price: float
    target_price: float
    risk_price: float

    spread_pips: float
    entry_slippage_pips: float
    exit_slippage_pips: float
    entry_delay_bars: int
    entry_gap_atr: float

    session: str
    retrace_fraction: float

    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = values.cumsum()
    running_high = equity.cummax()
    drawdown = equity - running_high
    return float(drawdown.min())


def _profit_factor(values: pd.Series) -> float | None:
    wins = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))

    if losses == 0:
        return None if wins == 0 else float("inf")

    return wins / losses


def _performance_summary(trades: pd.DataFrame) -> dict:
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
        "max_drawdown_r": _max_drawdown(values),
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


def _streak_summary(values: pd.Series) -> dict:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    longest_win_streak = 0
    longest_loss_streak = 0
    current_wins = 0
    current_losses = 0

    for value in values:
        if value > 0:
            current_wins += 1
            current_losses = 0
            longest_win_streak = max(
                longest_win_streak,
                current_wins,
            )
        elif value < 0:
            current_losses += 1
            current_wins = 0
            longest_loss_streak = max(
                longest_loss_streak,
                current_losses,
            )
        else:
            current_wins = 0
            current_losses = 0

    return {
        "longest_win_streak": int(longest_win_streak),
        "longest_loss_streak": int(longest_loss_streak),
    }


class RealisticExecutionReplay:
    """
    Causal bar-by-bar signal replay with execution realism.

    Signal logic only sees completed candles. Orders are executed on a later
    candle according to entry_delay_bars. Mid-price OHLC data is converted
    into approximate bid/ask prices using the configured spread.
    """

    def __init__(
        self,
        research_config: ResearchConfig,
        execution_config: ExecutionConfig,
        retrace_threshold: float = 0.25,
        required_session: str = "london_open",
    ):
        self.cfg = research_config
        self.execution = execution_config
        self.retrace_threshold = float(retrace_threshold)
        self.required_session = required_session

        if self.execution.entry_delay_bars < 0:
            raise ValueError(
                "entry_delay_bars cannot be negative."
            )

        if self.execution.max_open_trades < 1:
            raise ValueError(
                "max_open_trades must be at least 1."
            )

        if self.execution.spread_pips < 0:
            raise ValueError(
                "spread_pips cannot be negative."
            )

    @property
    def half_spread_price(self) -> float:
        return (
            self.execution.spread_pips
            * self.cfg.pip_size
            / 2.0
        )

    @property
    def entry_slippage_price(self) -> float:
        return (
            self.execution.entry_slippage_pips
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
        return f"{timestamp.date().isoformat()}::{session}"

    def _entry_fill_price(
        self,
        direction: str,
        mid_price: float,
    ) -> float:
        if direction == "long":
            return (
                mid_price
                + self.half_spread_price
                + self.entry_slippage_price
            )

        return (
            mid_price
            - self.half_spread_price
            - self.entry_slippage_price
        )

    def _exit_fill_price(
        self,
        direction: str,
        mid_price: float,
    ) -> float:
        if direction == "long":
            return (
                mid_price
                - self.half_spread_price
                - self.exit_slippage_price
            )

        return (
            mid_price
            + self.half_spread_price
            + self.exit_slippage_price
        )

    def _tradable_bid_ask_extremes(
        self,
        candle: pd.Series,
    ) -> dict:
        mid_high = float(candle["high"])
        mid_low = float(candle["low"])

        return {
            "bid_high": mid_high - self.half_spread_price,
            "bid_low": mid_low - self.half_spread_price,
            "ask_high": mid_high + self.half_spread_price,
            "ask_low": mid_low + self.half_spread_price,
        }

    def _create_trade(
        self,
        order: PendingOrder,
        candle: pd.Series,
        index: int,
    ) -> tuple[ExecutedTrade | None, str | None]:
        signal = order.signal
        direction = str(signal["direction"])

        mid_entry_price = float(candle["open"])
        filled_entry_price = self._entry_fill_price(
            direction=direction,
            mid_price=mid_entry_price,
        )

        atr = float(signal["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None, "invalid_signal_atr"

        risk_price = self.cfg.stop_atr * atr
        minimum_risk = (
            self.execution.minimum_stop_pips
            * self.cfg.pip_size
        )
        risk_price = max(risk_price, minimum_risk)

        raw_signal_price = float(signal["entry_price"])
        entry_gap_atr = (
            abs(mid_entry_price - raw_signal_price) / atr
        )

        gap_limit = self.execution.reject_if_entry_gap_atr

        if (
            gap_limit is not None
            and entry_gap_atr > gap_limit
        ):
            return None, "entry_gap_exceeded"

        if direction == "long":
            stop_price = filled_entry_price - risk_price
            target_price = (
                filled_entry_price
                + self.cfg.target_r * risk_price
            )
        else:
            stop_price = filled_entry_price + risk_price
            target_price = (
                filled_entry_price
                - self.cfg.target_r * risk_price
            )

        trade = ExecutedTrade(
            setup_id=str(signal["setup_id"]),
            instrument=str(signal["instrument"]),
            direction=direction,
            impulse_index=int(signal["impulse_index"]),
            signal_index=int(signal["entry_index"]),
            entry_index=index,
            impulse_timestamp=pd.Timestamp(
                signal["impulse_timestamp"]
            ),
            signal_timestamp=pd.Timestamp(
                signal["timestamp"]
            ),
            entry_timestamp=pd.Timestamp(
                candle["timestamp"]
            ),
            raw_signal_entry_price=raw_signal_price,
            mid_entry_price=mid_entry_price,
            filled_entry_price=filled_entry_price,
            stop_price=float(stop_price),
            target_price=float(target_price),
            risk_price=float(risk_price),
            spread_pips=self.execution.spread_pips,
            entry_slippage_pips=(
                self.execution.entry_slippage_pips
            ),
            exit_slippage_pips=(
                self.execution.exit_slippage_pips
            ),
            entry_delay_bars=(
                index - int(signal["entry_index"])
            ),
            entry_gap_atr=float(entry_gap_atr),
            session=str(signal["session"]),
            retrace_fraction=float(
                signal["retrace_fraction"]
            ),
        )

        return trade, None

    def _update_trade(
        self,
        trade: ExecutedTrade,
        candle: pd.Series,
    ) -> dict | None:
        extremes = self._tradable_bid_ask_extremes(candle)

        trade.bars_held += 1

        if trade.direction == "long":
            favorable_price = extremes["bid_high"]
            adverse_price = extremes["bid_low"]

            favorable_r = (
                favorable_price - trade.filled_entry_price
            ) / trade.risk_price

            adverse_r = (
                adverse_price - trade.filled_entry_price
            ) / trade.risk_price

            stop_hit = (
                extremes["bid_low"] <= trade.stop_price
            )
            target_hit = (
                extremes["bid_high"] >= trade.target_price
            )
        else:
            favorable_price = extremes["ask_low"]
            adverse_price = extremes["ask_high"]

            favorable_r = (
                trade.filled_entry_price - favorable_price
            ) / trade.risk_price

            adverse_r = (
                trade.filled_entry_price - adverse_price
            ) / trade.risk_price

            stop_hit = (
                extremes["ask_high"] >= trade.stop_price
            )
            target_hit = (
                extremes["ask_low"] <= trade.target_price
            )

        trade.mfe_r = max(
            trade.mfe_r,
            float(favorable_r),
        )
        trade.mae_r = min(
            trade.mae_r,
            float(adverse_r),
        )

        gross_r = None
        exit_reason = None
        mid_exit_price = None

        if stop_hit and target_hit:
            if self.cfg.intrabar_policy == "target_first":
                gross_r = self.cfg.target_r
                exit_reason = "target"
                mid_exit_price = trade.target_price
            else:
                gross_r = -1.0
                exit_reason = "stop"
                mid_exit_price = trade.stop_price

        elif stop_hit:
            gross_r = -1.0
            exit_reason = "stop"
            mid_exit_price = trade.stop_price

        elif target_hit:
            gross_r = self.cfg.target_r
            exit_reason = "target"
            mid_exit_price = trade.target_price

        elif trade.bars_held >= self.cfg.max_holding_bars:
            mid_exit_price = float(candle["close"])
            filled_exit = self._exit_fill_price(
                direction=trade.direction,
                mid_price=mid_exit_price,
            )

            if trade.direction == "long":
                gross_r = (
                    filled_exit - trade.filled_entry_price
                ) / trade.risk_price
            else:
                gross_r = (
                    trade.filled_entry_price - filled_exit
                ) / trade.risk_price

            exit_reason = "time"

        if gross_r is None:
            return None

        if exit_reason in {"stop", "target"}:
            # Stop and target levels are executable-side levels. Add modeled
            # adverse exit slippage after the trigger.
            slippage_r = (
                self.exit_slippage_price
                / trade.risk_price
            )

            if exit_reason == "target":
                net_r = float(gross_r - slippage_r)
            else:
                net_r = float(gross_r - slippage_r)

            filled_exit_price = (
                trade.target_price
                - self.exit_slippage_price
                if trade.direction == "long"
                and exit_reason == "target"
                else trade.target_price
                + self.exit_slippage_price
                if trade.direction == "short"
                and exit_reason == "target"
                else trade.stop_price
                - self.exit_slippage_price
                if trade.direction == "long"
                else trade.stop_price
                + self.exit_slippage_price
            )
        else:
            filled_exit_price = self._exit_fill_price(
                direction=trade.direction,
                mid_price=float(mid_exit_price),
            )

            if trade.direction == "long":
                net_r = (
                    filled_exit_price
                    - trade.filled_entry_price
                ) / trade.risk_price
            else:
                net_r = (
                    trade.filled_entry_price
                    - filled_exit_price
                ) / trade.risk_price

            gross_r = net_r

        return {
            **asdict(trade),
            "exit_timestamp": pd.Timestamp(
                candle["timestamp"]
            ),
            "mid_exit_price": float(mid_exit_price),
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
            "hit_1r": bool(trade.mfe_r >= 1.0),
            "hit_target": bool(
                trade.mfe_r >= self.cfg.target_r
            ),
        }

    def _period_performance(
        self,
        trades: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        out = trades.copy()
        out["entry_timestamp"] = pd.to_datetime(
            out["entry_timestamp"],
            utc=True,
        )

        if frequency == "month":
            out["period"] = (
                out["entry_timestamp"]
                .dt.tz_localize(None)
                .dt.to_period("M")
                .astype(str)
            )
        elif frequency == "year":
            out["period"] = (
                out["entry_timestamp"]
                .dt.year
                .astype(str)
            )
        else:
            raise ValueError(
                f"Unsupported frequency: {frequency}"
            )

        rows = []

        for period, group in out.groupby(
            "period",
            sort=True,
        ):
            summary = _performance_summary(group)
            rows.append(
                {
                    "period": period,
                    **summary,
                }
            )

        return pd.DataFrame(rows)

    def _risk_simulations(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        r_values = pd.to_numeric(
            trades["net_r"],
            errors="coerce",
        ).dropna().to_numpy()

        rows = []

        for risk_pct in self.execution.risk_percentages:
            balance = (
                self.execution.account_starting_balance
            )
            peak = balance
            max_drawdown_pct = 0.0

            for r_value in r_values:
                amount_at_risk = (
                    balance * risk_pct / 100.0
                )
                balance += amount_at_risk * r_value

                peak = max(peak, balance)

                if peak > 0:
                    drawdown_pct = (
                        balance - peak
                    ) / peak * 100.0
                    max_drawdown_pct = min(
                        max_drawdown_pct,
                        drawdown_pct,
                    )

            rows.append(
                {
                    "risk_percent_per_trade": risk_pct,
                    "starting_balance": (
                        self.execution.account_starting_balance
                    ),
                    "ending_balance": float(balance),
                    "net_profit": float(
                        balance
                        - self.execution.account_starting_balance
                    ),
                    "return_percent": float(
                        (
                            balance
                            / self.execution.account_starting_balance
                            - 1
                        )
                        * 100.0
                    ),
                    "max_drawdown_percent": float(
                        max_drawdown_pct
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _monte_carlo(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        values = pd.to_numeric(
            trades["net_r"],
            errors="coerce",
        ).dropna().to_numpy()

        if len(values) == 0:
            return pd.DataFrame()

        rng = np.random.default_rng(
            self.execution.monte_carlo_seed
        )

        risk_pct = 1.0
        runs = []

        for run in range(
            self.execution.monte_carlo_runs
        ):
            shuffled = rng.permutation(values)

            balance = (
                self.execution.account_starting_balance
            )
            peak = balance
            max_dd_pct = 0.0
            longest_loss_streak = 0
            current_loss_streak = 0

            for result_r in shuffled:
                risk_amount = (
                    balance * risk_pct / 100.0
                )
                balance += risk_amount * result_r

                peak = max(peak, balance)

                if peak > 0:
                    drawdown_pct = (
                        balance - peak
                    ) / peak * 100.0
                    max_dd_pct = min(
                        max_dd_pct,
                        drawdown_pct,
                    )

                if result_r < 0:
                    current_loss_streak += 1
                    longest_loss_streak = max(
                        longest_loss_streak,
                        current_loss_streak,
                    )
                else:
                    current_loss_streak = 0

            runs.append(
                {
                    "run": run + 1,
                    "risk_percent_per_trade": risk_pct,
                    "ending_balance": float(balance),
                    "return_percent": float(
                        (
                            balance
                            / self.execution.account_starting_balance
                            - 1
                        )
                        * 100.0
                    ),
                    "max_drawdown_percent": float(
                        max_dd_pct
                    ),
                    "longest_loss_streak": int(
                        longest_loss_streak
                    ),
                }
            )

        return pd.DataFrame(runs)

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

        detector = CausalSignalEngine(
            config=self.cfg,
            retrace_threshold=self.retrace_threshold,
            required_session=self.required_session,
        )

        pending_orders: list[PendingOrder] = []
        open_trades: list[ExecutedTrade] = []

        signal_rows: list[dict] = []
        rejected_order_rows: list[dict] = []
        completed_rows: list[dict] = []

        traded_impulses: set[int] = set()
        traded_sessions: set[str] = set()
        last_entry_timestamp: pd.Timestamp | None = None

        warmup = max(
            self.cfg.ema_slow + 12,
            250,
        )

        for index in range(len(featured)):
            candle = featured.iloc[index]
            current_timestamp = pd.Timestamp(
                candle["timestamp"]
            )

            remaining_trades = []

            for trade in open_trades:
                # Never evaluate execution on its entry candle.
                if index <= trade.entry_index:
                    remaining_trades.append(trade)
                    continue

                completed = self._update_trade(
                    trade=trade,
                    candle=candle,
                )

                if completed is None:
                    remaining_trades.append(trade)
                else:
                    completed_rows.append(completed)

            open_trades = remaining_trades

            remaining_orders = []

            for order in pending_orders:
                if index < order.execution_index:
                    remaining_orders.append(order)
                    continue

                signal = order.signal
                reject_reason = None

                if (
                    len(open_trades)
                    >= self.execution.max_open_trades
                ):
                    reject_reason = "max_open_trades"

                impulse_index = int(
                    signal["impulse_index"]
                )

                if (
                    reject_reason is None
                    and self.execution.one_trade_per_impulse
                    and impulse_index in traded_impulses
                ):
                    reject_reason = (
                        "impulse_already_traded"
                    )

                session_key = self._session_key(
                    timestamp=pd.Timestamp(
                        signal["timestamp"]
                    ),
                    session=str(signal["session"]),
                )

                if (
                    reject_reason is None
                    and self.execution.one_trade_per_session
                    and session_key in traded_sessions
                ):
                    reject_reason = (
                        "session_already_traded"
                    )

                if (
                    reject_reason is None
                    and last_entry_timestamp is not None
                    and (
                        current_timestamp
                        - last_entry_timestamp
                    ).total_seconds()
                    < (
                        self.execution
                        .minimum_minutes_between_entries
                        * 60
                    )
                ):
                    reject_reason = (
                        "minimum_entry_spacing"
                    )

                if reject_reason is not None:
                    rejected_order_rows.append(
                        {
                            **signal,
                            "execution_timestamp": (
                                current_timestamp
                            ),
                            "execution_index": index,
                            "order_rejection_reason": (
                                reject_reason
                            ),
                        }
                    )
                    continue

                trade, creation_error = (
                    self._create_trade(
                        order=order,
                        candle=candle,
                        index=index,
                    )
                )

                if trade is None:
                    rejected_order_rows.append(
                        {
                            **signal,
                            "execution_timestamp": (
                                current_timestamp
                            ),
                            "execution_index": index,
                            "order_rejection_reason": (
                                creation_error
                                or "execution_rejected"
                            ),
                        }
                    )
                    continue

                open_trades.append(trade)
                traded_impulses.add(
                    trade.impulse_index
                )
                traded_sessions.add(session_key)
                last_entry_timestamp = (
                    trade.entry_timestamp
                )

            pending_orders = remaining_orders

            if index < warmup:
                continue

            signals = detector.process_bar(
                index=index,
                candle=candle,
            )

            for signal in signals:
                signal_rows.append(signal)

                if not signal["accepted"]:
                    continue

                execution_index = (
                    index
                    + self.execution.entry_delay_bars
                )

                if execution_index >= len(featured):
                    rejected_order_rows.append(
                        {
                            **signal,
                            "execution_timestamp": None,
                            "execution_index": (
                                execution_index
                            ),
                            "order_rejection_reason": (
                                "insufficient_future_data"
                            ),
                        }
                    )
                    continue

                pending_orders.append(
                    PendingOrder(
                        signal=signal,
                        signal_index=index,
                        execution_index=execution_index,
                    )
                )

        if len(featured) > 0:
            final_candle = featured.iloc[-1]

            for trade in open_trades:
                mid_exit = float(
                    final_candle["close"]
                )
                filled_exit = self._exit_fill_price(
                    trade.direction,
                    mid_exit,
                )

                if trade.direction == "long":
                    net_r = (
                        filled_exit
                        - trade.filled_entry_price
                    ) / trade.risk_price
                else:
                    net_r = (
                        trade.filled_entry_price
                        - filled_exit
                    ) / trade.risk_price

                completed_rows.append(
                    {
                        **asdict(trade),
                        "exit_timestamp": pd.Timestamp(
                            final_candle["timestamp"]
                        ),
                        "mid_exit_price": mid_exit,
                        "filled_exit_price": filled_exit,
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
                        "exit_reason": "end_of_data",
                        "hit_1r": bool(
                            trade.mfe_r >= 1.0
                        ),
                        "hit_target": bool(
                            trade.mfe_r
                            >= self.cfg.target_r
                        ),
                    }
                )

        signals_df = pd.DataFrame(signal_rows)
        rejected_df = pd.DataFrame(
            rejected_order_rows
        )
        trades_df = pd.DataFrame(completed_rows)

        if not trades_df.empty:
            trades_df = trades_df.sort_values(
                "entry_timestamp"
            ).reset_index(drop=True)

        monthly_df = self._period_performance(
            trades_df,
            "month",
        )
        yearly_df = self._period_performance(
            trades_df,
            "year",
        )
        risk_df = self._risk_simulations(
            trades_df
        )
        monte_carlo_df = self._monte_carlo(
            trades_df
        )

        performance = _performance_summary(
            trades_df
        )

        streaks = (
            _streak_summary(trades_df["net_r"])
            if not trades_df.empty
            else {
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
            }
        )

        monte_carlo_summary = None

        if not monte_carlo_df.empty:
            monte_carlo_summary = {
                "runs": int(
                    len(monte_carlo_df)
                ),
                "risk_percent_per_trade": 1.0,
                "median_return_percent": float(
                    monte_carlo_df[
                        "return_percent"
                    ].median()
                ),
                "fifth_percentile_return_percent": (
                    float(
                        monte_carlo_df[
                            "return_percent"
                        ].quantile(0.05)
                    )
                ),
                "median_max_drawdown_percent": (
                    float(
                        monte_carlo_df[
                            "max_drawdown_percent"
                        ].median()
                    )
                ),
                "fifth_percentile_max_drawdown_percent": (
                    float(
                        monte_carlo_df[
                            "max_drawdown_percent"
                        ].quantile(0.05)
                    )
                ),
                "ninety_fifth_percentile_loss_streak": (
                    float(
                        monte_carlo_df[
                            "longest_loss_streak"
                        ].quantile(0.95)
                    )
                ),
            }

        prefix = (
            f"{self.cfg.instrument}_v11_2_execution"
        )

        signals_path = (
            report_dir / f"{prefix}_signals.csv"
        )
        rejected_path = (
            report_dir
            / f"{prefix}_rejected_orders.csv"
        )
        trades_path = (
            report_dir / f"{prefix}_trades.csv"
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
        monte_carlo_path = (
            report_dir
            / f"{prefix}_monte_carlo.csv"
        )
        summary_path = (
            report_dir
            / f"{prefix}_summary.json"
        )

        signals_df.to_csv(
            signals_path,
            index=False,
        )
        rejected_df.to_csv(
            rejected_path,
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
        monte_carlo_df.to_csv(
            monte_carlo_path,
            index=False,
        )

        accepted_signals = (
            int(signals_df["accepted"].sum())
            if (
                not signals_df.empty
                and "accepted" in signals_df.columns
            )
            else 0
        )

        summary = {
            "engine_version": (
                "11.2-execution-realism"
            ),
            "instrument": self.cfg.instrument,
            "candles_processed": int(
                len(featured)
            ),
            "locked_rule": (
                f"retrace_fraction le "
                f"{self.retrace_threshold} "
                f"AND session eq "
                f"{self.required_session}"
            ),
            "execution_assumptions": asdict(
                self.execution
            ),
            "signals": {
                "all_signals": int(
                    len(signals_df)
                ),
                "accepted_by_strategy": (
                    accepted_signals
                ),
                "orders_rejected": int(
                    len(rejected_df)
                ),
                "trades_executed": int(
                    len(trades_df)
                ),
            },
            "performance_after_execution_costs": (
                performance
            ),
            "streaks": streaks,
            "monte_carlo_summary": (
                monte_carlo_summary
            ),
            "causality": {
                "bar_by_bar_processing": True,
                "future_candles_used_for_signal": False,
                "entry_delay_applied": True,
                "entry_candle_used_for_exit": False,
                "signal_and_execution_separated": True,
            },
            "reports": {
                "signals": str(signals_path),
                "rejected_orders": str(
                    rejected_path
                ),
                "trades": str(trades_path),
                "monthly": str(monthly_path),
                "yearly": str(yearly_path),
                "risk_sizing": str(risk_path),
                "monte_carlo": str(
                    monte_carlo_path
                ),
                "summary": str(summary_path),
            },
            "warning": (
                "Bid and ask prices are approximated from "
                "mid-price candles. Live spreads, slippage, "
                "latency and broker fills can be worse."
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
