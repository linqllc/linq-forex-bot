from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd


class LinqSignalReplayStrategy(bt.Strategy):
    params = (
        ("signals", None),
        ("fixed_units", 10_000),
        ("pip_size", 0.0001),
        ("slippage_pips_each_side", 0.10),
        ("maximum_open_positions", 1),
    )

    def __init__(self):
        if self.p.signals is None:
            raise ValueError("signals parameter is required")

        self.signal_map = {}

        for _, signal in self.p.signals.iterrows():
            timestamp = pd.Timestamp(signal["timestamp"])

            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC").tz_localize(None)

            self.signal_map[timestamp.to_pydatetime()] = signal.to_dict()

        self.pending_orders = set()
        self.order_roles = {}
        self.active_trade = None

        self.trade_log = []
        self.signal_log = []
        self.skipped_signals = []

    def _current_timestamp(self):
        return pd.Timestamp(
            self.data.datetime.datetime(0)
        ).to_pydatetime()

    def _trade_is_active(self):
        return bool(
            self.position.size != 0
            or self.active_trade is not None
        )

    def next(self):
        timestamp = self._current_timestamp()
        signal = self.signal_map.get(timestamp)

        if signal is None:
            return

        if self._trade_is_active():
            self.skipped_signals.append(
                {
                    "timestamp": timestamp,
                    "direction": signal.get("direction"),
                    "reason": "position_or_bracket_already_active",
                }
            )
            return

        direction = str(signal["direction"]).lower()
        stop_price = float(signal["stop_price"])
        target_price = float(signal["target_price"])
        reference_entry = float(signal["entry_price"])
        current_close = float(self.data.close[0])

        if direction == "long":
            valid_geometry = (
                stop_price < current_close < target_price
            )
        else:
            valid_geometry = (
                target_price < current_close < stop_price
            )

        if not valid_geometry:
            self.skipped_signals.append(
                {
                    "timestamp": timestamp,
                    "direction": direction,
                    "reason": "invalid_bracket_geometry_at_replay_bar",
                    "current_close": current_close,
                    "reference_entry": reference_entry,
                    "stop_price": stop_price,
                    "target_price": target_price,
                }
            )
            return

        size = int(self.p.fixed_units)

        if direction == "long":
            orders = self.buy_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                stopexec=bt.Order.Stop,
                limitprice=target_price,
                limitexec=bt.Order.Limit,
            )
        else:
            orders = self.sell_bracket(
                size=size,
                exectype=bt.Order.Market,
                stopprice=stop_price,
                stopexec=bt.Order.Stop,
                limitprice=target_price,
                limitexec=bt.Order.Limit,
            )

        parent, stop_order, target_order = orders

        self.order_roles[parent.ref] = "entry"
        self.order_roles[stop_order.ref] = "stop"
        self.order_roles[target_order.ref] = "target"

        for order in orders:
            self.pending_orders.add(order.ref)

        self.active_trade = {
            "signal": signal,
            "entry_order_ref": parent.ref,
            "stop_order_ref": stop_order.ref,
            "target_order_ref": target_order.ref,
            "entry_fill_price": None,
            "entry_fill_time": None,
            "exit_fill_price": None,
            "exit_fill_time": None,
            "exit_role": None,
        }

        self.signal_log.append(
            {
                "timestamp": timestamp,
                "direction": direction,
                "reference_entry": reference_entry,
                "replay_close": current_close,
                "stop_price": stop_price,
                "target_price": target_price,
                "spread_pips_at_entry": signal.get(
                    "spread_pips_at_entry"
                ),
                "probability_1r": signal.get("probability_1r"),
                "order_status": "submitted",
            }
        )

    def notify_order(self, order):
        terminal_statuses = {
            order.Completed,
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        }

        if order.status not in terminal_statuses:
            return

        self.pending_orders.discard(order.ref)
        role = self.order_roles.get(order.ref, "unknown")

        if order.status == order.Completed and self.active_trade:
            fill_time = bt.num2date(
                order.executed.dt
            ).replace(tzinfo=None)

            fill_price = float(order.executed.price)

            if role == "entry":
                self.active_trade["entry_fill_price"] = fill_price
                self.active_trade["entry_fill_time"] = fill_time

            elif role in {"stop", "target"}:
                self.active_trade["exit_fill_price"] = fill_price
                self.active_trade["exit_fill_time"] = fill_time
                self.active_trade["exit_role"] = role

        if order.status in {
            order.Margin,
            order.Rejected,
            order.Expired,
        }:
            self.skipped_signals.append(
                {
                    "timestamp": self._current_timestamp(),
                    "reason": (
                        f"order_{order.getstatusname().lower()}"
                    ),
                    "order_role": role,
                }
            )

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        active = self.active_trade or {}
        signal = active.get("signal", {})

        direction = str(
            signal.get("direction", "")
        ).lower()

        entry_price = active.get("entry_fill_price")
        exit_price = active.get("exit_fill_price")

        stop_price = pd.to_numeric(
            signal.get("stop_price"),
            errors="coerce",
        )
        target_price = pd.to_numeric(
            signal.get("target_price"),
            errors="coerce",
        )
        reference_entry = pd.to_numeric(
            signal.get("entry_price"),
            errors="coerce",
        )
        spread_pips = pd.to_numeric(
            signal.get("spread_pips_at_entry"),
            errors="coerce",
        )

        risk_price = np.nan
        realized_r_before_cost = np.nan

        if (
            entry_price is not None
            and np.isfinite(stop_price)
        ):
            risk_price = abs(
                float(entry_price) - float(stop_price)
            )

        if (
            entry_price is not None
            and exit_price is not None
            and np.isfinite(risk_price)
            and risk_price > 0
        ):
            signed_move = (
                float(exit_price) - float(entry_price)
                if direction == "long"
                else float(entry_price) - float(exit_price)
            )
            realized_r_before_cost = (
                signed_move / risk_price
            )

        if (
            np.isfinite(spread_pips)
            and np.isfinite(risk_price)
            and risk_price > 0
        ):
            spread_cost_r = (
                float(spread_pips)
                * self.p.pip_size
                / risk_price
            )
        else:
            spread_cost_r = 0.0

        if np.isfinite(risk_price) and risk_price > 0:
            slippage_cost_r = (
                2.0
                * self.p.slippage_pips_each_side
                * self.p.pip_size
                / risk_price
            )
        else:
            slippage_cost_r = 0.0

        total_execution_cost_r = (
            spread_cost_r + slippage_cost_r
        )

        if np.isfinite(realized_r_before_cost):
            realized_r_after_all_costs = (
                realized_r_before_cost
                - total_execution_cost_r
            )
        else:
            realized_r_after_all_costs = np.nan

        self.trade_log.append(
            {
                "signal_timestamp": signal.get("timestamp"),
                "direction": direction,
                "probability_1r": signal.get("probability_1r"),
                "reference_entry_price": reference_entry,
                "entry_fill_time": active.get("entry_fill_time"),
                "entry_fill_price": entry_price,
                "entry_difference_pips": (
                    (
                        float(entry_price)
                        - float(reference_entry)
                    )
                    / self.p.pip_size
                    if (
                        entry_price is not None
                        and np.isfinite(reference_entry)
                    )
                    else np.nan
                ),
                "exit_fill_time": active.get("exit_fill_time"),
                "exit_fill_price": exit_price,
                "exit_reason": active.get("exit_role"),
                "stop_price": stop_price,
                "target_price": target_price,
                "risk_price": risk_price,
                "risk_pips": (
                    risk_price / self.p.pip_size
                    if np.isfinite(risk_price)
                    else np.nan
                ),
                "spread_pips_at_entry": spread_pips,
                "spread_cost_r": spread_cost_r,
                "slippage_cost_r": slippage_cost_r,
                "total_execution_cost_r": (
                    total_execution_cost_r
                ),
                "realized_r_before_cost": (
                    realized_r_before_cost
                ),
                "realized_r_after_all_costs": (
                    realized_r_after_all_costs
                ),
                "phase4_gross_result_r": signal.get(
                    "gross_result_r"
                ),
                "phase4_net_result_r": signal.get(
                    "net_result_r"
                ),
                "backtrader_gross_pnl": float(trade.pnl),
                "backtrader_net_pnl": float(trade.pnlcomm),
                "bars_held": int(trade.barlen),
            }
        )

        self.active_trade = None
