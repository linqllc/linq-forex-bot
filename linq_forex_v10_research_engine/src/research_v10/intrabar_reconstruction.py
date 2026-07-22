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
class IntrabarScenario:
    name: str
    entry_mode: str

    retrace_fraction: float = 0.25
    max_retrace_fraction: float = 0.25

    spread_pips: float = 0.8
    entry_slippage_pips: float = 0.2
    exit_slippage_pips: float = 0.2

    max_wait_m1_bars: int = 120
    max_holding_m1_bars: int = 480

    minimum_stop_pips: float = 2.0


@dataclass
class M5Impulse:
    setup_id: str
    direction: str

    m5_index: int
    m5_timestamp: pd.Timestamp
    available_from: pd.Timestamp

    impulse_open: float
    impulse_high: float
    impulse_low: float
    impulse_close: float
    impulse_range: float

    atr: float
    session: str


@dataclass
class OpenTrade:
    scenario: str
    setup_id: str
    direction: str

    impulse_timestamp: pd.Timestamp
    available_from: pd.Timestamp

    entry_timestamp: pd.Timestamp
    entry_m1_index: int

    raw_entry_price: float
    filled_entry_price: float

    risk_price: float
    stop_price: float
    target_price: float

    retrace_fraction_observed: float
    spread_pips: float
    entry_slippage_pips: float
    exit_slippage_pips: float

    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def _profit_factor(values: pd.Series) -> float | None:
    gross_profit = float(values[values > 0].sum())
    gross_loss = abs(float(values[values < 0].sum()))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else None

    return gross_profit / gross_loss


def _max_drawdown_r(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = values.cumsum()
    drawdown = equity - equity.cummax()

    return float(drawdown.min())


def _longest_streak(
    values: pd.Series,
    winning: bool,
) -> int:
    longest = 0
    current = 0

    for value in values:
        matched = value > 0 if winning else value < 0

        if matched:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


def performance_summary(
    trades: pd.DataFrame,
) -> dict:
    if trades.empty:
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
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
        }

    values = pd.to_numeric(
        trades["net_r"],
        errors="coerce",
    ).dropna()

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
        "longest_win_streak": _longest_streak(
            values,
            winning=True,
        ),
        "longest_loss_streak": _longest_streak(
            values,
            winning=False,
        ),
    }


class CausalIntrabarReconstruction:
    """
    M5 impulse detection with future-only M1 execution.

    Assumptions:
    - M5 and M1 timestamps represent candle-open times.
    - An M5 impulse stamped 06:00 becomes available at 06:05.
    - Only M1 candles stamped 06:05 or later may be used.
    - Limit fills and exits use approximated bid/ask prices.
    - If stop and target are touched in the same M1 candle,
      stop-first is used unless config says target-first.
    """

    def __init__(
        self,
        config: ResearchConfig,
        required_session: str = "london_open",
        timestamps_are_bar_open: bool = True,
    ):
        self.cfg = config
        self.required_session = required_session
        self.timestamps_are_bar_open = (
            timestamps_are_bar_open
        )

    def _pips_to_price(
        self,
        pips: float,
    ) -> float:
        return float(pips * self.cfg.pip_size)

    def _available_from(
        self,
        timestamp: pd.Timestamp,
        minutes: int,
    ) -> pd.Timestamp:
        timestamp = pd.Timestamp(timestamp)

        if not self.timestamps_are_bar_open:
            return timestamp

        return timestamp + pd.Timedelta(
            minutes=minutes
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
            and str(candle["session"])
            == self.required_session
        )

    def _build_impulses(
        self,
        featured_m5: pd.DataFrame,
    ) -> list[M5Impulse]:
        impulses: list[M5Impulse] = []

        warmup = max(
            self.cfg.ema_slow + 12,
            250,
        )

        for index in range(
            warmup,
            len(featured_m5),
        ):
            candle = featured_m5.iloc[index]

            if not self._is_displacement(candle):
                continue

            open_price = float(candle["open"])
            close_price = float(candle["close"])
            high = float(candle["high"])
            low = float(candle["low"])

            impulse_range = high - low

            if impulse_range <= 0:
                continue

            direction = (
                "long"
                if close_price > open_price
                else "short"
            )

            timestamp = pd.Timestamp(
                candle["timestamp"]
            )

            impulses.append(
                M5Impulse(
                    setup_id=(
                        f"{self.cfg.instrument}-"
                        f"{direction}-"
                        f"{timestamp.isoformat()}"
                    ),
                    direction=direction,
                    m5_index=index,
                    m5_timestamp=timestamp,
                    available_from=self._available_from(
                        timestamp,
                        minutes=5,
                    ),
                    impulse_open=open_price,
                    impulse_high=high,
                    impulse_low=low,
                    impulse_close=close_price,
                    impulse_range=impulse_range,
                    atr=float(candle["atr"]),
                    session=str(candle["session"]),
                )
            )

        return impulses

    def _m1_window(
        self,
        impulse: M5Impulse,
        m1: pd.DataFrame,
        max_wait_bars: int,
    ) -> pd.DataFrame:
        timestamps = pd.to_datetime(
            m1["timestamp"],
            utc=True,
        )

        start_position = int(
            timestamps.searchsorted(
                impulse.available_from,
                side="left",
            )
        )

        end_position = min(
            start_position + max_wait_bars,
            len(m1),
        )

        return m1.iloc[
            start_position:end_position
        ].copy()

    def _retrace_fraction(
        self,
        impulse: M5Impulse,
        price: float,
    ) -> float:
        if impulse.direction == "long":
            return (
                impulse.impulse_high - price
            ) / impulse.impulse_range

        return (
            price - impulse.impulse_low
        ) / impulse.impulse_range

    def _invalidated(
        self,
        impulse: M5Impulse,
        candle: pd.Series,
    ) -> bool:
        if impulse.direction == "long":
            return (
                float(candle["low"])
                < impulse.impulse_low
            )

        return (
            float(candle["high"])
            > impulse.impulse_high
        )

    def _limit_entry(
        self,
        impulse: M5Impulse,
        window: pd.DataFrame,
        scenario: IntrabarScenario,
    ) -> dict | None:
        half_spread = (
            self._pips_to_price(
                scenario.spread_pips
            )
            / 2.0
        )

        if impulse.direction == "long":
            raw_limit = (
                impulse.impulse_high
                - scenario.retrace_fraction
                * impulse.impulse_range
            )

            executable_limit = (
                raw_limit + half_spread
            )
        else:
            raw_limit = (
                impulse.impulse_low
                + scenario.retrace_fraction
                * impulse.impulse_range
            )

            executable_limit = (
                raw_limit - half_spread
            )

        for index, candle in window.iterrows():
            if self._invalidated(
                impulse,
                candle,
            ):
                return None

            if impulse.direction == "long":
                ask_low = (
                    float(candle["low"])
                    + half_spread
                )

                touched = (
                    ask_low <= executable_limit
                )
            else:
                bid_high = (
                    float(candle["high"])
                    - half_spread
                )

                touched = (
                    bid_high >= executable_limit
                )

            if not touched:
                continue

            adverse_slippage = self._pips_to_price(
                scenario.entry_slippage_pips
            )

            if impulse.direction == "long":
                filled_entry = (
                    executable_limit
                    + adverse_slippage
                )
            else:
                filled_entry = (
                    executable_limit
                    - adverse_slippage
                )

            return {
                "entry_m1_index": int(index),
                "entry_timestamp": pd.Timestamp(
                    candle["timestamp"]
                ),
                "raw_entry_price": float(
                    raw_limit
                ),
                "filled_entry_price": float(
                    filled_entry
                ),
                "retrace_fraction_observed": float(
                    scenario.retrace_fraction
                ),
                "outcome_start_index": int(index),
                "entry_fill_mode": "intrabar_limit",
            }

        return None

    def _reversal_entry(
        self,
        impulse: M5Impulse,
        window: pd.DataFrame,
        scenario: IntrabarScenario,
    ) -> dict | None:
        pullback_seen = False
        deepest_retrace = 0.0

        half_spread = (
            self._pips_to_price(
                scenario.spread_pips
            )
            / 2.0
        )

        for index, candle in window.iterrows():
            if self._invalidated(
                impulse,
                candle,
            ):
                return None

            candle_open = float(candle["open"])
            candle_close = float(candle["close"])

            if impulse.direction == "long":
                observed_price = float(
                    candle["low"]
                )
            else:
                observed_price = float(
                    candle["high"]
                )

            retrace = self._retrace_fraction(
                impulse,
                observed_price,
            )

            if retrace > 0:
                pullback_seen = True
                deepest_retrace = max(
                    deepest_retrace,
                    retrace,
                )

            if (
                deepest_retrace
                > scenario.max_retrace_fraction
            ):
                return None

            if not pullback_seen:
                continue

            resumed = (
                candle_close > candle_open
                if impulse.direction == "long"
                else candle_close < candle_open
            )

            if not resumed:
                continue

            adverse_cost = (
                half_spread
                + self._pips_to_price(
                    scenario.entry_slippage_pips
                )
            )

            if impulse.direction == "long":
                filled_entry = (
                    candle_close + adverse_cost
                )
            else:
                filled_entry = (
                    candle_close - adverse_cost
                )

            return {
                "entry_m1_index": int(index),
                "entry_timestamp": pd.Timestamp(
                    candle["timestamp"]
                ),
                "raw_entry_price": candle_close,
                "filled_entry_price": float(
                    filled_entry
                ),
                "retrace_fraction_observed": float(
                    deepest_retrace
                ),
                "outcome_start_index": int(index) + 1,
                "entry_fill_mode": "m1_close",
            }

        return None

    def _create_trade(
        self,
        impulse: M5Impulse,
        entry: dict,
        scenario: IntrabarScenario,
    ) -> OpenTrade:
        minimum_risk = (
            scenario.minimum_stop_pips
            * self.cfg.pip_size
        )

        risk_price = max(
            self.cfg.stop_atr * impulse.atr,
            minimum_risk,
        )

        filled_entry = float(
            entry["filled_entry_price"]
        )

        if impulse.direction == "long":
            stop_price = (
                filled_entry - risk_price
            )

            target_price = (
                filled_entry
                + self.cfg.target_r
                * risk_price
            )
        else:
            stop_price = (
                filled_entry + risk_price
            )

            target_price = (
                filled_entry
                - self.cfg.target_r
                * risk_price
            )

        return OpenTrade(
            scenario=scenario.name,
            setup_id=impulse.setup_id,
            direction=impulse.direction,
            impulse_timestamp=(
                impulse.m5_timestamp
            ),
            available_from=impulse.available_from,
            entry_timestamp=entry[
                "entry_timestamp"
            ],
            entry_m1_index=entry[
                "entry_m1_index"
            ],
            raw_entry_price=entry[
                "raw_entry_price"
            ],
            filled_entry_price=filled_entry,
            risk_price=float(risk_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            retrace_fraction_observed=entry[
                "retrace_fraction_observed"
            ],
            spread_pips=scenario.spread_pips,
            entry_slippage_pips=(
                scenario.entry_slippage_pips
            ),
            exit_slippage_pips=(
                scenario.exit_slippage_pips
            ),
        )

    def _evaluate_trade(
        self,
        trade: OpenTrade,
        m1: pd.DataFrame,
        outcome_start_index: int,
        scenario: IntrabarScenario,
    ) -> dict:
        final_index = min(
            outcome_start_index
            + scenario.max_holding_m1_bars
            - 1,
            len(m1) - 1,
        )

        half_spread = (
            self._pips_to_price(
                scenario.spread_pips
            )
            / 2.0
        )

        exit_slippage = self._pips_to_price(
            scenario.exit_slippage_pips
        )

        for index in range(
            outcome_start_index,
            final_index + 1,
        ):
            candle = m1.iloc[index]

            mid_high = float(candle["high"])
            mid_low = float(candle["low"])

            bid_high = mid_high - half_spread
            bid_low = mid_low - half_spread
            ask_high = mid_high + half_spread
            ask_low = mid_low + half_spread

            trade.bars_held += 1

            if trade.direction == "long":
                favorable_r = (
                    bid_high
                    - trade.filled_entry_price
                ) / trade.risk_price

                adverse_r = (
                    bid_low
                    - trade.filled_entry_price
                ) / trade.risk_price

                stop_hit = (
                    bid_low <= trade.stop_price
                )

                target_hit = (
                    bid_high >= trade.target_price
                )
            else:
                favorable_r = (
                    trade.filled_entry_price
                    - ask_low
                ) / trade.risk_price

                adverse_r = (
                    trade.filled_entry_price
                    - ask_high
                ) / trade.risk_price

                stop_hit = (
                    ask_high >= trade.stop_price
                )

                target_hit = (
                    ask_low <= trade.target_price
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
                    trigger_price = (
                        trade.target_price
                    )
                    gross_r = float(
                        self.cfg.target_r
                    )
                else:
                    exit_reason = "stop"
                    trigger_price = (
                        trade.stop_price
                    )
                    gross_r = -1.0

            elif stop_hit:
                exit_reason = "stop"
                trigger_price = (
                    trade.stop_price
                )
                gross_r = -1.0

            elif target_hit:
                exit_reason = "target"
                trigger_price = (
                    trade.target_price
                )
                gross_r = float(
                    self.cfg.target_r
                )

            elif index == final_index:
                exit_reason = "time"
                trigger_price = float(
                    candle["close"]
                )

            if exit_reason is None:
                continue

            if exit_reason in {
                "stop",
                "target",
            }:
                slippage_r = (
                    exit_slippage
                    / trade.risk_price
                )

                net_r = float(
                    gross_r - slippage_r
                )

                if trade.direction == "long":
                    filled_exit_price = (
                        trigger_price
                        - exit_slippage
                    )
                else:
                    filled_exit_price = (
                        trigger_price
                        + exit_slippage
                    )
            else:
                if trade.direction == "long":
                    filled_exit_price = (
                        trigger_price
                        - half_spread
                        - exit_slippage
                    )

                    net_r = (
                        filled_exit_price
                        - trade.filled_entry_price
                    ) / trade.risk_price
                else:
                    filled_exit_price = (
                        trigger_price
                        + half_spread
                        + exit_slippage
                    )

                    net_r = (
                        trade.filled_entry_price
                        - filled_exit_price
                    ) / trade.risk_price

                gross_r = float(net_r)

            exit_timestamp = pd.Timestamp(
                candle["timestamp"]
            )

            return {
                **asdict(trade),
                "exit_m1_index": int(index),
                "exit_timestamp": exit_timestamp,
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
                "entry_after_impulse_close": bool(
                    trade.entry_timestamp
                    >= trade.available_from
                ),
            }

        raise RuntimeError(
            "Trade did not produce an exit."
        )

    def _run_scenario(
        self,
        impulses: list[M5Impulse],
        m1: pd.DataFrame,
        scenario: IntrabarScenario,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        trade_rows: list[dict] = []
        rejection_rows: list[dict] = []

        for impulse in impulses:
            window = self._m1_window(
                impulse,
                m1,
                scenario.max_wait_m1_bars,
            )

            if window.empty:
                rejection_rows.append(
                    {
                        "scenario": scenario.name,
                        "setup_id": impulse.setup_id,
                        "reason": "no_future_m1_data",
                    }
                )
                continue

            if scenario.entry_mode == "limit_touch":
                entry = self._limit_entry(
                    impulse,
                    window,
                    scenario,
                )
            elif (
                scenario.entry_mode
                == "reversal_confirm"
            ):
                entry = self._reversal_entry(
                    impulse,
                    window,
                    scenario,
                )
            else:
                raise ValueError(
                    "Unsupported entry mode: "
                    f"{scenario.entry_mode}"
                )

            if entry is None:
                rejection_rows.append(
                    {
                        "scenario": scenario.name,
                        "setup_id": impulse.setup_id,
                        "reason": (
                            "not_filled_or_invalidated"
                        ),
                    }
                )
                continue

            if (
                entry["entry_timestamp"]
                < impulse.available_from
            ):
                raise AssertionError(
                    "Lookahead detected: M1 entry "
                    "occurred before M5 impulse close."
                )

            outcome_start_index = int(
                entry["outcome_start_index"]
            )

            if outcome_start_index >= len(m1):
                rejection_rows.append(
                    {
                        "scenario": scenario.name,
                        "setup_id": impulse.setup_id,
                        "reason": (
                            "insufficient_outcome_data"
                        ),
                    }
                )
                continue

            trade = self._create_trade(
                impulse,
                entry,
                scenario,
            )

            result = self._evaluate_trade(
                trade,
                m1,
                outcome_start_index,
                scenario,
            )

            trade_rows.append(result)

        trades = pd.DataFrame(trade_rows)
        rejections = pd.DataFrame(
            rejection_rows
        )

        if not trades.empty:
            trades = trades.sort_values(
                [
                    "entry_timestamp",
                    "setup_id",
                ]
            ).reset_index(drop=True)

        return trades, rejections

    def run(
        self,
        m5_path: str | Path,
        m1_path: str | Path,
        report_dir: str | Path,
        scenarios: list[IntrabarScenario],
    ) -> dict:
        report_dir = Path(report_dir)
        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        m5 = load_candles(
            m5_path,
            self.cfg.timestamp_column,
        )

        m1 = load_candles(
            m1_path,
            self.cfg.timestamp_column,
        ).reset_index(drop=True)

        featured_m5 = add_market_features(
            m5,
            atr_period=self.cfg.atr_period,
            ema_fast=self.cfg.ema_fast,
            ema_slow=self.cfg.ema_slow,
        ).reset_index(drop=True)

        impulses = self._build_impulses(
            featured_m5
        )

        all_trade_frames: list[pd.DataFrame] = []
        all_rejection_frames: list[
            pd.DataFrame
        ] = []
        comparison_rows: list[dict] = []

        for scenario in scenarios:
            trades, rejections = (
                self._run_scenario(
                    impulses,
                    m1,
                    scenario,
                )
            )

            if (
                not trades.empty
                and not bool(
                    trades[
                        "entry_after_impulse_close"
                    ].all()
                )
            ):
                raise AssertionError(
                    "One or more entries occurred "
                    "before the M5 impulse closed."
                )

            summary = performance_summary(
                trades
            )

            comparison_rows.append(
                {
                    "scenario": scenario.name,
                    "entry_mode": (
                        scenario.entry_mode
                    ),
                    "retrace_fraction": (
                        scenario.retrace_fraction
                    ),
                    "max_retrace_fraction": (
                        scenario
                        .max_retrace_fraction
                    ),
                    "spread_pips": (
                        scenario.spread_pips
                    ),
                    "entry_slippage_pips": (
                        scenario
                        .entry_slippage_pips
                    ),
                    "exit_slippage_pips": (
                        scenario
                        .exit_slippage_pips
                    ),
                    "m5_impulses": int(
                        len(impulses)
                    ),
                    "rejected_or_unfilled": int(
                        len(rejections)
                    ),
                    "fill_rate": (
                        float(
                            len(trades)
                            / len(impulses)
                        )
                        if impulses
                        else None
                    ),
                    **summary,
                }
            )

            all_trade_frames.append(trades)
            all_rejection_frames.append(
                rejections
            )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        trades_df = (
            pd.concat(
                all_trade_frames,
                ignore_index=True,
            )
            if all_trade_frames
            else pd.DataFrame()
        )

        rejections_df = (
            pd.concat(
                all_rejection_frames,
                ignore_index=True,
            )
            if all_rejection_frames
            else pd.DataFrame()
        )

        impulses_df = pd.DataFrame(
            [asdict(x) for x in impulses]
        )

        prefix = (
            f"{self.cfg.instrument}"
            f"_v13_intrabar"
        )

        comparison_path = (
            report_dir
            / f"{prefix}_comparison.csv"
        )
        trades_path = (
            report_dir
            / f"{prefix}_trades.csv"
        )
        rejections_path = (
            report_dir
            / f"{prefix}_rejections.csv"
        )
        impulses_path = (
            report_dir
            / f"{prefix}_m5_impulses.csv"
        )
        summary_path = (
            report_dir
            / f"{prefix}_summary.json"
        )

        comparison_df.to_csv(
            comparison_path,
            index=False,
        )
        trades_df.to_csv(
            trades_path,
            index=False,
        )
        rejections_df.to_csv(
            rejections_path,
            index=False,
        )
        impulses_df.to_csv(
            impulses_path,
            index=False,
        )

        summary = {
            "engine_version": (
                "13.0-causal-m5-m1-reconstruction"
            ),
            "instrument": self.cfg.instrument,
            "m5_candles": int(len(featured_m5)),
            "m1_candles": int(len(m1)),
            "m5_impulses": int(len(impulses)),
            "required_session": (
                self.required_session
            ),
            "comparison": (
                comparison_df.to_dict(
                    orient="records"
                )
            ),
            "causality": {
                "m5_impulse_must_close_before_entry_search": True,
                "future_m1_bars_only": True,
                "retrospective_m5_entry_price_used": False,
                "signal_candle_entry_allowed": False,
                "entry_timestamp_asserted_after_m5_close": True,
                "m1_reversal_entry_outcome_starts_next_bar": True,
                "limit_fill_same_bar_exit_policy": (
                    self.cfg.intrabar_policy
                ),
            },
            "reports": {
                "comparison": str(
                    comparison_path
                ),
                "trades": str(
                    trades_path
                ),
                "rejections": str(
                    rejections_path
                ),
                "m5_impulses": str(
                    impulses_path
                ),
                "summary": str(
                    summary_path
                ),
            },
            "warning": (
                "M1 OHLC still does not reveal exact "
                "intrabar path or broker queue priority. "
                "Same-candle stop/target ambiguity uses "
                "the configured conservative policy."
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
