from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import load_candles
from .features import add_market_features
from .signal_engine import CausalSignalEngine


@dataclass(frozen=True)
class ExecutionScenario:
    name: str
    entry_mode: str

    spread_pips: float = 0.0
    entry_slippage_pips: float = 0.0
    exit_slippage_pips: float = 0.0
    latency_pips: float = 0.0


@dataclass
class ExecutedTrade:
    scenario: str

    setup_id: str
    instrument: str
    direction: str

    signal_index: int
    entry_index: int
    outcome_start_index: int

    impulse_timestamp: pd.Timestamp
    signal_timestamp: pd.Timestamp
    entry_timestamp: pd.Timestamp

    signal_entry_price: float
    raw_execution_price: float
    filled_entry_price: float

    atr: float
    risk_price: float
    stop_price: float
    target_price: float

    spread_pips: float
    entry_slippage_pips: float
    exit_slippage_pips: float
    latency_pips: float

    session: str
    retrace_fraction: float

    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def profit_factor(values: pd.Series) -> float | None:
    wins = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))

    if losses == 0:
        if wins == 0:
            return None

        return float("inf")

    return wins / losses


def max_drawdown_r(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = values.cumsum()
    high_watermark = equity.cummax()
    drawdown = equity - high_watermark

    return float(drawdown.min())


def longest_streak(values: pd.Series, winning: bool) -> int:
    longest = 0
    current = 0

    for value in values:
        match = value > 0 if winning else value < 0

        if match:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return int(longest)


def performance_summary(trades: pd.DataFrame) -> dict:
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
        "profit_factor": profit_factor(values),
        "max_drawdown_r": max_drawdown_r(values),
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
        "longest_win_streak": longest_streak(
            values,
            winning=True,
        ),
        "longest_loss_streak": longest_streak(
            values,
            winning=False,
        ),
    }


class ExactSignalExecutionComparison:
    """
    Execution comparison using one frozen causal signal population.

    Execution models are not allowed to generate, filter or modify signals.
    Every scenario receives the same accepted signal list.
    """

    def __init__(
        self,
        research_config: ResearchConfig,
        retrace_threshold: float = 0.25,
        required_session: str = "london_open",
        expected_accepted_signals: int | None = 498,
        minimum_stop_pips: float = 2.0,
    ):
        self.cfg = research_config
        self.retrace_threshold = float(
            retrace_threshold
        )
        self.required_session = required_session
        self.expected_accepted_signals = (
            expected_accepted_signals
        )
        self.minimum_stop_pips = float(
            minimum_stop_pips
        )

    def _price_from_pips(
        self,
        pips: float,
    ) -> float:
        return float(pips * self.cfg.pip_size)

    def _signal_identity_hash(
        self,
        signals: list[dict],
    ) -> str:
        identities = sorted(
            str(signal["setup_id"])
            for signal in signals
        )

        payload = "\n".join(identities)

        return sha256(
            payload.encode("utf-8")
        ).hexdigest()

    def _collect_signals(
        self,
        featured: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        detector = CausalSignalEngine(
            config=self.cfg,
            retrace_threshold=self.retrace_threshold,
            required_session=self.required_session,
        )

        all_signals: list[dict] = []
        accepted_signals: list[dict] = []

        warmup = max(
            self.cfg.ema_slow + 12,
            250,
        )

        for index in range(
            warmup,
            len(featured),
        ):
            candle = featured.iloc[index]

            emitted = detector.process_bar(
                index=index,
                candle=candle,
            )

            for signal in emitted:
                normalized = dict(signal)

                normalized["entry_index"] = int(
                    normalized.get(
                        "entry_index",
                        index,
                    )
                )

                all_signals.append(normalized)

                if bool(normalized["accepted"]):
                    accepted_signals.append(
                        normalized
                    )

        setup_ids = [
            str(signal["setup_id"])
            for signal in accepted_signals
        ]

        if len(setup_ids) != len(set(setup_ids)):
            duplicates = (
                pd.Series(setup_ids)
                .value_counts()
            )

            duplicates = duplicates[
                duplicates > 1
            ]

            raise AssertionError(
                "Duplicate accepted setup IDs found: "
                f"{duplicates.to_dict()}"
            )

        if (
            self.expected_accepted_signals
            is not None
            and len(accepted_signals)
            != self.expected_accepted_signals
        ):
            raise AssertionError(
                "Frozen signal population mismatch. "
                f"Expected "
                f"{self.expected_accepted_signals}, "
                f"received "
                f"{len(accepted_signals)}."
            )

        return all_signals, accepted_signals

    def _entry_details(
        self,
        signal: dict,
        featured: pd.DataFrame,
        scenario: ExecutionScenario,
    ) -> tuple[
        int,
        int,
        pd.Timestamp,
        float,
    ] | None:
        signal_index = int(
            signal["entry_index"]
        )

        if scenario.entry_mode == "signal_price":
            entry_index = signal_index
            outcome_start_index = (
                signal_index + 1
            )

            raw_execution_price = float(
                signal["entry_price"]
            )

            entry_timestamp = pd.Timestamp(
                signal["timestamp"]
            )

        elif scenario.entry_mode == "signal_close":
            entry_index = signal_index
            outcome_start_index = (
                signal_index + 1
            )

            raw_execution_price = float(
                featured.iloc[
                    signal_index
                ]["close"]
            )

            entry_timestamp = pd.Timestamp(
                featured.iloc[
                    signal_index
                ]["timestamp"]
            )

        elif scenario.entry_mode == "next_open":
            entry_index = signal_index + 1
            outcome_start_index = (
                entry_index + 1
            )

            if entry_index >= len(featured):
                return None

            raw_execution_price = float(
                featured.iloc[
                    entry_index
                ]["open"]
            )

            entry_timestamp = pd.Timestamp(
                featured.iloc[
                    entry_index
                ]["timestamp"]
            )

        else:
            raise ValueError(
                "Unsupported entry mode: "
                f"{scenario.entry_mode}"
            )

        if outcome_start_index >= len(featured):
            return None

        return (
            entry_index,
            outcome_start_index,
            entry_timestamp,
            raw_execution_price,
        )

    def _create_trade(
        self,
        signal: dict,
        featured: pd.DataFrame,
        scenario: ExecutionScenario,
    ) -> ExecutedTrade | None:
        entry_details = self._entry_details(
            signal=signal,
            featured=featured,
            scenario=scenario,
        )

        if entry_details is None:
            return None

        (
            entry_index,
            outcome_start_index,
            entry_timestamp,
            raw_execution_price,
        ) = entry_details

        direction = str(signal["direction"])
        atr = float(signal["atr"])

        if not np.isfinite(atr) or atr <= 0:
            return None

        half_spread = self._price_from_pips(
            scenario.spread_pips
        ) / 2.0

        entry_slippage = self._price_from_pips(
            scenario.entry_slippage_pips
        )

        latency = self._price_from_pips(
            scenario.latency_pips
        )

        total_adverse_entry_cost = (
            half_spread
            + entry_slippage
            + latency
        )

        if direction == "long":
            filled_entry_price = (
                raw_execution_price
                + total_adverse_entry_cost
            )
        elif direction == "short":
            filled_entry_price = (
                raw_execution_price
                - total_adverse_entry_cost
            )
        else:
            raise ValueError(
                f"Unsupported direction: {direction}"
            )

        minimum_risk = (
            self.minimum_stop_pips
            * self.cfg.pip_size
        )

        risk_price = max(
            self.cfg.stop_atr * atr,
            minimum_risk,
        )

        if direction == "long":
            stop_price = (
                filled_entry_price - risk_price
            )

            target_price = (
                filled_entry_price
                + self.cfg.target_r
                * risk_price
            )
        else:
            stop_price = (
                filled_entry_price + risk_price
            )

            target_price = (
                filled_entry_price
                - self.cfg.target_r
                * risk_price
            )

        return ExecutedTrade(
            scenario=scenario.name,
            setup_id=str(signal["setup_id"]),
            instrument=str(signal["instrument"]),
            direction=direction,
            signal_index=int(
                signal["entry_index"]
            ),
            entry_index=entry_index,
            outcome_start_index=(
                outcome_start_index
            ),
            impulse_timestamp=pd.Timestamp(
                signal["impulse_timestamp"]
            ),
            signal_timestamp=pd.Timestamp(
                signal["timestamp"]
            ),
            entry_timestamp=entry_timestamp,
            signal_entry_price=float(
                signal["entry_price"]
            ),
            raw_execution_price=float(
                raw_execution_price
            ),
            filled_entry_price=float(
                filled_entry_price
            ),
            atr=atr,
            risk_price=float(risk_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            spread_pips=float(
                scenario.spread_pips
            ),
            entry_slippage_pips=float(
                scenario.entry_slippage_pips
            ),
            exit_slippage_pips=float(
                scenario.exit_slippage_pips
            ),
            latency_pips=float(
                scenario.latency_pips
            ),
            session=str(signal["session"]),
            retrace_fraction=float(
                signal["retrace_fraction"]
            ),
        )

    def _tradable_extremes(
        self,
        candle: pd.Series,
        scenario: ExecutionScenario,
    ) -> dict[str, float]:
        half_spread = self._price_from_pips(
            scenario.spread_pips
        ) / 2.0

        mid_high = float(candle["high"])
        mid_low = float(candle["low"])

        return {
            "bid_high": (
                mid_high - half_spread
            ),
            "bid_low": (
                mid_low - half_spread
            ),
            "ask_high": (
                mid_high + half_spread
            ),
            "ask_low": (
                mid_low + half_spread
            ),
        }

    def _evaluate_trade(
        self,
        trade: ExecutedTrade,
        featured: pd.DataFrame,
        scenario: ExecutionScenario,
    ) -> dict:
        final_index = min(
            trade.outcome_start_index
            + self.cfg.max_holding_bars
            - 1,
            len(featured) - 1,
        )

        exit_slippage_price = (
            self._price_from_pips(
                scenario.exit_slippage_pips
            )
        )

        for index in range(
            trade.outcome_start_index,
            final_index + 1,
        ):
            candle = featured.iloc[index]

            extremes = self._tradable_extremes(
                candle,
                scenario,
            )

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
            gross_r = None
            trigger_exit_price = None

            if stop_hit and target_hit:
                if (
                    self.cfg.intrabar_policy
                    == "target_first"
                ):
                    exit_reason = "target"
                    gross_r = float(
                        self.cfg.target_r
                    )
                    trigger_exit_price = (
                        trade.target_price
                    )
                else:
                    exit_reason = "stop"
                    gross_r = -1.0
                    trigger_exit_price = (
                        trade.stop_price
                    )

            elif stop_hit:
                exit_reason = "stop"
                gross_r = -1.0
                trigger_exit_price = (
                    trade.stop_price
                )

            elif target_hit:
                exit_reason = "target"
                gross_r = float(
                    self.cfg.target_r
                )
                trigger_exit_price = (
                    trade.target_price
                )

            elif index == final_index:
                exit_reason = (
                    "time"
                    if (
                        trade.bars_held
                        >= self.cfg.max_holding_bars
                    )
                    else "end_of_data"
                )

                trigger_exit_price = float(
                    candle["close"]
                )

            if exit_reason is None:
                continue

            if exit_reason in {
                "stop",
                "target",
            }:
                slippage_r = (
                    exit_slippage_price
                    / trade.risk_price
                )

                net_r = float(
                    gross_r - slippage_r
                )

                if trade.direction == "long":
                    filled_exit_price = (
                        trigger_exit_price
                        - exit_slippage_price
                    )
                else:
                    filled_exit_price = (
                        trigger_exit_price
                        + exit_slippage_price
                    )

            else:
                half_spread = (
                    self._price_from_pips(
                        scenario.spread_pips
                    )
                    / 2.0
                )

                if trade.direction == "long":
                    filled_exit_price = (
                        trigger_exit_price
                        - half_spread
                        - exit_slippage_price
                    )

                    net_r = (
                        filled_exit_price
                        - trade.filled_entry_price
                    ) / trade.risk_price
                else:
                    filled_exit_price = (
                        trigger_exit_price
                        + half_spread
                        + exit_slippage_price
                    )

                    net_r = (
                        trade.filled_entry_price
                        - filled_exit_price
                    ) / trade.risk_price

                gross_r = float(net_r)

            return {
                **asdict(trade),
                "exit_index": index,
                "exit_timestamp": pd.Timestamp(
                    candle["timestamp"]
                ),
                "trigger_exit_price": float(
                    trigger_exit_price
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

        raise RuntimeError(
            "Trade evaluation ended without "
            "producing an exit."
        )

    def _run_scenario(
        self,
        accepted_signals: list[dict],
        featured: pd.DataFrame,
        scenario: ExecutionScenario,
    ) -> tuple[pd.DataFrame, list[dict]]:
        completed_rows: list[dict] = []
        unexecuted_rows: list[dict] = []

        for signal in accepted_signals:
            trade = self._create_trade(
                signal=signal,
                featured=featured,
                scenario=scenario,
            )

            if trade is None:
                unexecuted_rows.append(
                    {
                        "scenario": scenario.name,
                        "setup_id": signal.get(
                            "setup_id"
                        ),
                        "signal_timestamp": (
                            signal.get("timestamp")
                        ),
                        "reason": (
                            "insufficient_data_or_invalid_atr"
                        ),
                    }
                )
                continue

            completed_rows.append(
                self._evaluate_trade(
                    trade=trade,
                    featured=featured,
                    scenario=scenario,
                )
            )

        trades = pd.DataFrame(
            completed_rows
        )

        if not trades.empty:
            trades = trades.sort_values(
                [
                    "signal_timestamp",
                    "setup_id",
                ]
            ).reset_index(drop=True)

        return trades, unexecuted_rows

    def _monthly_summary(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()

        output = trades.copy()

        output["entry_timestamp"] = (
            pd.to_datetime(
                output["entry_timestamp"],
                utc=True,
            )
        )

        output["period"] = (
            output["entry_timestamp"]
            .dt.tz_localize(None)
            .dt.to_period("M")
            .astype(str)
        )

        rows = []

        for (
            scenario,
            period,
        ), group in output.groupby(
            [
                "scenario",
                "period",
            ],
            sort=True,
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "period": period,
                    **performance_summary(group),
                }
            )

        return pd.DataFrame(rows)

    def run(
        self,
        csv_path: str | Path,
        report_dir: str | Path,
        scenarios: list[ExecutionScenario],
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

        (
            all_signals,
            accepted_signals,
        ) = self._collect_signals(featured)

        frozen_hash = self._signal_identity_hash(
            accepted_signals
        )

        comparison_rows: list[dict] = []
        all_trade_frames: list[pd.DataFrame] = []
        all_unexecuted_rows: list[dict] = []

        scenario_signal_checks: dict[
            str,
            dict,
        ] = {}

        for scenario in scenarios:
            trades, unexecuted = (
                self._run_scenario(
                    accepted_signals=accepted_signals,
                    featured=featured,
                    scenario=scenario,
                )
            )

            executed_ids = (
                set(
                    trades["setup_id"].astype(str)
                )
                if not trades.empty
                else set()
            )

            accepted_ids = {
                str(signal["setup_id"])
                for signal in accepted_signals
            }

            unexpected_ids = (
                executed_ids - accepted_ids
            )

            if unexpected_ids:
                raise AssertionError(
                    f"Scenario {scenario.name} "
                    "created trades that were not in "
                    "the frozen signal population."
                )

            executed_hash = (
                self._signal_identity_hash(
                    [
                        {
                            "setup_id": setup_id
                        }
                        for setup_id
                        in sorted(executed_ids)
                    ]
                )
                if len(executed_ids)
                == len(accepted_ids)
                else None
            )

            full_identity_match = bool(
                executed_ids == accepted_ids
            )

            scenario_signal_checks[
                scenario.name
            ] = {
                "accepted_signal_count": int(
                    len(accepted_ids)
                ),
                "executed_trade_count": int(
                    len(executed_ids)
                ),
                "unexecuted_signal_count": int(
                    len(unexecuted)
                ),
                "full_signal_identity_match": (
                    full_identity_match
                ),
                "frozen_signal_hash": frozen_hash,
                "executed_signal_hash": (
                    executed_hash
                ),
            }

            summary = performance_summary(
                trades
            )

            comparison_rows.append(
                {
                    "scenario": scenario.name,
                    "entry_mode": (
                        scenario.entry_mode
                    ),
                    "spread_pips": (
                        scenario.spread_pips
                    ),
                    "entry_slippage_pips": (
                        scenario.entry_slippage_pips
                    ),
                    "exit_slippage_pips": (
                        scenario.exit_slippage_pips
                    ),
                    "latency_pips": (
                        scenario.latency_pips
                    ),
                    **summary,
                }
            )

            all_trade_frames.append(trades)
            all_unexecuted_rows.extend(
                unexecuted
            )

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        all_trades_df = (
            pd.concat(
                all_trade_frames,
                ignore_index=True,
            )
            if all_trade_frames
            else pd.DataFrame()
        )

        unexecuted_df = pd.DataFrame(
            all_unexecuted_rows
        )

        monthly_df = self._monthly_summary(
            all_trades_df
        )

        signals_df = pd.DataFrame(
            accepted_signals
        )

        all_signals_df = pd.DataFrame(
            all_signals
        )

        prefix = (
            f"{self.cfg.instrument}"
            f"_v12_execution_compare"
        )

        comparison_path = (
            report_dir
            / f"{prefix}_comparison.csv"
        )

        trades_path = (
            report_dir
            / f"{prefix}_trades.csv"
        )

        monthly_path = (
            report_dir
            / f"{prefix}_monthly.csv"
        )

        accepted_signals_path = (
            report_dir
            / f"{prefix}_accepted_signals.csv"
        )

        all_signals_path = (
            report_dir
            / f"{prefix}_all_signals.csv"
        )

        unexecuted_path = (
            report_dir
            / f"{prefix}_unexecuted.csv"
        )

        summary_path = (
            report_dir
            / f"{prefix}_summary.json"
        )

        comparison_df.to_csv(
            comparison_path,
            index=False,
        )

        all_trades_df.to_csv(
            trades_path,
            index=False,
        )

        monthly_df.to_csv(
            monthly_path,
            index=False,
        )

        signals_df.to_csv(
            accepted_signals_path,
            index=False,
        )

        all_signals_df.to_csv(
            all_signals_path,
            index=False,
        )

        unexecuted_df.to_csv(
            unexecuted_path,
            index=False,
        )

        summary = {
            "engine_version": (
                "12.0-frozen-signal-execution-compare"
            ),
            "instrument": self.cfg.instrument,
            "candles_processed": int(
                len(featured)
            ),
            "locked_rule": {
                "retrace_fraction_max": (
                    self.retrace_threshold
                ),
                "required_session": (
                    self.required_session
                ),
            },
            "frozen_signal_population": {
                "all_signals": int(
                    len(all_signals)
                ),
                "accepted_signals": int(
                    len(accepted_signals)
                ),
                "expected_accepted_signals": (
                    self.expected_accepted_signals
                ),
                "count_match": bool(
                    self.expected_accepted_signals
                    is None
                    or len(accepted_signals)
                    == self.expected_accepted_signals
                ),
                "unique_setup_ids": bool(
                    signals_df["setup_id"].nunique()
                    == len(signals_df)
                ),
                "identity_hash": frozen_hash,
            },
            "scenario_signal_checks": (
                scenario_signal_checks
            ),
            "comparison": (
                comparison_df.to_dict(
                    orient="records"
                )
            ),
            "causality": {
                "bar_by_bar_signal_generation": True,
                "future_candles_used_for_signal": False,
                "signal_population_frozen_before_execution": True,
                "execution_models_generate_signals": False,
                "signal_candle_used_for_outcome": False,
                "next_open_entry_candle_used_for_outcome": False,
            },
            "reports": {
                "comparison": str(
                    comparison_path
                ),
                "trades": str(
                    trades_path
                ),
                "monthly": str(
                    monthly_path
                ),
                "accepted_signals": str(
                    accepted_signals_path
                ),
                "all_signals": str(
                    all_signals_path
                ),
                "unexecuted": str(
                    unexecuted_path
                ),
                "summary": str(
                    summary_path
                ),
            },
            "warning": (
                "Signal-close execution is approximated "
                "using completed M5 candle data. Accurate "
                "post-close latency and fill reconstruction "
                "requires lower-timeframe bid/ask or tick data."
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
