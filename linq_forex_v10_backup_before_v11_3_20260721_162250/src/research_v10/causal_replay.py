from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .data import load_candles
from .features import add_market_features
from .research import performance_summary
from .signal_engine import CausalSignalEngine, OpenTrade


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


class CausalReplayEngine:
    """
    Replays candles in strict chronological order.

    Signal generation and outcome evaluation are separate operations.
    """

    def __init__(
        self,
        config: ResearchConfig,
        retrace_threshold: float = 0.25,
        required_session: str = "london_open",
    ):
        self.config = config
        self.retrace_threshold = retrace_threshold
        self.required_session = required_session

    def _open_trade(
        self,
        signal: dict,
    ) -> OpenTrade:
        entry_price = float(signal["entry_price"])
        atr = float(signal["atr"])
        risk_price = self.config.stop_atr * atr

        if risk_price <= 0:
            raise ValueError("Trade risk must be positive.")

        if signal["direction"] == "long":
            stop_price = entry_price - risk_price
            target_price = (
                entry_price
                + self.config.target_r * risk_price
            )
        else:
            stop_price = entry_price + risk_price
            target_price = (
                entry_price
                - self.config.target_r * risk_price
            )

        return OpenTrade(
            setup_id=str(signal["setup_id"]),
            instrument=str(signal["instrument"]),
            direction=str(signal["direction"]),
            impulse_index=int(signal["impulse_index"]),
            entry_index=int(signal["entry_index"]),
            impulse_timestamp=pd.Timestamp(
                signal["impulse_timestamp"]
            ),
            timestamp=pd.Timestamp(signal["timestamp"]),
            entry_price=entry_price,
            stop_price=float(stop_price),
            target_price=float(target_price),
            risk_price=float(risk_price),
        )

    def _update_trade(
        self,
        trade: OpenTrade,
        candle: pd.Series,
    ) -> dict | None:
        high = float(candle["high"])
        low = float(candle["low"])

        trade.bars_held += 1

        if trade.direction == "long":
            favorable = (
                high - trade.entry_price
            ) / trade.risk_price
            adverse = (
                low - trade.entry_price
            ) / trade.risk_price

            stop_hit = low <= trade.stop_price
            target_hit = high >= trade.target_price
        else:
            favorable = (
                trade.entry_price - low
            ) / trade.risk_price
            adverse = (
                trade.entry_price - high
            ) / trade.risk_price

            stop_hit = high >= trade.stop_price
            target_hit = low <= trade.target_price

        trade.mfe_r = max(trade.mfe_r, favorable)
        trade.mae_r = min(trade.mae_r, adverse)

        outcome = None
        r_multiple = None
        exit_reason = None

        if stop_hit and target_hit:
            if self.config.intrabar_policy == "target_first":
                outcome = "win"
                r_multiple = self.config.target_r
                exit_reason = "target"
            else:
                outcome = "loss"
                r_multiple = -1.0
                exit_reason = "stop"

        elif stop_hit:
            outcome = "loss"
            r_multiple = -1.0
            exit_reason = "stop"

        elif target_hit:
            outcome = "win"
            r_multiple = self.config.target_r
            exit_reason = "target"

        elif trade.bars_held >= self.config.max_holding_bars:
            close = float(candle["close"])

            if trade.direction == "long":
                r_multiple = (
                    close - trade.entry_price
                ) / trade.risk_price
            else:
                r_multiple = (
                    trade.entry_price - close
                ) / trade.risk_price

            outcome = (
                "win"
                if r_multiple > 0
                else "loss"
                if r_multiple < 0
                else "flat"
            )
            exit_reason = "time"

        if outcome is None:
            return None

        return {
            "setup_id": trade.setup_id,
            "outcome": outcome,
            "r_multiple": float(r_multiple),
            "mfe_r": float(trade.mfe_r),
            "mae_r": float(trade.mae_r),
            "bars_held": int(trade.bars_held),
            "hit_1r": bool(trade.mfe_r >= 1.0),
            "hit_target": bool(
                trade.mfe_r >= self.config.target_r
            ),
            "exit_reason": exit_reason,
            "exit_timestamp": pd.Timestamp(
                candle["timestamp"]
            ),
        }

    def _comparison_report(
        self,
        causal: pd.DataFrame,
        historical_setups_path: str | Path | None,
    ) -> tuple[pd.DataFrame, dict]:
        if historical_setups_path is None:
            return pd.DataFrame(), {
                "historical_file_supplied": False,
            }

        path = Path(historical_setups_path)
        if not path.exists():
            return pd.DataFrame(), {
                "historical_file_supplied": True,
                "historical_file_found": False,
            }

        historical = pd.read_csv(path)

        if historical.empty:
            return pd.DataFrame(), {
                "historical_file_supplied": True,
                "historical_file_found": True,
                "historical_rows": 0,
            }

        historical["timestamp"] = pd.to_datetime(
            historical["timestamp"],
            utc=True,
            errors="coerce",
        )

        causal_compare = causal.copy()
        causal_compare["timestamp"] = pd.to_datetime(
            causal_compare["timestamp"],
            utc=True,
            errors="coerce",
        )

        historical_keys = historical[
            ["timestamp", "direction"]
        ].drop_duplicates()

        causal_keys = causal_compare[
            ["timestamp", "direction"]
        ].drop_duplicates()

        comparison = historical_keys.merge(
            causal_keys,
            on=["timestamp", "direction"],
            how="outer",
            indicator=True,
        )

        comparison["status"] = comparison["_merge"].map(
            {
                "both": "matched",
                "left_only": "historical_only",
                "right_only": "causal_only",
            }
        )

        comparison = comparison.drop(columns=["_merge"])

        matched = int(
            (comparison["status"] == "matched").sum()
        )
        historical_only = int(
            (comparison["status"] == "historical_only").sum()
        )
        causal_only = int(
            (comparison["status"] == "causal_only").sum()
        )

        historical_total = len(historical_keys)
        causal_total = len(causal_keys)

        summary = {
            "historical_file_supplied": True,
            "historical_file_found": True,
            "historical_unique_setups": int(historical_total),
            "causal_unique_setups": int(causal_total),
            "matched_setups": matched,
            "historical_only": historical_only,
            "causal_only": causal_only,
            "historical_recall": (
                matched / historical_total
                if historical_total
                else None
            ),
            "causal_precision": (
                matched / causal_total
                if causal_total
                else None
            ),
        }

        return comparison, summary

    def run(
        self,
        csv_path: str | Path,
        report_dir: str | Path,
        historical_setups_path: str | Path | None = None,
    ) -> dict:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        candles = load_candles(
            csv_path,
            self.config.timestamp_column,
        )

        featured = add_market_features(
            candles,
            atr_period=self.config.atr_period,
            ema_fast=self.config.ema_fast,
            ema_slow=self.config.ema_slow,
        ).reset_index(drop=True)

        detector = CausalSignalEngine(
            config=self.config,
            retrace_threshold=self.retrace_threshold,
            required_session=self.required_session,
        )

        signal_rows: list[dict] = []
        completed_rows: list[dict] = []
        open_trades: list[OpenTrade] = []

        warmup = max(
            self.config.ema_slow + 12,
            250,
        )

        for index in range(len(featured)):
            candle = featured.iloc[index]

            still_open: list[OpenTrade] = []

            for trade in open_trades:
                # Never evaluate a trade against its entry candle.
                if index <= trade.entry_index:
                    still_open.append(trade)
                    continue

                result = self._update_trade(
                    trade=trade,
                    candle=candle,
                )

                if result is None:
                    still_open.append(trade)
                else:
                    completed_rows.append(result)

            open_trades = still_open

            if index < warmup:
                continue

            signals = detector.process_bar(
                index=index,
                candle=candle,
            )

            for signal in signals:
                signal_rows.append(signal)

                if signal["accepted"]:
                    open_trades.append(
                        self._open_trade(signal)
                    )

        # Close trades still open at the end of available data.
        if len(featured):
            final_candle = featured.iloc[-1]

            for trade in open_trades:
                close = float(final_candle["close"])

                if trade.direction == "long":
                    r_multiple = (
                        close - trade.entry_price
                    ) / trade.risk_price
                else:
                    r_multiple = (
                        trade.entry_price - close
                    ) / trade.risk_price

                completed_rows.append(
                    {
                        "setup_id": trade.setup_id,
                        "outcome": (
                            "win"
                            if r_multiple > 0
                            else "loss"
                            if r_multiple < 0
                            else "flat"
                        ),
                        "r_multiple": float(r_multiple),
                        "mfe_r": float(trade.mfe_r),
                        "mae_r": float(trade.mae_r),
                        "bars_held": int(trade.bars_held),
                        "hit_1r": bool(trade.mfe_r >= 1.0),
                        "hit_target": bool(
                            trade.mfe_r
                            >= self.config.target_r
                        ),
                        "exit_reason": "end_of_data",
                        "exit_timestamp": pd.Timestamp(
                            final_candle["timestamp"]
                        ),
                    }
                )

        signals_df = pd.DataFrame(signal_rows)
        outcomes_df = pd.DataFrame(completed_rows)

        if signals_df.empty:
            causal = signals_df.copy()
        elif outcomes_df.empty:
            causal = signals_df.copy()
        else:
            causal = signals_df.merge(
                outcomes_df,
                on="setup_id",
                how="left",
                validate="one_to_one",
            )

        if causal.empty:
            accepted = causal.copy()
            rejected = causal.copy()
        else:
            accepted = causal[
                causal["accepted"] == True  # noqa: E712
            ].copy()

            rejected = causal[
                causal["accepted"] == False  # noqa: E712
            ].copy()

        comparison, comparison_summary = (
            self._comparison_report(
                causal=causal,
                historical_setups_path=historical_setups_path,
            )
        )

        prefix = f"{self.config.instrument}_v11_causal"

        signals_path = report_dir / f"{prefix}_signals.csv"
        accepted_path = report_dir / f"{prefix}_accepted.csv"
        rejected_path = report_dir / f"{prefix}_rejected.csv"
        comparison_path = (
            report_dir / f"{prefix}_historical_comparison.csv"
        )
        summary_path = report_dir / f"{prefix}_summary.json"

        causal.to_csv(signals_path, index=False)
        accepted.to_csv(accepted_path, index=False)
        rejected.to_csv(rejected_path, index=False)
        comparison.to_csv(comparison_path, index=False)

        completed_accepted = accepted.dropna(
            subset=["r_multiple"]
        ) if "r_multiple" in accepted.columns else accepted.iloc[0:0]

        performance = (
            performance_summary(completed_accepted)
            if not completed_accepted.empty
            else {
                "trades": 0,
                "win_rate": None,
                "expectancy_r": None,
                "profit_factor": None,
                "max_drawdown_r": None,
                "total_r": None,
            }
        )

        summary = {
            "engine_version": "11.1-causal-bar-replay",
            "instrument": self.config.instrument,
            "candles_processed": int(len(featured)),
            "locked_rule": (
                f"retrace_fraction le "
                f"{self.retrace_threshold} "
                f"AND session eq "
                f"{self.required_session}"
            ),
            "all_causal_signals": int(len(causal)),
            "accepted_trades": int(len(accepted)),
            "rejected_signals": int(len(rejected)),
            "completed_accepted_trades": int(
                len(completed_accepted)
            ),
            "performance": performance,
            "causality": {
                "bar_by_bar_processing": True,
                "future_candles_used_for_signal": False,
                "entry_candle_used_for_outcome": False,
                "signal_and_outcome_logic_separated": True,
            },
            "historical_comparison": comparison_summary,
            "reports": {
                "signals": str(signals_path),
                "accepted": str(accepted_path),
                "rejected": str(rejected_path),
                "historical_comparison": str(comparison_path),
                "summary": str(summary_path),
            },
            "warning": (
                "Results still exclude spread, slippage, latency, "
                "financing, rejected orders, and broker execution."
            ),
        }

        summary_path.write_text(
            json.dumps(summary, indent=2, default=str)
        )

        return summary
