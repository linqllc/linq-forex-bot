from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd


LABEL_COLUMNS = {
    "outcome_r",
    "result_r",
    "realized_r",
    "r_multiple",
    "win",
    "winner",
    "target_hit",
    "stop_hit",
    "exit_price",
    "exit_timestamp",
    "exit_index",
    "mfe_r",
    "mae_r",
}


@dataclass(frozen=True)
class ReplayConfig:
    rule: str
    minimum_minutes_between_trades: int = 0
    one_trade_per_session: bool = False


def parse_rule(rule: str) -> list[tuple[str, str, str]]:
    clauses: list[tuple[str, str, str]] = []

    for raw_clause in rule.split(" AND "):
        parts = raw_clause.strip().split(" ", 2)

        if len(parts) != 3:
            raise ValueError(f"Invalid rule clause: {raw_clause!r}")

        feature, operator, value = parts

        if operator not in {"le", "ge", "eq"}:
            raise ValueError(f"Unsupported rule operator: {operator}")

        clauses.append((feature, operator, value))

    if not clauses:
        raise ValueError("Rule cannot be empty.")

    return clauses


def apply_rule(
    dataframe: pd.DataFrame,
    rule: str,
) -> pd.Series:
    mask = pd.Series(True, index=dataframe.index, dtype=bool)

    for feature, operator, raw_value in parse_rule(rule):
        if feature not in dataframe.columns:
            raise KeyError(
                f"Rule requires missing feature column: {feature}"
            )

        series = dataframe[feature]

        if operator == "eq":
            mask &= series.astype(str).str.lower() == raw_value.lower()
            continue

        threshold = float(raw_value)
        numeric = pd.to_numeric(series, errors="coerce")

        if operator == "le":
            mask &= numeric <= threshold
        elif operator == "ge":
            mask &= numeric >= threshold

    return mask.fillna(False)


def find_timestamp_column(dataframe: pd.DataFrame) -> str:
    for candidate in (
        "timestamp",
        "time",
        "entry_timestamp",
        "setup_timestamp",
        "datetime",
        "date",
    ):
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        "Could not locate a timestamp column in the setup report."
    )


def find_outcome_column(dataframe: pd.DataFrame) -> str | None:
    for candidate in (
        "outcome_r",
        "result_r",
        "realized_r",
        "r_multiple",
    ):
        if candidate in dataframe.columns:
            return candidate

    return None


def calculate_performance(
    accepted: pd.DataFrame,
) -> dict[str, Any]:
    outcome_column = find_outcome_column(accepted)

    if outcome_column is None or accepted.empty:
        return {
            "trades": int(len(accepted)),
            "win_rate": None,
            "expectancy_r": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "total_r": None,
            "outcome_column": outcome_column,
        }

    outcomes = pd.to_numeric(
        accepted[outcome_column],
        errors="coerce",
    ).dropna()

    if outcomes.empty:
        return {
            "trades": int(len(accepted)),
            "win_rate": None,
            "expectancy_r": None,
            "profit_factor": None,
            "max_drawdown_r": None,
            "total_r": None,
            "outcome_column": outcome_column,
        }

    wins = outcomes[outcomes > 0]
    losses = outcomes[outcomes < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else math.inf
    )

    equity = outcomes.cumsum()
    running_peak = equity.cummax().clip(lower=0)
    drawdown = equity - running_peak

    return {
        "trades": int(len(outcomes)),
        "win_rate": float((outcomes > 0).mean()),
        "expectancy_r": float(outcomes.mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": float(drawdown.min()),
        "total_r": float(outcomes.sum()),
        "outcome_column": outcome_column,
    }


def build_decision_log(
    setups: pd.DataFrame,
    config: ReplayConfig,
) -> pd.DataFrame:
    timestamp_column = find_timestamp_column(setups)

    working = setups.copy()
    working[timestamp_column] = pd.to_datetime(
        working[timestamp_column],
        utc=True,
        errors="coerce",
    )
    working = working.dropna(subset=[timestamp_column])
    working = working.sort_values(timestamp_column).reset_index(drop=True)

    signal_mask = apply_rule(working, config.rule)

    decisions: list[dict[str, Any]] = []
    last_trade_time: pd.Timestamp | None = None
    traded_sessions: set[tuple[Any, ...]] = set()

    rule_features = {
        feature for feature, _, _ in parse_rule(config.rule)
    }

    illegal_rule_features = sorted(rule_features & LABEL_COLUMNS)

    if illegal_rule_features:
        raise ValueError(
            "Replay rule contains future/outcome fields: "
            + ", ".join(illegal_rule_features)
        )

    for index, row in working.iterrows():
        timestamp = row[timestamp_column]
        signal_passed = bool(signal_mask.iloc[index])

        accepted = signal_passed
        rejection_reason = ""

        if not signal_passed:
            accepted = False
            rejection_reason = "rule_failed"

        if accepted and config.minimum_minutes_between_trades > 0:
            if last_trade_time is not None:
                elapsed_minutes = (
                    timestamp - last_trade_time
                ).total_seconds() / 60

                if elapsed_minutes < config.minimum_minutes_between_trades:
                    accepted = False
                    rejection_reason = "cooldown_active"

        if accepted and config.one_trade_per_session:
            session_value = row.get("session", "unknown")
            session_key = (
                timestamp.date(),
                str(session_value),
            )

            if session_key in traded_sessions:
                accepted = False
                rejection_reason = "session_trade_limit"
            else:
                traded_sessions.add(session_key)

        if accepted:
            last_trade_time = timestamp

        decision = row.to_dict()
        decision.update(
            {
                "replay_timestamp": timestamp,
                "signal_passed": signal_passed,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "locked_rule": config.rule,
            }
        )
        decisions.append(decision)

    return pd.DataFrame(decisions)


def prefix_causality_audit(
    candles_csv: str | Path,
    accepted: pd.DataFrame,
    rule: str,
    instrument: str,
    checks: int = 12,
) -> pd.DataFrame:
    """
    Rebuild the research pipeline using candle prefixes.

    For sampled accepted signals, the engine truncates the candle file at the
    decision timestamp. The same setup must still be discoverable without any
    candles from the future.
    """
    from .config import ResearchConfig
    from .data import load_candles
    from .features import add_market_features
    from .setups import detect_candidate_setups

    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "prefix_candle_rows",
                "setup_found_in_prefix",
                "rule_passed_in_prefix",
                "audit_passed",
                "error",
            ]
        )

    setup_timestamp_column = find_timestamp_column(accepted)
    candles = load_candles(candles_csv, None)
    candle_timestamp_column = find_timestamp_column(candles)

    candles[candle_timestamp_column] = pd.to_datetime(
        candles[candle_timestamp_column],
        utc=True,
        errors="coerce",
    )
    candles = candles.dropna(subset=[candle_timestamp_column])
    candles = candles.sort_values(candle_timestamp_column)

    sample_count = min(checks, len(accepted))
    sample_positions = np.linspace(
        0,
        len(accepted) - 1,
        sample_count,
        dtype=int,
    )

    sampled = accepted.iloc[sample_positions]
    research_config = ResearchConfig(instrument=instrument)

    rows: list[dict[str, Any]] = []

    for _, accepted_row in sampled.iterrows():
        timestamp = pd.to_datetime(
            accepted_row[setup_timestamp_column],
            utc=True,
        )

        prefix = candles[
            candles[candle_timestamp_column] <= timestamp
        ].copy()

        setup_found = False
        rule_passed = False
        error = ""

        try:
            featured = add_market_features(
                prefix,
                atr_period=research_config.atr_period,
                ema_fast=research_config.ema_fast,
                ema_slow=research_config.ema_slow,
            )

            prefix_setups = detect_candidate_setups(
                featured,
                research_config,
            )

            if not prefix_setups.empty:
                prefix_timestamp_column = find_timestamp_column(
                    prefix_setups
                )

                prefix_setups[prefix_timestamp_column] = pd.to_datetime(
                    prefix_setups[prefix_timestamp_column],
                    utc=True,
                    errors="coerce",
                )

                matching = prefix_setups[
                    prefix_setups[prefix_timestamp_column] == timestamp
                ]

                setup_found = not matching.empty

                if setup_found:
                    rule_passed = bool(
                        apply_rule(matching, rule).any()
                    )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "timestamp": timestamp,
                "prefix_candle_rows": int(len(prefix)),
                "setup_found_in_prefix": setup_found,
                "rule_passed_in_prefix": rule_passed,
                "audit_passed": setup_found and rule_passed,
                "error": error,
            }
        )

    return pd.DataFrame(rows)


def run_replay(
    setups_csv: str | Path,
    candles_csv: str | Path,
    report_dir: str | Path,
    instrument: str,
    rule: str,
    minimum_minutes_between_trades: int = 0,
    one_trade_per_session: bool = False,
    causality_checks: int = 12,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    setups = pd.read_csv(setups_csv)

    config = ReplayConfig(
        rule=rule,
        minimum_minutes_between_trades=minimum_minutes_between_trades,
        one_trade_per_session=one_trade_per_session,
    )

    decision_log = build_decision_log(setups, config)
    accepted = decision_log[decision_log["accepted"]].copy()
    rejected = decision_log[~decision_log["accepted"]].copy()

    causality = prefix_causality_audit(
        candles_csv=candles_csv,
        accepted=accepted,
        rule=rule,
        instrument=instrument,
        checks=causality_checks,
    )

    performance = calculate_performance(accepted)

    prefix = f"{instrument}_v11"
    decisions_path = report_dir / f"{prefix}_decisions.csv"
    accepted_path = report_dir / f"{prefix}_accepted_trades.csv"
    rejected_path = report_dir / f"{prefix}_rejected_setups.csv"
    causality_path = report_dir / f"{prefix}_causality_audit.csv"
    summary_path = report_dir / f"{prefix}_summary.json"

    decision_log.to_csv(decisions_path, index=False)
    accepted.to_csv(accepted_path, index=False)
    rejected.to_csv(rejected_path, index=False)
    causality.to_csv(causality_path, index=False)

    causality_passes = (
        int(causality["audit_passed"].sum())
        if not causality.empty
        else 0
    )

    summary = {
        "engine_version": "11.0-replay",
        "instrument": instrument,
        "locked_rule": rule,
        "source_setups": int(len(setups)),
        "accepted_trades": int(len(accepted)),
        "rejected_setups": int(len(rejected)),
        "performance": performance,
        "causality_audit": {
            "checks": int(len(causality)),
            "passes": causality_passes,
            "failures": int(len(causality) - causality_passes),
            "pass_rate": (
                float(causality["audit_passed"].mean())
                if not causality.empty
                else None
            ),
        },
        "execution_controls": {
            "minimum_minutes_between_trades":
                minimum_minutes_between_trades,
            "one_trade_per_session": one_trade_per_session,
        },
        "reports": {
            "decisions": str(decisions_path),
            "accepted_trades": str(accepted_path),
            "rejected_setups": str(rejected_path),
            "causality_audit": str(causality_path),
            "summary": str(summary_path),
        },
        "warning": (
            "Replay validation reduces look-ahead risk but does not model "
            "spread, slippage, latency, financing, or broker execution."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
