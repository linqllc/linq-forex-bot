"""
Automated research experiment grids.

This module evaluates combinations of probability thresholds, trading
sessions, and spread limits against already-scored out-of-sample predictions.

Parameters that change trade outcomes themselves—such as stop ATR and target R—
must be evaluated by regenerating the labeled dataset. A generic parameter-grid
utility is included so those experiments can be integrated later.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping

import pandas as pd

from .optimization import (
    evaluate_probability_thresholds,
    rank_experiments,
)


__all__ = [
    "ExperimentGridConfig",
    "build_parameter_grid",
    "run_experiment_grid",
]


@dataclass(frozen=True)
class ExperimentGridConfig:
    """Configuration for scored-prediction research."""

    thresholds: tuple[float, ...]
    sessions: tuple[str, ...] = ("all",)
    maximum_spreads: tuple[float | None, ...] = (None,)
    minimum_trades: int = 10
    probability_column: str = "probability_1r"
    result_column: str = "net_result_r"
    holdout_column: str = "holdout"
    session_column: str = "session"
    spread_column: str = "spread_pips_at_entry"


def build_parameter_grid(
    parameters: Mapping[str, Iterable[Any]],
) -> list[dict[str, Any]]:
    """
    Return the Cartesian product of a parameter mapping.

    Example
    -------
    ``{"stop_atr": [1.0, 1.5], "target_r": [1.0, 2.0]}``
    produces four experiment dictionaries.
    """

    if not parameters:
        raise ValueError("At least one experiment parameter is required.")

    names = list(parameters)
    values = [list(parameters[name]) for name in names]

    empty = [name for name, options in zip(names, values, strict=True) if not options]

    if empty:
        raise ValueError("Experiment parameters cannot have empty value lists: " + ", ".join(empty))

    return [dict(zip(names, combination, strict=True)) for combination in product(*values)]


def _filter_session(
    predictions: pd.DataFrame,
    session: str,
    session_column: str,
) -> pd.DataFrame:
    if session == "all":
        return predictions.copy()

    if session_column not in predictions.columns:
        raise ValueError(f"Session research requires column: {session_column}")

    normalized = predictions[session_column].astype(str).str.lower()

    return predictions.loc[normalized == session.lower()].copy()


def _filter_spread(
    predictions: pd.DataFrame,
    maximum_spread: float | None,
    spread_column: str,
) -> pd.DataFrame:
    if maximum_spread is None:
        return predictions.copy()

    if maximum_spread < 0:
        raise ValueError("Maximum spread cannot be negative.")

    if spread_column not in predictions.columns:
        raise ValueError(f"Spread research requires column: {spread_column}")

    spread = pd.to_numeric(
        predictions[spread_column],
        errors="coerce",
    )

    return predictions.loc[spread.notna() & (spread <= maximum_spread)].copy()


def run_experiment_grid(
    predictions: pd.DataFrame,
    config: ExperimentGridConfig,
) -> pd.DataFrame:
    """
    Evaluate and rank all configured experiment combinations.

    Only holdout rows are scored by the underlying threshold evaluator.
    The returned leaderboard includes both eligible and ineligible experiments,
    with eligible experiments ranked first.
    """

    if not config.thresholds:
        raise ValueError("At least one threshold is required.")

    if not config.sessions:
        raise ValueError("At least one session is required.")

    if not config.maximum_spreads:
        raise ValueError("At least one spread option is required.")

    experiment_rows: list[pd.DataFrame] = []

    for session, maximum_spread in product(
        config.sessions,
        config.maximum_spreads,
    ):
        filtered = _filter_session(
            predictions,
            session,
            config.session_column,
        )

        filtered = _filter_spread(
            filtered,
            maximum_spread,
            config.spread_column,
        )

        results = evaluate_probability_thresholds(
            filtered,
            config.thresholds,
            probability_column=config.probability_column,
            result_column=config.result_column,
            holdout_column=config.holdout_column,
            minimum_trades=config.minimum_trades,
        )

        results.insert(0, "session", session)
        results.insert(1, "maximum_spread_pips", maximum_spread)
        results["filtered_rows"] = len(filtered)

        experiment_rows.append(results)

    experiments = pd.concat(
        experiment_rows,
        ignore_index=True,
    )

    eligible = rank_experiments(
        experiments,
        eligible_only=True,
    )

    if not eligible.empty:
        eligible["global_rank"] = eligible["rank"]
        eligible = eligible.drop(columns=["rank"])

    ineligible = experiments.loc[~experiments["eligible"]].copy()

    ineligible = ineligible.sort_values(
        [
            "trades",
            "expectancy_r",
            "net_r",
        ],
        ascending=[
            False,
            False,
            False,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ineligible["global_rank"] = pd.NA

    combined = pd.concat(
        [
            eligible,
            ineligible,
        ],
        ignore_index=True,
        sort=False,
    )

    ordered_columns = [
        "global_rank",
        "session",
        "maximum_spread_pips",
        "probability_threshold",
        "eligible",
        "filtered_rows",
        "trades",
        "wins",
        "losses",
        "breakeven",
        "win_rate",
        "gross_profit_r",
        "gross_loss_r",
        "net_r",
        "expectancy_r",
        "profit_factor",
        "max_drawdown_r",
        "longest_losing_streak",
    ]

    return combined[ordered_columns]
