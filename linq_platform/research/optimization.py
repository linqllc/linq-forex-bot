"""
Strategy experiment and ranking tools.

This module evaluates probability thresholds against already-scored,
out-of-sample trade candidates. It does not train models, place trades,
or claim that historical performance guarantees future profitability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable

import numpy as np
import pandas as pd


__all__ = [
    "ExperimentResult",
    "evaluate_probability_thresholds",
    "rank_experiments",
]


@dataclass(frozen=True)
class ExperimentResult:
    """Performance summary for one research configuration."""

    probability_threshold: float
    trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    gross_profit_r: float
    gross_loss_r: float
    net_r: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_r: float
    longest_losing_streak: int
    eligible: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        """Return a serializable representation."""

        return asdict(self)


def _maximum_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    equity = np.concatenate(
        [
            np.array([0.0]),
            values.astype(float).cumsum().to_numpy(),
        ]
    )

    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks

    return float(abs(drawdowns.min()))


def _longest_losing_streak(values: pd.Series) -> int:
    longest = 0
    current = 0

    for value in values.astype(float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def _summarize_threshold(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    probability_column: str,
    result_column: str,
    holdout_column: str,
    minimum_trades: int,
) -> ExperimentResult:
    probabilities = pd.to_numeric(
        predictions[probability_column],
        errors="coerce",
    )

    results = pd.to_numeric(
        predictions[result_column],
        errors="coerce",
    )

    holdout = predictions[holdout_column].fillna(False).astype(bool)

    selected_mask = holdout & probabilities.notna() & results.notna() & (probabilities >= threshold)

    selected_results = results.loc[selected_mask].astype(float)

    trades = int(len(selected_results))
    wins = int((selected_results > 0).sum())
    losses = int((selected_results < 0).sum())
    breakeven = int((selected_results == 0).sum())

    gross_profit = float(selected_results.clip(lower=0).sum())
    gross_loss = float(abs(selected_results.clip(upper=0).sum()))
    net_r = float(selected_results.sum())

    win_rate = wins / trades if trades else 0.0
    expectancy = net_r / trades if trades else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = inf
    else:
        profit_factor = 0.0

    return ExperimentResult(
        probability_threshold=float(threshold),
        trades=trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=float(win_rate),
        gross_profit_r=gross_profit,
        gross_loss_r=gross_loss,
        net_r=net_r,
        expectancy_r=float(expectancy),
        profit_factor=float(profit_factor),
        max_drawdown_r=_maximum_drawdown(selected_results),
        longest_losing_streak=_longest_losing_streak(selected_results),
        eligible=trades >= minimum_trades,
    )


def evaluate_probability_thresholds(
    predictions: pd.DataFrame,
    thresholds: Iterable[float],
    *,
    probability_column: str = "probability_1r",
    result_column: str = "net_result_r",
    holdout_column: str = "holdout",
    minimum_trades: int = 10,
) -> pd.DataFrame:
    """
    Evaluate several probability thresholds on holdout predictions.

    Parameters
    ----------
    predictions:
        Scored candidate rows containing holdout status, predicted
        probability, and realized net result in R.
    thresholds:
        Probability thresholds to evaluate.
    minimum_trades:
        Minimum selected-trade count required for an experiment to be
        marked eligible.

    Returns
    -------
    pandas.DataFrame
        One performance row per unique threshold.
    """

    required = {
        probability_column,
        result_column,
        holdout_column,
    }

    missing = sorted(required.difference(predictions.columns))

    if missing:
        raise ValueError("Threshold research requires missing column(s): " + ", ".join(missing))

    if minimum_trades < 1:
        raise ValueError("minimum_trades must be at least 1.")

    normalized_thresholds = sorted({float(threshold) for threshold in thresholds})

    if not normalized_thresholds:
        raise ValueError("At least one probability threshold is required.")

    invalid = [threshold for threshold in normalized_thresholds if not 0.0 <= threshold <= 1.0]

    if invalid:
        raise ValueError("Probability thresholds must be between 0 and 1.")

    rows = [
        _summarize_threshold(
            predictions,
            threshold,
            probability_column=probability_column,
            result_column=result_column,
            holdout_column=holdout_column,
            minimum_trades=minimum_trades,
        ).to_dict()
        for threshold in normalized_thresholds
    ]

    return pd.DataFrame(rows)


def rank_experiments(
    experiments: pd.DataFrame,
    *,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """
    Rank experiments using transparent historical metrics.

    Eligible experiments are ordered by:

    1. Highest expectancy R
    2. Highest net R
    3. Lowest maximum drawdown
    4. Greater trade count

    The ranking is descriptive research output, not a guarantee of
    future performance.
    """

    required = {
        "eligible",
        "expectancy_r",
        "net_r",
        "max_drawdown_r",
        "trades",
        "probability_threshold",
    }

    missing = sorted(required.difference(experiments.columns))

    if missing:
        raise ValueError("Experiment ranking requires missing column(s): " + ", ".join(missing))

    ranked = experiments.copy()

    if eligible_only:
        ranked = ranked.loc[ranked["eligible"]].copy()

    ranked = ranked.sort_values(
        by=[
            "expectancy_r",
            "net_r",
            "max_drawdown_r",
            "trades",
            "probability_threshold",
        ],
        ascending=[
            False,
            False,
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "rank",
        range(1, len(ranked) + 1),
    )

    return ranked
