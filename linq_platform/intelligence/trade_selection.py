"""
Native trade-selection engine.

The validated Phase 4 strategy selects a candidate when:

1. It belongs to the holdout period.
2. It has a valid probability estimate.
3. Its predicted probability meets or exceeds the configured threshold.

This module keeps that behavior exact while providing an extensible API for
future risk, exposure, session, and portfolio rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


__all__ = [
    "SelectionConfig",
    "TradeSelector",
    "select_trades",
]


class SelectionConfig(Protocol):
    """Minimum configuration required by the selector."""

    probability_threshold: float


@dataclass(frozen=True)
class TradeSelector:
    """
    Select scored trade candidates using the validated probability rule.

    Parameters
    ----------
    probability_threshold:
        Minimum predicted probability required to select a candidate.
    probability_column:
        Column containing model probabilities.
    holdout_column:
        Boolean column identifying out-of-sample rows.
    selected_column:
        Output column written by :meth:`apply`.
    """

    probability_threshold: float
    probability_column: str = "probability_1r"
    holdout_column: str = "holdout"
    selected_column: str = "selected"

    def selection_mask(self, candidates: pd.DataFrame) -> pd.Series:
        """
        Return a Boolean mask identifying selected candidates.

        This reproduces the validated Phase 4 selection expression.
        """

        self._validate_columns(candidates)

        probabilities = pd.to_numeric(
            candidates[self.probability_column],
            errors="coerce",
        )

        holdout = candidates[self.holdout_column].fillna(False).astype(bool)

        return holdout & probabilities.notna() & (probabilities >= self.probability_threshold)

    def apply(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of the candidates with the selection column populated.
        """

        result = candidates.copy()
        result[self.selected_column] = self.selection_mask(result)
        return result

    def selected_trades(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Return only selected candidates as an independent DataFrame."""

        scored = self.apply(candidates)
        return scored.loc[scored[self.selected_column]].copy()

    def _validate_columns(self, candidates: pd.DataFrame) -> None:
        required = {
            self.probability_column,
            self.holdout_column,
        }

        missing = sorted(required.difference(candidates.columns))

        if missing:
            raise ValueError("Trade selection requires missing column(s): " + ", ".join(missing))


def select_trades(
    candidates: pd.DataFrame,
    cfg: SelectionConfig,
) -> pd.DataFrame:
    """
    Apply the validated Phase 4 trade-selection rule.

    This functional interface keeps pipeline integration simple while
    :class:`TradeSelector` provides the extensible object-oriented API.
    """

    selector = TradeSelector(
        probability_threshold=float(cfg.probability_threshold),
    )

    return selector.apply(candidates)
