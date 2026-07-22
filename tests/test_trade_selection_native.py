from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from linq_platform.intelligence.trade_selection import (
    TradeSelector,
    select_trades,
)


def scored_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setup_id": [
                "train_high",
                "holdout_missing",
                "holdout_below",
                "holdout_equal",
                "holdout_above",
                "holdout_invalid",
            ],
            "holdout": [
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            "probability_1r": [
                0.99,
                np.nan,
                0.6499,
                0.65,
                0.82,
                "invalid",
            ],
            "actual_win": [
                1,
                0,
                0,
                1,
                1,
                0,
            ],
        }
    )


def reference_selection(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    """
    Validated Phase 4 selection expression from predict_holdout().
    """

    probability = pd.to_numeric(
        frame["probability_1r"],
        errors="coerce",
    )

    return (
        frame["holdout"].fillna(False).astype(bool)
        & probability.notna()
        & (probability >= threshold)
    )


def test_selector_matches_validated_phase4_rule() -> None:
    candidates = scored_candidates()
    selector = TradeSelector(probability_threshold=0.65)

    native = selector.selection_mask(candidates)
    expected = reference_selection(candidates, 0.65)

    assert_series_equal(
        native,
        expected,
        check_names=False,
    )

    assert native.tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]


def test_apply_preserves_input_and_adds_selected_column() -> None:
    candidates = scored_candidates()
    original = candidates.copy(deep=True)

    selector = TradeSelector(probability_threshold=0.65)
    result = selector.apply(candidates)

    assert_frame_equal(candidates, original)
    assert "selected" not in candidates.columns
    assert "selected" in result.columns

    expected = reference_selection(result, 0.65)

    assert_series_equal(
        result["selected"],
        expected,
        check_names=False,
    )


def test_functional_interface_uses_config_threshold() -> None:
    candidates = scored_candidates()
    cfg = SimpleNamespace(probability_threshold=0.80)

    result = select_trades(candidates, cfg)

    expected = reference_selection(candidates, 0.80)

    assert_series_equal(
        result["selected"],
        expected,
        check_names=False,
    )

    assert result.loc[
        result["selected"],
        "setup_id",
    ].tolist() == ["holdout_above"]


def test_selected_trades_returns_only_approved_rows() -> None:
    candidates = scored_candidates()
    selector = TradeSelector(probability_threshold=0.65)

    selected = selector.selected_trades(candidates)

    assert selected["setup_id"].tolist() == [
        "holdout_equal",
        "holdout_above",
    ]

    assert selected["selected"].all()


def test_custom_column_names_are_supported() -> None:
    candidates = pd.DataFrame(
        {
            "is_test": [False, True, True],
            "score": [0.99, 0.70, 0.40],
        }
    )

    selector = TradeSelector(
        probability_threshold=0.65,
        probability_column="score",
        holdout_column="is_test",
        selected_column="approved",
    )

    result = selector.apply(candidates)

    assert result["approved"].tolist() == [
        False,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "missing_column",
    [
        "holdout",
        "probability_1r",
    ],
)
def test_missing_required_columns_raise_clear_error(
    missing_column: str,
) -> None:
    candidates = scored_candidates().drop(columns=[missing_column])
    selector = TradeSelector(probability_threshold=0.65)

    with pytest.raises(
        ValueError,
        match=missing_column,
    ):
        selector.apply(candidates)
