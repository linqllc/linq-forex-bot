from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linq_platform.research.optimization import (
    evaluate_probability_thresholds,
    rank_experiments,
)


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setup_id": [
                "training",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "h7",
                "h8",
            ],
            "holdout": [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "probability_1r": [
                0.99,
                0.51,
                0.58,
                0.62,
                0.66,
                0.71,
                0.76,
                0.81,
                np.nan,
            ],
            "net_result_r": [
                20.0,
                -1.0,
                1.2,
                -0.8,
                1.5,
                -1.0,
                2.0,
                1.0,
                10.0,
            ],
        }
    )


def test_threshold_evaluation_uses_only_holdout_rows() -> None:
    results = evaluate_probability_thresholds(
        predictions(),
        thresholds=[0.50, 0.65, 0.75],
        minimum_trades=2,
    )

    assert results["probability_threshold"].tolist() == [
        0.50,
        0.65,
        0.75,
    ]

    threshold_50 = results.iloc[0]

    assert threshold_50["trades"] == 7
    assert threshold_50["wins"] == 4
    assert threshold_50["losses"] == 3
    assert threshold_50["net_r"] == pytest.approx(2.9)

    assert threshold_50["net_r"] != pytest.approx(22.9)


def test_threshold_metrics_are_calculated_correctly() -> None:
    results = evaluate_probability_thresholds(
        predictions(),
        thresholds=[0.75],
        minimum_trades=2,
    )

    row = results.iloc[0]

    assert row["trades"] == 2
    assert row["wins"] == 2
    assert row["losses"] == 0
    assert row["win_rate"] == pytest.approx(1.0)
    assert row["net_r"] == pytest.approx(3.0)
    assert row["expectancy_r"] == pytest.approx(1.5)
    assert np.isinf(row["profit_factor"])
    assert row["max_drawdown_r"] == pytest.approx(0.0)
    assert row["longest_losing_streak"] == 0
    assert bool(row["eligible"])


def test_minimum_trade_requirement_marks_eligibility() -> None:
    results = evaluate_probability_thresholds(
        predictions(),
        thresholds=[0.75, 0.80],
        minimum_trades=2,
    )

    assert bool(results.iloc[0]["eligible"])
    assert not bool(results.iloc[1]["eligible"])


def test_duplicate_thresholds_are_removed_and_sorted() -> None:
    results = evaluate_probability_thresholds(
        predictions(),
        thresholds=[0.75, 0.50, 0.75, 0.65],
        minimum_trades=1,
    )

    assert results["probability_threshold"].tolist() == [
        0.50,
        0.65,
        0.75,
    ]


def test_ranking_excludes_ineligible_experiments() -> None:
    results = evaluate_probability_thresholds(
        predictions(),
        thresholds=[0.50, 0.65, 0.75, 0.80],
        minimum_trades=2,
    )

    ranked = rank_experiments(results)

    assert ranked["eligible"].all()
    assert 0.80 not in ranked["probability_threshold"].tolist()
    assert ranked.iloc[0]["probability_threshold"] == pytest.approx(0.75)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        [-0.1],
        [1.1],
    ],
)
def test_invalid_thresholds_raise_errors(
    thresholds,
) -> None:
    with pytest.raises(ValueError):
        evaluate_probability_thresholds(
            predictions(),
            thresholds=thresholds,
        )


def test_missing_columns_raise_clear_error() -> None:
    frame = predictions().drop(columns=["net_result_r"])

    with pytest.raises(
        ValueError,
        match="net_result_r",
    ):
        evaluate_probability_thresholds(
            frame,
            thresholds=[0.65],
        )
