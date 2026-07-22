from __future__ import annotations

import pandas as pd
import pytest

from linq_platform.research.experiment_grid import (
    ExperimentGridConfig,
    build_parameter_grid,
    run_experiment_grid,
)


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "holdout": [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            "probability_1r": [
                0.99,
                0.55,
                0.62,
                0.68,
                0.72,
                0.78,
                0.85,
            ],
            "net_result_r": [
                50.0,
                -1.0,
                1.0,
                -0.5,
                1.5,
                -1.0,
                2.0,
            ],
            "session": [
                "training",
                "london",
                "london",
                "new_york",
                "new_york",
                "london",
                "new_york",
            ],
            "spread_pips_at_entry": [
                0.5,
                0.8,
                1.1,
                0.9,
                1.4,
                1.8,
                0.7,
            ],
        }
    )


def test_parameter_grid_builds_cartesian_product() -> None:
    grid = build_parameter_grid(
        {
            "stop_atr": [1.0, 1.5],
            "target_r": [1.0, 2.0, 3.0],
        }
    )

    assert len(grid) == 6
    assert {"stop_atr": 1.0, "target_r": 1.0} in grid
    assert {"stop_atr": 1.5, "target_r": 3.0} in grid


def test_experiment_grid_evaluates_all_combinations() -> None:
    config = ExperimentGridConfig(
        thresholds=(0.60, 0.70),
        sessions=("all", "london", "new_york"),
        maximum_spreads=(None, 1.0),
        minimum_trades=1,
    )

    leaderboard = run_experiment_grid(
        predictions(),
        config,
    )

    assert len(leaderboard) == 12
    assert set(leaderboard["session"]) == {
        "all",
        "london",
        "new_york",
    }
    assert set(leaderboard["probability_threshold"]) == {
        0.60,
        0.70,
    }


def test_grid_uses_only_holdout_results() -> None:
    config = ExperimentGridConfig(
        thresholds=(0.50,),
        sessions=("all",),
        maximum_spreads=(None,),
        minimum_trades=1,
    )

    leaderboard = run_experiment_grid(
        predictions(),
        config,
    )

    assert leaderboard.iloc[0]["net_r"] != pytest.approx(52.0)


def test_session_filter_is_applied() -> None:
    config = ExperimentGridConfig(
        thresholds=(0.60,),
        sessions=("london",),
        maximum_spreads=(None,),
        minimum_trades=1,
    )

    leaderboard = run_experiment_grid(
        predictions(),
        config,
    )

    row = leaderboard.iloc[0]

    assert row["filtered_rows"] == 3
    assert row["trades"] == 2
    assert row["net_r"] == pytest.approx(0.0)


def test_spread_filter_is_applied() -> None:
    config = ExperimentGridConfig(
        thresholds=(0.50,),
        sessions=("all",),
        maximum_spreads=(1.0,),
        minimum_trades=1,
    )

    leaderboard = run_experiment_grid(
        predictions(),
        config,
    )

    row = leaderboard.iloc[0]

    assert row["filtered_rows"] == 4
    assert row["trades"] == 3


def test_missing_session_column_raises() -> None:
    frame = predictions().drop(columns=["session"])

    config = ExperimentGridConfig(
        thresholds=(0.60,),
        sessions=("london",),
    )

    with pytest.raises(ValueError, match="session"):
        run_experiment_grid(frame, config)


def test_missing_spread_column_raises() -> None:
    frame = predictions().drop(columns=["spread_pips_at_entry"])

    config = ExperimentGridConfig(
        thresholds=(0.60,),
        maximum_spreads=(1.0,),
    )

    with pytest.raises(ValueError, match="spread"):
        run_experiment_grid(frame, config)


def test_empty_parameter_options_raise() -> None:
    with pytest.raises(ValueError):
        build_parameter_grid(
            {
                "stop_atr": [],
                "target_r": [1.0],
            }
        )
