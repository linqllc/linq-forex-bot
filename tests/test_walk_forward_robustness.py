from __future__ import annotations

import pandas as pd
import pytest

from linq_platform.research.walk_forward import (
    RobustnessConfig,
    evaluate_walk_forward_robustness,
    rank_robust_experiments,
)


def window_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": [
                "stable",
                "stable",
                "stable",
                "unstable",
                "unstable",
                "unstable",
                "weak",
                "weak",
                "weak",
            ],
            "window_id": [
                "w1",
                "w2",
                "w3",
                "w1",
                "w2",
                "w3",
                "w1",
                "w2",
                "w3",
            ],
            "trades": [
                20,
                22,
                21,
                20,
                4,
                21,
                15,
                16,
                14,
            ],
            "net_r": [
                4.0,
                3.0,
                5.0,
                12.0,
                -3.0,
                -4.0,
                -1.0,
                1.0,
                -0.5,
            ],
            "expectancy_r": [
                0.20,
                0.14,
                0.24,
                0.60,
                -0.75,
                -0.19,
                -0.07,
                0.06,
                -0.04,
            ],
            "max_drawdown_r": [
                2.0,
                2.5,
                1.5,
                4.0,
                3.0,
                6.0,
                3.0,
                2.0,
                2.5,
            ],
        }
    )


def test_robustness_aggregates_experiments() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=3,
            minimum_trades_per_window=5,
        ),
    )

    assert set(results["experiment_id"]) == {
        "stable",
        "unstable",
        "weak",
    }

    stable = results.loc[results["experiment_id"] == "stable"].iloc[0]

    assert stable["windows"] == 3
    assert stable["eligible_windows"] == 3
    assert stable["total_trades"] == 63
    assert stable["profitable_windows"] == 3
    assert stable["profitable_window_ratio"] == pytest.approx(1.0)
    assert stable["average_expectancy_r"] == pytest.approx((0.20 + 0.14 + 0.24) / 3)


def test_low_trade_windows_are_excluded() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=2,
            minimum_trades_per_window=5,
        ),
    )

    unstable = results.loc[results["experiment_id"] == "unstable"].iloc[0]

    assert unstable["windows"] == 3
    assert unstable["eligible_windows"] == 2
    assert unstable["total_trades"] == 41


def test_stable_experiment_is_eligible() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=3,
            minimum_trades_per_window=5,
            minimum_profitable_window_ratio=0.67,
            minimum_average_expectancy_r=0.10,
        ),
    )

    stable = results.loc[results["experiment_id"] == "stable"].iloc[0]

    assert bool(stable["eligible"])


def test_weak_experiment_is_ineligible() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=3,
            minimum_trades_per_window=5,
            minimum_profitable_window_ratio=0.67,
            minimum_average_expectancy_r=0.0,
        ),
    )

    weak = results.loc[results["experiment_id"] == "weak"].iloc[0]

    assert not bool(weak["eligible"])


def test_ranking_places_stable_experiment_first() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=2,
            minimum_trades_per_window=5,
            minimum_profitable_window_ratio=0.0,
            minimum_average_expectancy_r=-1.0,
        ),
    )

    ranked = rank_robust_experiments(results)

    assert ranked.iloc[0]["experiment_id"] == ("stable")

    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_ranking_can_include_ineligible_rows() -> None:
    results = evaluate_walk_forward_robustness(
        window_results(),
        RobustnessConfig(
            minimum_windows=3,
            minimum_trades_per_window=5,
            minimum_profitable_window_ratio=1.0,
            minimum_average_expectancy_r=0.5,
        ),
    )

    ranked = rank_robust_experiments(
        results,
        eligible_only=False,
    )

    assert len(ranked) == 3


@pytest.mark.parametrize(
    "config",
    [
        RobustnessConfig(minimum_windows=0),
        RobustnessConfig(minimum_trades_per_window=0),
        RobustnessConfig(minimum_profitable_window_ratio=-0.1),
        RobustnessConfig(minimum_profitable_window_ratio=1.1),
    ],
)
def test_invalid_config_raises(
    config: RobustnessConfig,
) -> None:
    with pytest.raises(ValueError):
        evaluate_walk_forward_robustness(
            window_results(),
            config,
        )


def test_missing_columns_raise_clear_error() -> None:
    frame = window_results().drop(columns=["expectancy_r"])

    with pytest.raises(
        ValueError,
        match="expectancy_r",
    ):
        evaluate_walk_forward_robustness(frame)
