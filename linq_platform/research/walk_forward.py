"""
Walk-forward robustness analysis.

This module evaluates already-generated experiment results across multiple
chronological validation windows. It does not train models or place trades.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


__all__ = [
    "RobustnessConfig",
    "RobustnessResult",
    "evaluate_walk_forward_robustness",
    "rank_robust_experiments",
]


@dataclass(frozen=True)
class RobustnessConfig:
    """Rules used to determine whether an experiment is robust."""

    minimum_windows: int = 3
    minimum_trades_per_window: int = 5
    minimum_profitable_window_ratio: float = 0.50
    minimum_average_expectancy_r: float = 0.0


@dataclass(frozen=True)
class RobustnessResult:
    """Aggregated performance for one experiment across many windows."""

    experiment_id: str
    windows: int
    eligible_windows: int
    total_trades: int
    average_trades_per_window: float
    minimum_trades_in_window: int
    maximum_trades_in_window: int
    profitable_windows: int
    profitable_window_ratio: float
    average_expectancy_r: float
    median_expectancy_r: float
    worst_expectancy_r: float
    best_expectancy_r: float
    average_net_r: float
    total_net_r: float
    worst_net_r: float
    average_max_drawdown_r: float
    worst_max_drawdown_r: float
    expectancy_std_r: float
    trade_count_cv: float
    robustness_score: float
    eligible: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return a serializable representation."""

        return asdict(self)


_REQUIRED_COLUMNS = {
    "experiment_id",
    "window_id",
    "trades",
    "net_r",
    "expectancy_r",
    "max_drawdown_r",
}


def _coefficient_of_variation(values: pd.Series) -> float:
    mean = float(values.mean())

    if mean == 0.0:
        return 0.0

    return float(values.std(ddof=0) / abs(mean))


def _robustness_score(
    *,
    profitable_window_ratio: float,
    average_expectancy_r: float,
    worst_expectancy_r: float,
    expectancy_std_r: float,
    trade_count_cv: float,
    worst_max_drawdown_r: float,
) -> float:
    """
    Produce a transparent consistency-oriented score.

    Higher values reward:
    - profitable performance across more windows;
    - positive average and worst-window expectancy.

    Lower values penalize:
    - unstable expectancy;
    - inconsistent trade counts;
    - larger worst-window drawdown.
    """

    score = (
        profitable_window_ratio * 40.0
        + average_expectancy_r * 25.0
        + worst_expectancy_r * 15.0
        - expectancy_std_r * 10.0
        - trade_count_cv * 5.0
        - worst_max_drawdown_r * 2.0
    )

    return float(score)


def _summarize_experiment(
    experiment_id: str,
    rows: pd.DataFrame,
    config: RobustnessConfig,
) -> RobustnessResult:
    trades = pd.to_numeric(
        rows["trades"],
        errors="coerce",
    ).fillna(0)

    net_r = pd.to_numeric(
        rows["net_r"],
        errors="coerce",
    ).fillna(0.0)

    expectancy = pd.to_numeric(
        rows["expectancy_r"],
        errors="coerce",
    ).fillna(0.0)

    drawdown = pd.to_numeric(
        rows["max_drawdown_r"],
        errors="coerce",
    ).fillna(0.0)

    eligible_window_mask = trades >= config.minimum_trades_per_window

    eligible_rows = rows.loc[eligible_window_mask].copy()

    eligible_trades = trades.loc[eligible_window_mask]
    eligible_net_r = net_r.loc[eligible_window_mask]
    eligible_expectancy = expectancy.loc[eligible_window_mask]
    eligible_drawdown = drawdown.loc[eligible_window_mask]

    windows = int(rows["window_id"].nunique())
    eligible_windows = int(len(eligible_rows))

    if eligible_windows:
        profitable_windows = int((eligible_net_r > 0).sum())

        profitable_ratio = profitable_windows / eligible_windows

        average_expectancy = float(eligible_expectancy.mean())
        median_expectancy = float(eligible_expectancy.median())
        worst_expectancy = float(eligible_expectancy.min())
        best_expectancy = float(eligible_expectancy.max())

        average_net = float(eligible_net_r.mean())
        total_net = float(eligible_net_r.sum())
        worst_net = float(eligible_net_r.min())

        average_drawdown = float(eligible_drawdown.mean())
        worst_drawdown = float(eligible_drawdown.max())

        expectancy_std = float(eligible_expectancy.std(ddof=0))

        trade_count_cv = _coefficient_of_variation(eligible_trades)

        minimum_trades = int(eligible_trades.min())
        maximum_trades = int(eligible_trades.max())
        total_trades = int(eligible_trades.sum())
        average_trades = float(eligible_trades.mean())
    else:
        profitable_windows = 0
        profitable_ratio = 0.0
        average_expectancy = 0.0
        median_expectancy = 0.0
        worst_expectancy = 0.0
        best_expectancy = 0.0
        average_net = 0.0
        total_net = 0.0
        worst_net = 0.0
        average_drawdown = 0.0
        worst_drawdown = 0.0
        expectancy_std = 0.0
        trade_count_cv = 0.0
        minimum_trades = 0
        maximum_trades = 0
        total_trades = 0
        average_trades = 0.0

    score = _robustness_score(
        profitable_window_ratio=profitable_ratio,
        average_expectancy_r=average_expectancy,
        worst_expectancy_r=worst_expectancy,
        expectancy_std_r=expectancy_std,
        trade_count_cv=trade_count_cv,
        worst_max_drawdown_r=worst_drawdown,
    )

    eligible = (
        eligible_windows >= config.minimum_windows
        and profitable_ratio >= config.minimum_profitable_window_ratio
        and average_expectancy >= config.minimum_average_expectancy_r
    )

    return RobustnessResult(
        experiment_id=str(experiment_id),
        windows=windows,
        eligible_windows=eligible_windows,
        total_trades=total_trades,
        average_trades_per_window=average_trades,
        minimum_trades_in_window=minimum_trades,
        maximum_trades_in_window=maximum_trades,
        profitable_windows=profitable_windows,
        profitable_window_ratio=float(profitable_ratio),
        average_expectancy_r=average_expectancy,
        median_expectancy_r=median_expectancy,
        worst_expectancy_r=worst_expectancy,
        best_expectancy_r=best_expectancy,
        average_net_r=average_net,
        total_net_r=total_net,
        worst_net_r=worst_net,
        average_max_drawdown_r=average_drawdown,
        worst_max_drawdown_r=worst_drawdown,
        expectancy_std_r=expectancy_std,
        trade_count_cv=trade_count_cv,
        robustness_score=score,
        eligible=eligible,
    )


def evaluate_walk_forward_robustness(
    window_results: pd.DataFrame,
    config: RobustnessConfig | None = None,
) -> pd.DataFrame:
    """
    Aggregate experiment performance across walk-forward windows.

    Input rows must contain one row per experiment and validation window.
    """

    config = config or RobustnessConfig()

    missing = sorted(_REQUIRED_COLUMNS.difference(window_results.columns))

    if missing:
        raise ValueError(
            "Walk-forward robustness requires missing column(s): " + ", ".join(missing)
        )

    if config.minimum_windows < 1:
        raise ValueError("minimum_windows must be at least 1.")

    if config.minimum_trades_per_window < 1:
        raise ValueError("minimum_trades_per_window must be at least 1.")

    if not 0.0 <= (config.minimum_profitable_window_ratio) <= 1.0:
        raise ValueError("minimum_profitable_window_ratio must be between 0 and 1.")

    summaries = [
        _summarize_experiment(
            str(experiment_id),
            group,
            config,
        ).to_dict()
        for experiment_id, group in window_results.groupby(
            "experiment_id",
            sort=True,
        )
    ]

    return pd.DataFrame(summaries)


def rank_robust_experiments(
    robustness: pd.DataFrame,
    *,
    eligible_only: bool = True,
) -> pd.DataFrame:
    """Rank experiments by robustness and risk-adjusted consistency."""

    required = {
        "experiment_id",
        "eligible",
        "robustness_score",
        "profitable_window_ratio",
        "average_expectancy_r",
        "worst_expectancy_r",
        "worst_max_drawdown_r",
        "total_trades",
    }

    missing = sorted(required.difference(robustness.columns))

    if missing:
        raise ValueError("Robustness ranking requires missing column(s): " + ", ".join(missing))

    ranked = robustness.copy()

    if eligible_only:
        ranked = ranked.loc[ranked["eligible"]].copy()

    ranked = ranked.sort_values(
        by=[
            "robustness_score",
            "profitable_window_ratio",
            "average_expectancy_r",
            "worst_expectancy_r",
            "worst_max_drawdown_r",
            "total_trades",
            "experiment_id",
        ],
        ascending=[
            False,
            False,
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
