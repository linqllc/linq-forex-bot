from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    trades_per_simulation: int
    median_net_r: float
    percentile_05_net_r: float
    percentile_95_net_r: float
    median_max_drawdown_r: float
    percentile_95_max_drawdown_r: float
    probability_of_loss: float
    probability_drawdown_over_10r: float


def maximum_drawdown(values: np.ndarray) -> float:
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    padded_equity = np.concatenate(([0.0], equity))
    drawdowns = padded_equity - peaks
    return float(abs(drawdowns.min()))


def simulate(
    trade_results: pd.Series,
    simulations: int = 5000,
    trades_per_simulation: int | None = None,
    seed: int = 42,
) -> tuple[MonteCarloResult, pd.DataFrame]:
    values = pd.to_numeric(
        trade_results,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(values) < 5:
        raise ValueError(
            "At least five completed trades are required "
            "for Monte Carlo analysis."
        )

    count = trades_per_simulation or len(values)
    random = np.random.default_rng(seed)

    net_results: list[float] = []
    drawdowns: list[float] = []

    for simulation_number in range(simulations):
        sample = random.choice(values, size=count, replace=True)
        net_results.append(float(sample.sum()))
        drawdowns.append(maximum_drawdown(sample))

    details = pd.DataFrame(
        {
            "simulation": np.arange(1, simulations + 1),
            "net_r": net_results,
            "max_drawdown_r": drawdowns,
        }
    )

    result = MonteCarloResult(
        simulations=simulations,
        trades_per_simulation=count,
        median_net_r=float(details["net_r"].median()),
        percentile_05_net_r=float(
            details["net_r"].quantile(0.05)
        ),
        percentile_95_net_r=float(
            details["net_r"].quantile(0.95)
        ),
        median_max_drawdown_r=float(
            details["max_drawdown_r"].median()
        ),
        percentile_95_max_drawdown_r=float(
            details["max_drawdown_r"].quantile(0.95)
        ),
        probability_of_loss=float(
            (details["net_r"] < 0).mean()
        ),
        probability_drawdown_over_10r=float(
            (details["max_drawdown_r"] >= 10).mean()
        ),
    )

    return result, details
