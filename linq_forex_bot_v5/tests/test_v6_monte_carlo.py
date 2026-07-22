import pandas as pd

from linq_quant.monte_carlo import maximum_drawdown, simulate


def test_maximum_drawdown():
    assert maximum_drawdown(
        pd.Series([1.0, -1.0, -1.0, 2.0]).to_numpy()
    ) == 2.0


def test_monte_carlo_runs():
    result, details = simulate(
        pd.Series([2.0, -1.0, 1.5, -1.0, 3.0, -1.0]),
        simulations=100,
        seed=1,
    )

    assert result.simulations == 100
    assert len(details) == 100
    assert 0 <= result.probability_of_loss <= 1
