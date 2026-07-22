import pandas as pd

from src.setup_outcomes import OutcomeConfig, analyze_outcomes


def test_long_setup_hits_targets_before_stop():
    candles = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:10:00",
                    "2026-01-01 00:15:00",
                ],
                utc=True,
            ),
            "open": [1.1000, 1.1000, 1.1010, 1.1020],
            "high": [1.1005, 1.1012, 1.1022, 1.1032],
            "low": [1.0995, 1.0998, 1.1005, 1.1015],
            "close": [1.1000, 1.1010, 1.1020, 1.1030],
        }
    )

    setups = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "direction": ["long"],
            "entry": [1.1000],
            "stop": [1.0990],
        }
    )

    results = analyze_outcomes(candles, setups)

    assert len(results) == 1
    assert bool(results.loc[0, "hit_1r"]) is True
    assert bool(results.loc[0, "hit_2r"]) is True
    assert bool(results.loc[0, "hit_3r"]) is True
    assert bool(results.loc[0, "stop_hit"]) is False
    assert results.loc[0, "outcome"] == "hit_3r"


def test_short_setup_stops_out():
    candles = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:05:00",
                    "2026-01-01 00:10:00",
                ],
                utc=True,
            ),
            "open": [1.1000, 1.1000, 1.1010],
            "high": [1.1005, 1.1012, 1.1020],
            "low": [1.0995, 1.0998, 1.1005],
            "close": [1.1000, 1.1010, 1.1015],
        }
    )

    setups = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "direction": ["short"],
            "entry": [1.1000],
            "stop": [1.1010],
        }
    )

    results = analyze_outcomes(candles, setups)

    assert bool(results.loc[0, "stop_hit"]) is True
    assert bool(results.loc[0, "hit_1r"]) is False
    assert results.loc[0, "outcome"] == "stopped"


def test_same_bar_conflict_defaults_to_stop_first():
    candles = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 00:05:00",
                ],
                utc=True,
            ),
            "open": [1.1000, 1.1000],
            "high": [1.1005, 1.1015],
            "low": [1.0995, 1.0985],
            "close": [1.1000, 1.1002],
        }
    )

    setups = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")],
            "direction": ["long"],
            "entry": [1.1000],
            "stop": [1.0990],
        }
    )

    config = OutcomeConfig(ambiguous_policy="stop_first")
    results = analyze_outcomes(candles, setups, config)

    assert bool(results.loc[0, "stop_hit"]) is True
    assert bool(results.loc[0, "hit_1r"]) is False
    assert results.loc[0, "outcome"] == "ambiguous_loss"
