from pathlib import Path

import pandas as pd

from linq_quant.metadata import (
    classify_session,
    infer_data_source,
    infer_instrument,
    infer_timeframe,
)


def test_source_and_instrument_inference():
    path = Path(
        "EUR_USD_oanda_intelligence_dataset.csv"
    )

    assert infer_data_source(path) == "oanda"
    assert infer_instrument(path) == "EUR_USD"


def test_demo_source_inference():
    path = Path(
        "EUR_USD_synthetic_demo_intelligence_dataset.csv"
    )

    assert infer_data_source(path) == "synthetic_demo"


def test_session_classification():
    assert (
        classify_session("2026-01-01T13:00:00Z")
        == "london_new_york_overlap"
    )
    assert (
        classify_session("2026-01-01T09:00:00Z")
        == "london"
    )
    assert (
        classify_session("2026-01-01T18:00:00Z")
        == "new_york"
    )
    assert (
        classify_session("2026-01-01T01:00:00Z")
        == "asia"
    )


def test_timeframe_inference():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=20,
                freq="5min",
                tz="UTC",
            )
        }
    )

    assert infer_timeframe(frame) == "M5"
