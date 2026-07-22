from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from linq_platform.intelligence import pipeline


def make_dataset() -> pd.DataFrame:
    """
    Create enough deterministic rows to satisfy the native pipeline's
    minimum-training requirement and fixed holdout boundary.
    """

    training_times = pd.date_range(
        "2026-04-01 00:00:00+00:00",
        periods=45,
        freq="6h",
    )

    holdout_times = pd.date_range(
        pipeline.HOLDOUT_START,
        periods=4,
        freq="6h",
    )

    timestamps = training_times.append(holdout_times)
    row_count = len(timestamps)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "setup_id": [f"setup_{index:03d}" for index in range(row_count)],
            "direction": ["long" if index % 2 == 0 else "short" for index in range(row_count)],
            "actual_win": [index % 2 for index in range(row_count)],
            "entry_price": [1.1000 + index * 0.0001 for index in range(row_count)],
            "stop_price": [1.0990 + index * 0.0001 for index in range(row_count)],
            "target_price": [1.1010 + index * 0.0001 for index in range(row_count)],
            "spread_pips_at_entry": [1.0] * row_count,
            "gross_result_r": [1.0 if index % 2 else -1.0 for index in range(row_count)],
            "total_cost_r": [0.05] * row_count,
            "net_result_r": [0.95 if index % 2 else -1.05 for index in range(row_count)],
            "exit_reason": ["target" if index % 2 else "stop" for index in range(row_count)],
            "bars_held": [5] * row_count,
        }
    )


def make_exclusion_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setup_id": [
                "setup_001",
                "excluded_001",
            ],
            "included": [
                True,
                False,
            ],
            "exclusion_reason": [
                "included",
                "spread_filter",
            ],
        }
    )


def test_native_phase4_pipeline_end_to_end(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = make_dataset()
    exclusion_log = make_exclusion_log()

    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01",
                periods=10,
                freq="h",
                tz="UTC",
            ),
            "close": [1.10] * 10,
        }
    )

    setups = pd.DataFrame({"setup_id": [f"source_{index:03d}" for index in range(60)]})

    phase1 = pd.DataFrame(
        {
            "setup_id": setups["setup_id"],
        }
    )

    monkeypatch.setattr(
        pipeline,
        "load_candles",
        lambda _path: candles.copy(),
    )
    monkeypatch.setattr(
        pipeline,
        "load_setups",
        lambda _path: setups.copy(),
    )
    monkeypatch.setattr(
        pipeline,
        "load_phase1",
        lambda _path: phase1.copy(),
    )

    def fake_build_controlled_dataset(
        *,
        candles,
        setups,
        phase1,
        config,
    ):
        assert not candles.empty
        assert not setups.empty
        assert not phase1.empty
        assert config.probability_threshold == 0.65

        return (
            dataset.copy(),
            exclusion_log.copy(),
        )

    monkeypatch.setattr(
        pipeline,
        "build_controlled_dataset",
        fake_build_controlled_dataset,
    )

    def fake_predict_holdout(
        input_dataset,
        holdout_start,
        config,
    ):
        result = input_dataset.copy()

        expected_start = int((result["timestamp"] < pipeline.HOLDOUT_START).sum())

        assert holdout_start == expected_start
        assert config.minimum_training_rows == 40

        result["holdout"] = False
        result.loc[
            result.index[holdout_start:],
            "holdout",
        ] = True

        result["probability_1r"] = pd.NA

        holdout_indexes = result.index[holdout_start:]

        probabilities = [
            0.40,
            0.65,
            0.80,
            0.55,
        ]

        result.loc[
            holdout_indexes,
            "probability_1r",
        ] = probabilities

        result["probability_1r"] = pd.to_numeric(
            result["probability_1r"],
            errors="coerce",
        )

        result["selected"] = (
            result["holdout"]
            & result["probability_1r"].notna()
            & (result["probability_1r"] >= config.probability_threshold)
        )

        return result

    monkeypatch.setattr(
        pipeline,
        "predict_holdout",
        fake_predict_holdout,
    )

    def fake_summarize(
        name,
        trades,
        prediction_rows=None,
    ):
        return {
            "strategy": name,
            "trades": int(len(trades)),
            "wins": int((trades["net_result_r"] > 0).sum()),
            "losses": int((trades["net_result_r"] < 0).sum()),
            "win_rate": (float((trades["net_result_r"] > 0).mean()) if len(trades) else 0.0),
            "gross_r": float(trades["gross_result_r"].sum()),
            "cost_r": float(trades["total_cost_r"].sum()),
            "net_r": float(trades["net_result_r"].sum()),
            "expectancy_r": float(trades["net_result_r"].mean()),
            "profit_factor": 1.0,
            "max_drawdown_r": 0.0,
            "longest_losing_streak": 0,
            "auc": 0.5,
            "brier": 0.25,
        }

    monkeypatch.setattr(
        pipeline,
        "summarize",
        fake_summarize,
    )

    def fake_equity_curve(trades):
        result = trades.copy()
        result["trade_number"] = range(
            1,
            len(result) + 1,
        )
        result["equity_r"] = result["net_result_r"].cumsum()
        result["drawdown_r"] = 0.0
        return result

    monkeypatch.setattr(
        pipeline,
        "equity_curve",
        fake_equity_curve,
    )

    monkeypatch.setattr(
        pipeline,
        "calibration_table",
        lambda _predictions: pd.DataFrame(
            {
                "bucket": ["0.6-0.8"],
                "count": [2],
            }
        ),
    )

    def fake_build_report(
        *,
        summaries,
        selected,
        benchmark,
        equity,
        calibration,
        holdout_start,
        cfg,
        path,
    ):
        assert len(summaries) == 2
        assert len(selected) == 2
        assert len(benchmark) == 4
        assert len(equity) == 2
        assert len(calibration) == 1
        assert holdout_start == pipeline.HOLDOUT_START
        assert cfg.probability_threshold == 0.65

        Path(path).write_text(
            "<html><body>integration report</body></html>",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        pipeline,
        "build_report",
        fake_build_report,
    )

    output_dir = tmp_path / "research_output"

    payload = pipeline.run_native_phase4_1(
        candles_path=tmp_path / "candles.csv",
        setups_path=tmp_path / "setups.csv",
        phase1_path=tmp_path / "phase1.csv",
        output_dir=output_dir,
        slippage_pips_each_side=0.15,
    )

    assert payload["engine"] == ("LINQ V10 native intelligence")
    assert payload["original_setups"] == 60
    assert payload["filtered_usable_setups"] == 49
    assert payload["training_rows"] == 45
    assert payload["holdout_rows"] == 4
    assert payload["selected_trades"] == 2

    assert payload["rules"]["slippage_pips_each_side"] == 0.15

    expected_files = {
        "dataset",
        "predictions",
        "selected",
        "exclusions",
        "exclusion_summary",
        "summary",
        "json",
        "report",
    }

    assert set(payload["files"]) == expected_files

    for file_path in payload["files"].values():
        assert Path(file_path).exists()

    predictions = pd.read_csv(payload["files"]["predictions"])
    selected = pd.read_csv(payload["files"]["selected"])

    assert len(predictions) == 49
    assert len(selected) == 2

    assert selected["probability_1r"].tolist() == [
        0.65,
        0.80,
    ]

    summary_payload = json.loads(Path(payload["files"]["json"]).read_text(encoding="utf-8"))

    assert summary_payload["selected_trades"] == 2
    assert summary_payload["training_rows"] == 45
    assert summary_payload["holdout_rows"] == 4
