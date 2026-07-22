from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Phase4Config
from .controlled_rules import (
    HOLDOUT_START,
    MAXIMUM_SPREAD_PIPS,
    MAXIMUM_SPREAD_TO_STOP_RATIO,
    MINIMUM_STOP_PIPS,
    build_controlled_dataset,
    calibration_table,
)
from .data import load_candles, load_phase1, load_setups
from .probability_model import predict_holdout
from .reporting import build_report, equity_curve, summarize


def run_native_phase4_1(
    candles_path: str | Path,
    setups_path: str | Path,
    phase1_path: str | Path,
    output_dir: str | Path,
    slippage_pips_each_side: float = 0.10,
) -> dict[str, Any]:
    """Generate Phase 4.1 signals natively inside LINQ V10."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Phase4Config(
        holdout_trades=20,
        slippage_pips_each_side=slippage_pips_each_side,
        maximum_spread_pips=None,
    )

    candles = load_candles(candles_path)
    setups = load_setups(setups_path)
    phase1 = load_phase1(phase1_path)

    dataset, exclusion_log = build_controlled_dataset(
        candles=candles,
        setups=setups,
        phase1=phase1,
        config=config,
    )
    if dataset.empty:
        raise RuntimeError("No usable setups remained after filters.")

    holdout_mask = dataset["timestamp"] >= HOLDOUT_START
    training_rows = int((~holdout_mask).sum())
    holdout_rows = int(holdout_mask.sum())
    if training_rows < config.minimum_training_rows:
        raise RuntimeError(
            f"Only {training_rows} filtered training setups exist before "
            f"fixed holdout start; need {config.minimum_training_rows}."
        )
    if holdout_rows == 0:
        raise RuntimeError("No filtered setups exist in the fixed holdout.")

    holdout_start_index = int(np.flatnonzero(holdout_mask.to_numpy())[0])
    predictions = predict_holdout(dataset, holdout_start_index, config)
    holdout = predictions[predictions["timestamp"] >= HOLDOUT_START].copy()
    holdout["holdout"] = True
    selected = holdout[holdout["selected"]].copy()

    summaries = [
        summarize("model_selected_filtered", selected, holdout),
        summarize("all_filtered_holdout_setups", holdout),
    ]
    equity = equity_curve(selected)
    calibration = calibration_table(predictions)
    exclusion_summary = (
        exclusion_log.groupby(["included", "exclusion_reason"], dropna=False)
        .size()
        .reset_index(name="setups")
        .sort_values(["included", "setups"], ascending=[False, False])
    )

    paths = {
        "dataset": output_dir / "EUR_USD_phase4_1_dataset.csv",
        "predictions": output_dir / "EUR_USD_phase4_1_predictions.csv",
        "selected": output_dir / "EUR_USD_phase4_1_selected_trades.csv",
        "exclusions": output_dir / "EUR_USD_phase4_1_exclusion_log.csv",
        "exclusion_summary": output_dir / "EUR_USD_phase4_1_exclusion_summary.csv",
        "summary": output_dir / "EUR_USD_phase4_1_summary.csv",
        "json": output_dir / "EUR_USD_phase4_1_summary.json",
        "report": output_dir / "EUR_USD_phase4_1_report.html",
    }
    dataset.to_csv(paths["dataset"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    selected.to_csv(paths["selected"], index=False)
    exclusion_log.to_csv(paths["exclusions"], index=False)
    exclusion_summary.to_csv(paths["exclusion_summary"], index=False)
    pd.DataFrame(summaries).to_csv(paths["summary"], index=False)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "LINQ V10 native intelligence",
        "fixed_holdout_start": str(HOLDOUT_START),
        "rules": {
            "threshold": config.probability_threshold,
            "stop_atr": config.stop_atr,
            "minimum_stop_pips": MINIMUM_STOP_PIPS,
            "maximum_spread_pips": MAXIMUM_SPREAD_PIPS,
            "maximum_spread_to_stop_ratio": MAXIMUM_SPREAD_TO_STOP_RATIO,
            "target_r": config.target_r,
            "slippage_pips_each_side": config.slippage_pips_each_side,
        },
        "original_setups": int(len(setups)),
        "filtered_usable_setups": int(len(dataset)),
        "training_rows": training_rows,
        "holdout_rows": holdout_rows,
        "selected_trades": int(len(selected)),
        "summaries": summaries,
        "files": {key: str(value) for key, value in paths.items()},
    }
    paths["json"].write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    build_report(
        summaries=summaries,
        selected=selected,
        benchmark=holdout,
        equity=equity,
        calibration=calibration,
        holdout_start=HOLDOUT_START,
        cfg=config,
        path=paths["report"],
    )
    return payload
