from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from linq_engine.config import Phase4Config
from linq_engine.core import (
    build_report,
    equity_curve,
    load_candles,
    load_phase1,
    load_setups,
    predict_holdout,
    summarize,
)

HOLDOUT_START = pd.Timestamp("2026-05-01 16:00:00+00:00")
MINIMUM_STOP_PIPS = 8.0
MAXIMUM_SPREAD_PIPS = 2.5
MAXIMUM_SPREAD_TO_STOP_RATIO = 0.20


def divider(ch="=", n=108):
    return ch * n


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions[
        predictions["holdout"]
        & predictions["probability_1r"].notna()
    ].copy()

    bins = [0.0, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 1.000001]
    labels = [
        "<50%",
        "50–59%",
        "60–64%",
        "65–69%",
        "70–79%",
        "80–89%",
        "90%+",
    ]

    valid["confidence_band"] = pd.cut(
        valid["probability_1r"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return (
        valid.groupby("confidence_band", observed=False)
        .agg(
            predictions=("setup_id", "size"),
            average_probability=("probability_1r", "mean"),
            actual_win_rate=("actual_win", "mean"),
        )
        .reset_index()
    )


def evaluate_trade(
    direction: str,
    entry: float,
    stop: float,
    future: pd.DataFrame,
    spread_pips: float,
    config: Phase4Config,
) -> dict | None:
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0 or future.empty:
        return None

    target = entry + risk if direction == "long" else entry - risk

    if direction == "long":
        target_hits = future["high"] >= target
        stop_hits = future["low"] <= stop
        favorable = future["high"] - entry
        adverse = entry - future["low"]
    else:
        target_hits = future["low"] <= target
        stop_hits = future["high"] >= stop
        favorable = entry - future["low"]
        adverse = future["high"] - entry

    target_indices = np.flatnonzero(target_hits.to_numpy())
    stop_indices = np.flatnonzero(stop_hits.to_numpy())

    first_target = int(target_indices[0]) if len(target_indices) else None
    first_stop = int(stop_indices[0]) if len(stop_indices) else None

    won = (
        first_target is not None
        and (first_stop is None or first_target < first_stop)
    )

    if won:
        gross_r = config.target_r
        exit_bar = first_target
        exit_reason = "target"
        exit_price = target
    elif first_stop is not None:
        gross_r = -1.0
        exit_bar = first_stop
        exit_reason = "stop"
        exit_price = stop
    else:
        final_close = float(future["close"].iloc[-1])
        signed_move = (
            final_close - entry
            if direction == "long"
            else entry - final_close
        )
        gross_r = float(
            np.clip(
                signed_move / risk,
                -1.0,
                config.target_r,
            )
        )
        exit_bar = len(future) - 1
        exit_reason = "time_exit"
        exit_price = final_close

    spread = (
        float(spread_pips)
        if np.isfinite(spread_pips)
        else 0.0
    )

    spread_cost_r = (
        spread * config.pip_size / risk
    )
    slippage_cost_r = (
        2.0
        * config.slippage_pips_each_side
        * config.pip_size
        / risk
    )
    total_cost_r = spread_cost_r + slippage_cost_r

    return {
        "actual_win": int(won),
        "gross_result_r": gross_r,
        "net_result_r": gross_r - total_cost_r,
        "exit_reason": exit_reason,
        "bars_held": int(exit_bar + 1),
        "exit_price": float(exit_price),
        "target_price": float(target),
        "risk_price": float(risk),
        "mfe_r": float((favorable / risk).max()),
        "mae_r": float((adverse / risk).max()),
        "spread_cost_r": float(spread_cost_r),
        "slippage_cost_r": float(slippage_cost_r),
        "total_cost_r": float(total_cost_r),
    }


def build_controlled_dataset(
    candles: pd.DataFrame,
    setups: pd.DataFrame,
    phase1: pd.DataFrame,
    config: Phase4Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_map = (
        phase1.set_index("setup_id", drop=False)
        .to_dict("index")
    )
    candle_times = candles["timestamp"]

    rows = []
    exclusions = []

    for _, setup in setups.iterrows():
        exclusion_reason = None
        position = int(
            candle_times.searchsorted(
                setup["timestamp"],
                side="left",
            )
        )

        if position >= len(candles) or position < 200:
            exclusion_reason = "missing_candle_history"
        else:
            candle = candles.iloc[position]
            atr = (
                float(candle["atr"])
                if pd.notna(candle["atr"])
                else np.nan
            )

            if not np.isfinite(atr) or atr <= 0:
                exclusion_reason = "invalid_atr"
            else:
                spread = candle.get("spread_pips", np.nan)

                if (
                    np.isfinite(spread)
                    and spread > MAXIMUM_SPREAD_PIPS
                ):
                    exclusion_reason = "spread_above_limit"
                else:
                    future = candles.iloc[
                        position + 1:
                        position + 1 + config.forward_bars
                    ]

                    if future.empty:
                        exclusion_reason = "missing_forward_data"
                    else:
                        if (
                            "entry" in setup.index
                            and pd.notna(setup["entry"])
                        ):
                            entry = float(setup["entry"])
                        else:
                            entry = float(candle["close"])

                        atr_stop_distance = (
                            config.stop_atr * atr
                        )
                        minimum_stop_distance = (
                            MINIMUM_STOP_PIPS
                            * config.pip_size
                        )
                        stop_distance = max(
                            atr_stop_distance,
                            minimum_stop_distance,
                        )

                        stop = (
                            entry - stop_distance
                            if setup["direction"] == "long"
                            else entry + stop_distance
                        )

                        spread_to_stop_ratio = (
                            spread
                            / (
                                stop_distance
                                / config.pip_size
                            )
                            if np.isfinite(spread)
                            else np.nan
                        )

                        if (
                            np.isfinite(spread_to_stop_ratio)
                            and spread_to_stop_ratio
                            > MAXIMUM_SPREAD_TO_STOP_RATIO
                        ):
                            exclusion_reason = (
                                "spread_to_stop_above_limit"
                            )
                        else:
                            outcome = evaluate_trade(
                                direction=setup["direction"],
                                entry=entry,
                                stop=stop,
                                future=future,
                                spread_pips=spread,
                                config=config,
                            )

                            if outcome is None:
                                exclusion_reason = (
                                    "invalid_trade_geometry"
                                )
                            else:
                                base = feature_map.get(
                                    setup["setup_id"],
                                    {},
                                )

                                row = {}
                                for key, value in base.items():
                                    text = str(key)
                                    if (
                                        text.startswith("hit_")
                                        or text.startswith("bars_to_")
                                        or text.startswith("probability")
                                        or key in {
                                            "timestamp",
                                            "entry_price",
                                            "stop_price",
                                            "target_price",
                                            "risk_price",
                                            "mfe_r",
                                            "mae_r",
                                            "stop_hit",
                                            "valid_risk",
                                            "bars_observed",
                                            "result_r",
                                            "gross_result_r",
                                            "net_result_r",
                                            "exit_reason",
                                            "bars_held",
                                            "exit_price",
                                            "spread_cost_r",
                                            "slippage_cost_r",
                                            "total_cost_r",
                                            "selected",
                                            "actual_win",
                                            "holdout",
                                        }
                                    ):
                                        continue
                                    row[key] = value

                                row.update(
                                    {
                                        "setup_id": setup["setup_id"],
                                        "timestamp": setup["timestamp"],
                                        "direction": setup["direction"],
                                        "entry_price": entry,
                                        "stop_price": stop,
                                        "stop_distance_atr": (
                                            stop_distance / atr
                                        ),
                                        "stop_distance_pips": (
                                            stop_distance
                                            / config.pip_size
                                        ),
                                        "minimum_stop_applied": int(
                                            minimum_stop_distance
                                            > atr_stop_distance
                                        ),
                                        "spread_pips_at_entry": spread,
                                        "spread_to_stop_ratio": (
                                            spread_to_stop_ratio
                                        ),
                                        "atr_at_entry": atr,
                                        "market_ema20_50_atr": (
                                            (
                                                candle["ema20"]
                                                - candle["ema50"]
                                            )
                                            / atr
                                        ),
                                        "market_ema50_200_atr": (
                                            (
                                                candle["ema50"]
                                                - candle["ema200"]
                                            )
                                            / atr
                                        ),
                                        "market_rsi": candle["rsi"],
                                        "market_ret3": candle["ret3"],
                                        "market_ret12": candle["ret12"],
                                        "market_range_atr": (
                                            candle["range_atr"]
                                        ),
                                        "market_hour_utc": int(
                                            candle["hour_utc"]
                                        ),
                                        "market_weekday": int(
                                            candle["weekday"]
                                        ),
                                    }
                                )
                                row.update(outcome)
                                rows.append(row)

        exclusions.append(
            {
                "setup_id": setup["setup_id"],
                "timestamp": setup["timestamp"],
                "direction": setup["direction"],
                "included": int(
                    exclusion_reason is None
                ),
                "exclusion_reason": (
                    exclusion_reason
                    if exclusion_reason is not None
                    else "included"
                ),
            }
        )

    dataset = (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    exclusion_log = (
        pd.DataFrame(exclusions)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return dataset, exclusion_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candles",
        default="data/cache/EUR_USD_M5.csv",
    )
    parser.add_argument(
        "--setups",
        default="reports/v7/EUR_USD_automatic_setups.csv",
    )
    parser.add_argument(
        "--phase1",
        default="reports/v9/EUR_USD_market_database.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/v9_phase4_1",
    )
    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.10,
    )
    args = parser.parse_args()

    config = Phase4Config(
        holdout_trades=20,
        slippage_pips_each_side=args.slippage_pips,
        maximum_spread_pips=None,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(divider())
    print(
        "LINQ MARKET INTELLIGENCE ENGINE — "
        "V9 PHASE 4.1"
    )
    print(divider())
    print("CONTROLLED RULES")
    print(
        f"Fixed holdout start:             "
        f"{HOLDOUT_START}"
    )
    print(
        f"Probability threshold:           "
        f"{config.probability_threshold:.0%}"
    )
    print(
        f"ATR stop floor:                  "
        f"{config.stop_atr:.2f} ATR"
    )
    print(
        f"Minimum stop:                    "
        f"{MINIMUM_STOP_PIPS:.1f} pips"
    )
    print(
        f"Maximum spread:                  "
        f"{MAXIMUM_SPREAD_PIPS:.1f} pips"
    )
    print(
        f"Maximum spread/stop ratio:       "
        f"{MAXIMUM_SPREAD_TO_STOP_RATIO:.0%}"
    )
    print(
        f"Target:                          "
        f"{config.target_r:.2f}R"
    )

    candles = load_candles(args.candles)
    setups = load_setups(args.setups)
    phase1 = load_phase1(args.phase1)

    dataset, exclusion_log = build_controlled_dataset(
        candles=candles,
        setups=setups,
        phase1=phase1,
        config=config,
    )

    if dataset.empty:
        raise SystemExit(
            "No usable setups remained after filters."
        )

    holdout_mask = (
        dataset["timestamp"] >= HOLDOUT_START
    )
    training_rows = int((~holdout_mask).sum())
    holdout_rows = int(holdout_mask.sum())

    if training_rows < config.minimum_training_rows:
        raise SystemExit(
            f"Only {training_rows} filtered training setups "
            f"exist before fixed holdout start. "
            f"Need at least "
            f"{config.minimum_training_rows}."
        )

    if holdout_rows == 0:
        raise SystemExit(
            "No filtered setups exist in the fixed holdout."
        )

    holdout_start_index = int(
        np.flatnonzero(
            holdout_mask.to_numpy()
        )[0]
    )

    predictions = predict_holdout(
        dataset=dataset,
        holdout_start=holdout_start_index,
        cfg=config,
    )

    holdout = predictions[
        predictions["timestamp"] >= HOLDOUT_START
    ].copy()
    holdout["holdout"] = True

    selected = holdout[
        holdout["selected"]
    ].copy()

    selected_summary = summarize(
        "model_selected_filtered",
        selected,
        holdout,
    )
    filtered_benchmark_summary = summarize(
        "all_filtered_holdout_setups",
        holdout,
    )

    summaries = [
        selected_summary,
        filtered_benchmark_summary,
    ]

    equity = equity_curve(selected)
    calibration = calibration_table(predictions)

    exclusion_summary = (
        exclusion_log.groupby(
            ["included", "exclusion_reason"],
            dropna=False,
        )
        .size()
        .reset_index(name="setups")
        .sort_values(
            ["included", "setups"],
            ascending=[False, False],
        )
    )

    paths = {
        "dataset": (
            output_dir
            / "EUR_USD_phase4_1_dataset.csv"
        ),
        "predictions": (
            output_dir
            / "EUR_USD_phase4_1_predictions.csv"
        ),
        "selected": (
            output_dir
            / "EUR_USD_phase4_1_selected_trades.csv"
        ),
        "exclusions": (
            output_dir
            / "EUR_USD_phase4_1_exclusion_log.csv"
        ),
        "exclusion_summary": (
            output_dir
            / "EUR_USD_phase4_1_exclusion_summary.csv"
        ),
        "summary": (
            output_dir
            / "EUR_USD_phase4_1_summary.csv"
        ),
        "json": (
            output_dir
            / "EUR_USD_phase4_1_summary.json"
        ),
        "report": (
            output_dir
            / "EUR_USD_phase4_1_report.html"
        ),
    }

    dataset.to_csv(paths["dataset"], index=False)
    predictions.to_csv(
        paths["predictions"],
        index=False,
    )
    selected.to_csv(
        paths["selected"],
        index=False,
    )
    exclusion_log.to_csv(
        paths["exclusions"],
        index=False,
    )
    exclusion_summary.to_csv(
        paths["exclusion_summary"],
        index=False,
    )
    pd.DataFrame(summaries).to_csv(
        paths["summary"],
        index=False,
    )

    paths["json"].write_text(
        json.dumps(
            {
                "created_at_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "fixed_holdout_start": str(
                    HOLDOUT_START
                ),
                "rules": {
                    "threshold": (
                        config.probability_threshold
                    ),
                    "stop_atr": config.stop_atr,
                    "minimum_stop_pips": (
                        MINIMUM_STOP_PIPS
                    ),
                    "maximum_spread_pips": (
                        MAXIMUM_SPREAD_PIPS
                    ),
                    "maximum_spread_to_stop_ratio": (
                        MAXIMUM_SPREAD_TO_STOP_RATIO
                    ),
                    "target_r": config.target_r,
                    "slippage_pips_each_side": (
                        config.slippage_pips_each_side
                    ),
                },
                "training_rows": training_rows,
                "holdout_rows": holdout_rows,
                "summaries": summaries,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

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

    print("\nDATA")
    print(divider("-"))
    print(
        f"Original setups:                 "
        f"{len(setups)}"
    )
    print(
        f"Filtered usable setups:          "
        f"{len(dataset)}"
    )
    print(
        f"Training setups before holdout:  "
        f"{training_rows}"
    )
    print(
        f"Filtered holdout setups:         "
        f"{holdout_rows}"
    )
    print(
        f"Selected holdout trades:         "
        f"{len(selected)}"
    )

    print("\nEXCLUSIONS")
    print(divider("-"))
    for _, row in exclusion_summary.iterrows():
        print(
            f"{str(row['exclusion_reason']):<34}"
            f"{int(row['setups']):>4}"
        )

    print("\nFIXED-HOLDOUT RESULTS")
    print(divider("-"))
    for item in summaries:
        print(f"\n{item['strategy']}")
        print(
            f"  Trades:                        "
            f"{item['trades']}"
        )
        print(
            f"  Wins / losses:                 "
            f"{item['wins']} / {item['losses']}"
        )
        print(
            f"  Win rate:                      "
            f"{item['win_rate']:.1%}"
        )
        print(
            f"  Gross R:                       "
            f"{item['gross_r']:+.3f}R"
        )
        print(
            f"  Execution cost:                "
            f"-{item['cost_r']:.3f}R"
        )
        print(
            f"  Net R:                         "
            f"{item['net_r']:+.3f}R"
        )
        print(
            f"  Net expectancy:                "
            f"{item['expectancy_r']:+.3f}R"
        )
        print(
            f"  Profit factor:                 "
            f"{item['profit_factor']:.3f}"
        )
        print(
            f"  Maximum drawdown:              "
            f"{item['max_drawdown_r']:.3f}R"
        )
        print(
            f"  Longest losing streak:         "
            f"{item['longest_losing_streak']}"
        )

        if (
            item["strategy"]
            == "model_selected_filtered"
        ):
            print(
                f"  Holdout AUC:                   "
                f"{item['auc']:.3f}"
            )
            print(
                f"  Holdout Brier:                 "
                f"{item['brier']:.3f}"
            )

    print("\nFILES SAVED")
    print(divider("-"))
    for name, path in paths.items():
        print(f"{name.title():<32}{path}")

    print(
        f"\nOpen with:\n"
        f"open {paths['report']}"
    )
    print("\n" + divider())
    print(
        "V9 Phase 4.1 completed successfully."
    )
    print(divider())


if __name__ == "__main__":
    main()
