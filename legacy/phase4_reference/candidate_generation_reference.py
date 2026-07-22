from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-learn. Run:\n"
        "python -m pip install scikit-learn\n\n"
        f"{exc}"
    )

PAIR = "EUR_USD"

# Locked Phase 3 rules
MIN_TRAIN = 40
LOCKED_THRESHOLD = 0.65
LOCKED_ATR_STOP = 2.0
TARGET_R = 1.0
FORWARD_BARS = 288            # 24 hours on M5
SWING_LOOKBACK = 12           # Prior completed candles only
STRUCTURE_BUFFER_ATR = 0.10

STRATEGIES = ("locked_2atr", "structure_aware")


def divider(ch="=", n=108):
    return ch * n


def find_col(df, aliases, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(
            f"Missing required column. Expected one of {aliases}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_candles(path):
    raw = pd.read_csv(path)

    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "date"],
        "open": ["open", "o", "mid_open", "mid_o"],
        "high": ["high", "h", "mid_high", "mid_h"],
        "low": ["low", "l", "mid_low", "mid_l"],
        "close": ["close", "c", "mid_close", "mid_c"],
        "volume": ["volume", "tick_volume", "vol"],
        "bid_close": ["bid_close", "bid_c"],
        "ask_close": ["ask_close", "ask_c"],
    }

    rename = {}
    for canonical, options in aliases.items():
        required = canonical in {"timestamp", "open", "high", "low", "close"}
        col = find_col(raw, options, required=required)
        if col is not None:
            rename[col] = canonical

    df = raw.rename(columns=rename).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    for col in [
        "open", "high", "low", "close", "volume", "bid_close", "ask_close"
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df:
        df["volume"] = np.nan

    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    previous_close = df["close"].shift()
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    df["ret3"] = df["close"].pct_change(3)
    df["ret12"] = df["close"].pct_change(12)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"]
    df["hour_utc"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek

    if {"bid_close", "ask_close"}.issubset(df.columns):
        df["spread_pips"] = (df["ask_close"] - df["bid_close"]) * 10000
    else:
        df["spread_pips"] = np.nan

    return df


def load_setups(path):
    raw = pd.read_csv(path)

    aliases = {
        "setup_id": ["setup_id", "id", "zone_id"],
        "timestamp": [
            "timestamp", "entry_timestamp", "entry_time",
            "setup_timestamp", "created_at"
        ],
        "direction": ["direction", "side", "trade_direction"],
        "entry": ["entry", "entry_price", "price"],
        "stop": ["stop", "stop_price", "stop_loss", "sl"],
        "zone_low": ["zone_low", "distal", "lower_bound"],
        "zone_high": ["zone_high", "proximal", "upper_bound"],
    }

    rename = {}
    for canonical, options in aliases.items():
        required = canonical in {"timestamp", "direction"}
        col = find_col(raw, options, required=required)
        if col is not None:
            rename[col] = canonical

    df = raw.rename(columns=rename).copy()

    if "setup_id" not in df:
        df["setup_id"] = [f"setup_{i:06d}" for i in range(len(df))]

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    raw_direction = df["direction"].astype(str).str.lower().str.strip()
    normalized = pd.Series(pd.NA, index=df.index, dtype="object")

    normalized.loc[
        raw_direction.isin(
            ["long", "buy", "bull", "bullish", "demand", "1"]
        )
    ] = "long"

    normalized.loc[
        raw_direction.isin(
            ["short", "sell", "bear", "bearish", "supply", "-1"]
        )
    ] = "short"

    df["direction"] = normalized

    for col in ["entry", "stop", "zone_low", "zone_high"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "direction"])
        .sort_values("timestamp")
        .drop_duplicates("setup_id")
        .reset_index(drop=True)
    )


def load_phase1(path):
    df = pd.read_csv(path)
    if "setup_id" not in df:
        raise ValueError("Phase 1 database must contain setup_id.")
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def resolve_entry(setup, candle):
    if "entry" in setup.index and pd.notna(setup["entry"]):
        return float(setup["entry"])
    return float(candle["close"])


def zone_structure_level(setup):
    if setup["direction"] == "long":
        if "zone_low" in setup.index and pd.notna(setup["zone_low"]):
            return float(setup["zone_low"]), "zone"
        if "stop" in setup.index and pd.notna(setup["stop"]):
            return float(setup["stop"]), "existing_stop"
    else:
        if "zone_high" in setup.index and pd.notna(setup["zone_high"]):
            return float(setup["zone_high"]), "zone"
        if "stop" in setup.index and pd.notna(setup["stop"]):
            return float(setup["stop"]), "existing_stop"
    return np.nan, "none"


def calculate_structure_stop(setup, candles, position, entry, atr):
    history = candles.iloc[max(0, position - SWING_LOOKBACK):position]

    if len(history) < max(5, SWING_LOOKBACK // 2):
        return np.nan, "insufficient_history"

    zone_level, zone_source = zone_structure_level(setup)

    if setup["direction"] == "long":
        swing = float(history["low"].min())
        candidates = [(swing, "swing_low")]
        if np.isfinite(zone_level) and zone_level < entry:
            candidates.append((zone_level, zone_source))

        raw_level, source = min(candidates, key=lambda item: item[0])
        stop = raw_level - STRUCTURE_BUFFER_ATR * atr
        if stop >= entry:
            return np.nan, "invalid_structure"
        return stop, source

    swing = float(history["high"].max())
    candidates = [(swing, "swing_high")]
    if np.isfinite(zone_level) and zone_level > entry:
        candidates.append((zone_level, zone_source))

    raw_level, source = max(candidates, key=lambda item: item[0])
    stop = raw_level + STRUCTURE_BUFFER_ATR * atr
    if stop <= entry:
        return np.nan, "invalid_structure"
    return stop, source


def evaluate_trade(direction, entry, stop, future):
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

    # Conservative OHLC rule:
    # if both occur in the same bar, count the stop first.
    won = (
        first_target is not None
        and (first_stop is None or first_target < first_stop)
    )

    if won:
        result_r = 1.0
        exit_bar = first_target
        exit_reason = "target"
    elif first_stop is not None:
        result_r = -1.0
        exit_bar = first_stop
        exit_reason = "stop"
    else:
        final_close = float(future["close"].iloc[-1])
        signed_move = (
            final_close - entry
            if direction == "long"
            else entry - final_close
        )
        result_r = float(np.clip(signed_move / risk, -1.0, 1.0))
        exit_bar = len(future) - 1
        exit_reason = "time_exit"

    return {
        "hit_1r": int(won),
        "result_r": result_r,
        "exit_reason": exit_reason,
        "bars_held": int(exit_bar + 1),
        "mfe_r": float((favorable / risk).max()),
        "mae_r": float((adverse / risk).max()),
        "risk_price": risk,
    }


def clean_phase1_features(row):
    excluded = {
        "timestamp", "entry_price", "stop_price", "risk_price",
        "mfe_r", "mae_r", "stop_hit", "valid_risk", "bars_observed",
        "result_r", "exit_reason", "bars_held",
    }

    clean = {}
    for key, value in row.items():
        key_text = str(key)
        if (
            key in excluded
            or key_text.startswith("hit_")
            or key_text.startswith("bars_to_")
            or key_text.startswith("probability")
        ):
            continue
        clean[key] = value
    return clean


def build_validation_dataset(candles, setups, phase1):
    feature_map = phase1.set_index("setup_id", drop=False).to_dict("index")
    candle_times = candles["timestamp"]
    rows = []

    for _, setup in setups.iterrows():
        position = int(candle_times.searchsorted(setup["timestamp"], side="left"))

        if position >= len(candles) or position < 200:
            continue

        candle = candles.iloc[position]
        atr = float(candle["atr"]) if pd.notna(candle["atr"]) else np.nan

        if not np.isfinite(atr) or atr <= 0:
            continue

        future = candles.iloc[position + 1:position + 1 + FORWARD_BARS]
        if future.empty:
            continue

        entry = resolve_entry(setup, candle)

        atr_stop = (
            entry - LOCKED_ATR_STOP * atr
            if setup["direction"] == "long"
            else entry + LOCKED_ATR_STOP * atr
        )

        structure_stop, structure_source = calculate_structure_stop(
            setup, candles, position, entry, atr
        )

        if np.isfinite(structure_stop):
            if setup["direction"] == "long":
                final_structure_stop = min(atr_stop, structure_stop)
            else:
                final_structure_stop = max(atr_stop, structure_stop)
        else:
            final_structure_stop = atr_stop
            structure_source = "atr_fallback"

        phase1_row = feature_map.get(setup["setup_id"], {})
        base_features = clean_phase1_features(phase1_row)

        shared = dict(base_features)
        shared.update(
            {
                "setup_id": setup["setup_id"],
                "timestamp": setup["timestamp"],
                "direction": setup["direction"],
                "entry_price": entry,
                "atr_at_entry": atr,
                "spread_pips_at_entry": candle.get("spread_pips", np.nan),
                "market_ema20_50_atr": (
                    (candle["ema20"] - candle["ema50"]) / atr
                ),
                "market_ema50_200_atr": (
                    (candle["ema50"] - candle["ema200"]) / atr
                ),
                "market_rsi": candle["rsi"],
                "market_ret3": candle["ret3"],
                "market_ret12": candle["ret12"],
                "market_range_atr": candle["range_atr"],
                "market_hour_utc": int(candle["hour_utc"]),
                "market_weekday": int(candle["weekday"]),
            }
        )

        for strategy, stop, source in [
            ("locked_2atr", atr_stop, "2atr"),
            ("structure_aware", final_structure_stop, structure_source),
        ]:
            outcome = evaluate_trade(
                setup["direction"], entry, float(stop), future
            )

            if outcome is None:
                continue

            row = dict(shared)
            row.update(
                {
                    "strategy": strategy,
                    "stop_price": float(stop),
                    "stop_source": source,
                    "stop_distance_atr": abs(entry - float(stop)) / atr,
                    "structure_stop_price": (
                        structure_stop if np.isfinite(structure_stop) else np.nan
                    ),
                    "structure_stop_distance_atr": (
                        abs(entry - structure_stop) / atr
                        if np.isfinite(structure_stop)
                        else np.nan
                    ),
                }
            )
            row.update(outcome)
            rows.append(row)

    return pd.DataFrame(rows)


def make_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.25,
                    random_state=42,
                ),
            ),
        ]
    )


def walk_forward_predictions(strategy_df):
    data = strategy_df.sort_values("timestamp").reset_index(drop=True).copy()

    excluded = {
        "setup_id", "timestamp", "strategy",
        "entry_price", "stop_price", "structure_stop_price",
        "hit_1r", "result_r", "exit_reason",
        "bars_held", "mfe_r", "mae_r", "risk_price",
    }

    feature_columns = [
        col for col in data.columns
        if col not in excluded
    ]

    X = data[feature_columns].copy()
    y = data["hit_1r"].astype(int)
    probabilities = np.full(len(data), np.nan)

    for index in range(MIN_TRAIN, len(data)):
        train_y = y.iloc[:index]

        if train_y.nunique() < 2:
            probabilities[index] = float(train_y.mean())
            continue

        model = make_model(X.iloc[:index])
        model.fit(X.iloc[:index], train_y)

        probabilities[index] = model.predict_proba(
            X.iloc[index:index + 1]
        )[0, 1]

    data["probability_1r"] = probabilities
    data["selected"] = (
        data["probability_1r"].notna()
        & (data["probability_1r"] >= LOCKED_THRESHOLD)
    )

    return data


def max_drawdown(r_values):
    if len(r_values) == 0:
        return np.nan
    equity = np.cumsum(np.asarray(r_values, dtype=float))
    equity_with_origin = np.concatenate([[0.0], equity])
    peaks = np.maximum.accumulate(equity_with_origin)
    drawdowns = peaks - equity_with_origin
    return float(drawdowns.max())


def longest_losing_streak(r_values):
    longest = 0
    current = 0
    for value in r_values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def performance_summary(strategy, predictions):
    validation = predictions[predictions["probability_1r"].notna()].copy()
    trades = validation[validation["selected"]].copy()

    auc = np.nan
    brier = np.nan

    if not validation.empty:
        brier = float(
            brier_score_loss(
                validation["hit_1r"],
                validation["probability_1r"],
            )
        )

        if validation["hit_1r"].nunique() > 1:
            auc = float(
                roc_auc_score(
                    validation["hit_1r"],
                    validation["probability_1r"],
                )
            )

    wins = int((trades["result_r"] > 0).sum())
    losses = int((trades["result_r"] < 0).sum())
    gross_profit = float(trades.loc[trades["result_r"] > 0, "result_r"].sum())
    gross_loss = abs(
        float(trades.loc[trades["result_r"] < 0, "result_r"].sum())
    )

    if trades.empty:
        start = end = pd.NaT
        weeks = np.nan
    else:
        start = trades["timestamp"].min()
        end = trades["timestamp"].max()
        weeks = max((end - start).total_seconds() / (7 * 86400), 1 / 7)

    return {
        "strategy": strategy,
        "locked_threshold": LOCKED_THRESHOLD,
        "validation_rows": int(len(validation)),
        "selected_trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) if len(trades) else np.nan,
        "net_r": float(trades["result_r"].sum()) if len(trades) else 0.0,
        "expectancy_r": float(trades["result_r"].mean()) if len(trades) else np.nan,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf if gross_profit > 0 else np.nan
        ),
        "max_drawdown_r": max_drawdown(trades["result_r"].tolist()),
        "longest_losing_streak": longest_losing_streak(
            trades["result_r"].tolist()
        ),
        "average_bars_held": float(trades["bars_held"].mean()) if len(trades) else np.nan,
        "median_stop_atr": float(trades["stop_distance_atr"].median()) if len(trades) else np.nan,
        "trades_per_week": len(trades) / weeks if len(trades) else 0.0,
        "auc": auc,
        "brier": brier,
        "start": str(start) if pd.notna(start) else None,
        "end": str(end) if pd.notna(end) else None,
    }


def group_performance(trades, group_column):
    rows = []

    for name, group in trades.groupby(group_column, dropna=False):
        wins = int((group["result_r"] > 0).sum())
        rows.append(
            {
                group_column: name,
                "trades": int(len(group)),
                "wins": wins,
                "win_rate": wins / len(group),
                "net_r": float(group["result_r"].sum()),
                "expectancy_r": float(group["result_r"].mean()),
            }
        )

    return pd.DataFrame(rows)


def add_reporting_columns(trades):
    df = trades.copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["week"] = (
        df["timestamp"]
        .dt.to_period("W")
        .apply(lambda period: str(period.start_time.date()))
    )

    hour = df["market_hour_utc"]
    df["session"] = np.select(
        [
            hour.between(0, 6),
            hour.between(7, 11),
            hour.between(12, 16),
            hour.between(17, 21),
        ],
        [
            "Asia / pre-London",
            "London",
            "New York overlap",
            "Late New York",
        ],
        default="Other",
    )

    return df


def calibration_table(predictions):
    valid = predictions[predictions["probability_1r"].notna()].copy()

    bins = [0.0, 0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 1.000001]
    labels = [
        "<50%", "50–59%", "60–64%", "65–69%",
        "70–79%", "80–89%", "90%+"
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
            actual_win_rate=("hit_1r", "mean"),
        )
        .reset_index()
    )


def html_table(df):
    if df is None or df.empty:
        return "<p>No rows available.</p>"
    return df.to_html(
        index=False,
        border=0,
        float_format=lambda x: f"{x:.3f}",
    )


def build_report(
    summaries,
    predictions,
    calibration,
    breakdowns,
    stop_sources,
    output_path,
):
    summary_df = pd.DataFrame(summaries)

    cards = ""
    for item in summaries:
        cards += f"""
        <div class="card">
            <h3>{item['strategy']}</h3>
            <div class="metric">{item['selected_trades']} trades</div>
            <p>{item['win_rate']:.1%} win rate</p>
            <p>{item['net_r']:+.2f}R net · {item['expectancy_r']:+.3f}R expectancy</p>
            <p>PF {item['profit_factor']:.2f} · DD {item['max_drawdown_r']:.2f}R</p>
        </div>
        """

    sections = ""
    for strategy in STRATEGIES:
        strategy_predictions = predictions[
            predictions["strategy"] == strategy
        ]
        selected = add_reporting_columns(
            strategy_predictions[strategy_predictions["selected"]]
        )

        trade_columns = [
            "timestamp", "setup_id", "direction", "probability_1r",
            "stop_source", "stop_distance_atr", "result_r",
            "exit_reason", "bars_held"
        ]
        trade_columns = [
            col for col in trade_columns if col in selected.columns
        ]

        sections += f"""
        <div class="section">
            <h2>{strategy}: selected trades</h2>
            {html_table(selected[trade_columns])}
        </div>
        <div class="section">
            <h2>{strategy}: confidence calibration</h2>
            {html_table(calibration[strategy])}
        </div>
        <div class="section">
            <h2>{strategy}: monthly performance</h2>
            {html_table(breakdowns[strategy]['month'])}
        </div>
        <div class="section">
            <h2>{strategy}: direction performance</h2>
            {html_table(breakdowns[strategy]['direction'])}
        </div>
        <div class="section">
            <h2>{strategy}: session performance</h2>
            {html_table(breakdowns[strategy]['session'])}
        </div>
        """

    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>LINQ V9 Phase 3</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #f3f5f7;
    color: #18212b;
    margin: 0;
}}
.container {{ max-width: 1450px; margin: auto; padding: 30px 20px; }}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
    gap: 16px;
}}
.card, .section {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
    margin: 18px 0;
    overflow: auto;
}}
.card {{ margin: 0; }}
.metric {{ font-size: 27px; font-weight: 700; }}
.notice {{
    background: #fff1d6;
    border-left: 5px solid #d89900;
    padding: 16px;
    border-radius: 8px;
    margin: 18px 0;
}}
.locked {{
    background: #eaf4ff;
    border-left: 5px solid #3977b8;
    padding: 16px;
    border-radius: 8px;
}}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #edf1f4; }}
th, td {{
    padding: 9px;
    border-bottom: 1px solid #ddd;
    text-align: left;
    white-space: nowrap;
}}
</style>
</head>
<body>
<div class="container">
<h1>LINQ Market Intelligence Engine — V9 Phase 3</h1>
<p>Locked-strategy validation and structure-aware stop testing</p>

<div class="locked">
<strong>Locked rules:</strong>
65% minimum model probability · 2 ATR baseline stop · 1R target ·
40 initial training examples · expanding walk-forward prediction ·
no trailing stop · no partial exit.
</div>

<div class="notice">
The structure-aware stop is a comparison strategy, not a retroactive replacement
for the locked 2 ATR baseline. The sample remains small and should not be treated
as live-capital validation.
</div>

<div class="grid">{cards}</div>

<div class="section">
<h2>Strategy comparison</h2>
{html_table(summary_df)}
</div>

<div class="section">
<h2>Structure-aware stop sources</h2>
{html_table(stop_sources)}
</div>

{sections}
</div>
</body>
</html>""",
        encoding="utf-8",
    )


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
        default="reports/v9_phase3",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(divider())
    print("LINQ MARKET INTELLIGENCE ENGINE — V9 PHASE 3")
    print(divider())
    print("LOCKED RULES")
    print(f"Probability threshold:           {LOCKED_THRESHOLD:.0%}")
    print(f"Baseline stop:                   {LOCKED_ATR_STOP:.2f} ATR")
    print(f"Target:                          {TARGET_R:.2f}R")
    print(f"Initial training rows:           {MIN_TRAIN}")
    print(f"Structure lookback:              {SWING_LOOKBACK} completed M5 bars")
    print(f"Structure buffer:                {STRUCTURE_BUFFER_ATR:.2f} ATR")

    candles = load_candles(args.candles)
    setups = load_setups(args.setups)
    phase1 = load_phase1(args.phase1)

    dataset = build_validation_dataset(candles, setups, phase1)

    if dataset.empty:
        raise SystemExit("No validation rows were generated.")

    predictions_list = []
    summaries = []
    calibration = {}
    breakdowns = {}

    for strategy in STRATEGIES:
        strategy_data = dataset[dataset["strategy"] == strategy].copy()
        predictions = walk_forward_predictions(strategy_data)
        predictions_list.append(predictions)

        summary = performance_summary(strategy, predictions)
        summaries.append(summary)

        calibration[strategy] = calibration_table(predictions)

        selected = add_reporting_columns(
            predictions[predictions["selected"]].copy()
        )

        breakdowns[strategy] = {
            "month": group_performance(selected, "month"),
            "week": group_performance(selected, "week"),
            "direction": group_performance(selected, "direction"),
            "session": group_performance(selected, "session"),
        }

    all_predictions = pd.concat(predictions_list, ignore_index=True)

    stop_sources = (
        dataset[dataset["strategy"] == "structure_aware"]
        .groupby("stop_source", dropna=False)
        .agg(
            examples=("setup_id", "size"),
            median_stop_atr=("stop_distance_atr", "median"),
            raw_win_rate=("hit_1r", "mean"),
        )
        .reset_index()
        .sort_values("examples", ascending=False)
    )

    dataset_path = output_dir / f"{PAIR}_phase3_validation_dataset.csv"
    predictions_path = output_dir / f"{PAIR}_phase3_walk_forward_predictions.csv"
    summary_csv_path = output_dir / f"{PAIR}_phase3_strategy_summary.csv"
    summary_json_path = output_dir / f"{PAIR}_phase3_summary.json"
    report_path = output_dir / f"{PAIR}_phase3_report.html"

    dataset.to_csv(dataset_path, index=False)
    all_predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv_path, index=False)

    json_payload = {
        "pair": PAIR,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_rules": {
            "minimum_probability": LOCKED_THRESHOLD,
            "baseline_stop_atr": LOCKED_ATR_STOP,
            "target_r": TARGET_R,
            "minimum_training_rows": MIN_TRAIN,
            "forward_bars": FORWARD_BARS,
            "swing_lookback": SWING_LOOKBACK,
            "structure_buffer_atr": STRUCTURE_BUFFER_ATR,
        },
        "strategy_summaries": summaries,
    }

    summary_json_path.write_text(
        json.dumps(json_payload, indent=2, default=str),
        encoding="utf-8",
    )

    build_report(
        summaries=summaries,
        predictions=all_predictions,
        calibration=calibration,
        breakdowns=breakdowns,
        stop_sources=stop_sources,
        output_path=report_path,
    )

    print("\nDATA")
    print(divider("-"))
    print(f"Candles loaded:                  {len(candles):,}")
    print(f"Setups loaded:                   {len(setups):,}")
    print(f"Validation rows:                 {len(dataset):,}")
    print(f"Strategies:                      {dataset['strategy'].nunique()}")

    print("\nLOCKED WALK-FORWARD RESULTS")
    print(divider("-"))

    for summary in summaries:
        print(f"\n{summary['strategy']}")
        print(f"  Validation predictions:        {summary['validation_rows']}")
        print(f"  Selected trades:               {summary['selected_trades']}")
        print(f"  Wins / losses:                 {summary['wins']} / {summary['losses']}")
        print(f"  Win rate:                      {summary['win_rate']:.1%}")
        print(f"  Net R:                         {summary['net_r']:+.2f}R")
        print(f"  Expectancy:                    {summary['expectancy_r']:+.3f}R")
        print(f"  Profit factor:                 {summary['profit_factor']:.3f}")
        print(f"  Maximum drawdown:              {summary['max_drawdown_r']:.2f}R")
        print(f"  Longest losing streak:         {summary['longest_losing_streak']}")
        print(f"  Median stop:                   {summary['median_stop_atr']:.2f} ATR")
        print(f"  Trades per week:               {summary['trades_per_week']:.2f}")
        print(f"  Walk-forward AUC:              {summary['auc']:.3f}")
        print(f"  Brier score:                   {summary['brier']:.3f}")

    print("\nSTRUCTURE STOP USAGE")
    print(divider("-"))
    for _, row in stop_sources.iterrows():
        print(
            f"{str(row['stop_source']):<22}"
            f" examples={int(row['examples']):>3}"
            f"  median={row['median_stop_atr']:.2f} ATR"
            f"  raw 1R win={row['raw_win_rate']:.1%}"
        )

    print("\nFILES SAVED")
    print(divider("-"))
    print(f"Validation dataset:              {dataset_path}")
    print(f"Walk-forward predictions:        {predictions_path}")
    print(f"Strategy summary:                {summary_csv_path}")
    print(f"JSON summary:                    {summary_json_path}")
    print(f"Readable report:                 {report_path}")
    print(f"\nOpen with:\nopen {report_path}")
    print("\n" + divider())
    print("V9 Phase 3 completed successfully.")
    print(divider())


if __name__ == "__main__":
    main()
