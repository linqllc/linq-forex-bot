from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PAIR = "EUR_USD"

TARGET_R_MULTIPLES = (
    1.00,
    1.25,
    1.50,
    2.00,
    3.00,
)

FORWARD_BARS = 288  # 24 hours of M5 candles
MINIMUM_HISTORY_BARS = 50


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def divider(character: str = "=", length: int = 100) -> str:
    return character * length


def find_column(
    frame: pd.DataFrame,
    aliases: list[str],
    required: bool = True,
) -> str | None:
    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]

    if required:
        raise ValueError(
            "Missing required column. Expected one of: "
            + ", ".join(aliases)
            + "\n\nAvailable columns:\n"
            + "\n".join(str(column) for column in frame.columns)
        )

    return None


def target_suffix(target_r: float) -> str:
    return str(target_r).replace(".", "_")


# =============================================================================
# CANDLE DATA
# =============================================================================

def load_candles(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise SystemExit(f"Candle file not found: {path}")

    raw = pd.read_csv(path)

    aliases = {
        "timestamp": [
            "timestamp",
            "time",
            "datetime",
            "date",
        ],
        "open": [
            "open",
            "o",
            "mid_o",
            "mid_open",
        ],
        "high": [
            "high",
            "h",
            "mid_h",
            "mid_high",
        ],
        "low": [
            "low",
            "l",
            "mid_l",
            "mid_low",
        ],
        "close": [
            "close",
            "c",
            "mid_c",
            "mid_close",
        ],
        "volume": [
            "volume",
            "tick_volume",
            "vol",
        ],
        "bid_open": [
            "bid_open",
            "bid_o",
        ],
        "bid_high": [
            "bid_high",
            "bid_h",
        ],
        "bid_low": [
            "bid_low",
            "bid_l",
        ],
        "bid_close": [
            "bid_close",
            "bid_c",
        ],
        "ask_open": [
            "ask_open",
            "ask_o",
        ],
        "ask_high": [
            "ask_high",
            "ask_h",
        ],
        "ask_low": [
            "ask_low",
            "ask_l",
        ],
        "ask_close": [
            "ask_close",
            "ask_c",
        ],
    }

    rename_map: dict[str, str] = {}

    for canonical, possible_names in aliases.items():
        required = canonical in {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        }

        column = find_column(
            raw,
            possible_names,
            required=required,
        )

        if column is not None:
            rename_map[column] = canonical

    candles = raw.rename(columns=rename_map).copy()

    candles["timestamp"] = pd.to_datetime(
        candles["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]

    for column in numeric_columns:
        if column in candles.columns:
            candles[column] = pd.to_numeric(
                candles[column],
                errors="coerce",
            )

    if "volume" not in candles.columns:
        candles["volume"] = np.nan

    candles = (
        candles.dropna(
            subset=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    # OANDA spread information.
    if {
        "bid_close",
        "ask_close",
    }.issubset(candles.columns):
        candles["spread_price"] = (
            candles["ask_close"] - candles["bid_close"]
        )

        candles["spread_pips"] = (
            candles["spread_price"] * 10_000
        )
    else:
        candles["spread_price"] = np.nan
        candles["spread_pips"] = np.nan

    return candles


def add_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()

    previous_close = frame["close"].shift(1)

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    frame["atr"] = true_range.rolling(
        14,
        min_periods=14,
    ).mean()

    frame["ema_20"] = frame["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    frame["ema_50"] = frame["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    frame["ema_200"] = frame["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    delta = frame["close"].diff()

    average_gain = (
        delta.clip(lower=0)
        .rolling(14, min_periods=14)
        .mean()
    )

    average_loss = (
        -delta.clip(upper=0)
        .rolling(14, min_periods=14)
        .mean()
    )

    relative_strength = average_gain / average_loss.replace(
        0,
        np.nan,
    )

    frame["rsi_14"] = (
        100 - (100 / (1 + relative_strength))
    )

    frame["return_1"] = frame["close"].pct_change(1)
    frame["return_3"] = frame["close"].pct_change(3)
    frame["return_6"] = frame["close"].pct_change(6)
    frame["return_12"] = frame["close"].pct_change(12)
    frame["return_24"] = frame["close"].pct_change(24)

    frame["candle_range"] = (
        frame["high"] - frame["low"]
    )

    frame["candle_body"] = (
        frame["close"] - frame["open"]
    ).abs()

    frame["upper_wick"] = (
        frame["high"]
        - frame[["open", "close"]].max(axis=1)
    )

    frame["lower_wick"] = (
        frame[["open", "close"]].min(axis=1)
        - frame["low"]
    )

    frame["rolling_high_20"] = (
        frame["high"].rolling(20).max()
    )

    frame["rolling_low_20"] = (
        frame["low"].rolling(20).min()
    )

    frame["rolling_high_50"] = (
        frame["high"].rolling(50).max()
    )

    frame["rolling_low_50"] = (
        frame["low"].rolling(50).min()
    )

    frame["volatility_20"] = (
        frame["return_1"].rolling(20).std()
    )

    frame["volume_average_20"] = (
        frame["volume"].rolling(20).mean()
    )

    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month

    frame["asian_session"] = (
        frame["hour_utc"].between(0, 6)
    ).astype(int)

    frame["london_session"] = (
        frame["hour_utc"].between(7, 11)
    ).astype(int)

    frame["new_york_session"] = (
        frame["hour_utc"].between(12, 16)
    ).astype(int)

    frame["london_new_york_overlap"] = (
        frame["hour_utc"].between(12, 15)
    ).astype(int)

    return frame


# =============================================================================
# SETUP DATA
# =============================================================================

def normalize_direction(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    long_values = {
        "long",
        "buy",
        "bull",
        "bullish",
        "demand",
        "1",
    }

    short_values = {
        "short",
        "sell",
        "bear",
        "bearish",
        "supply",
        "-1",
    }

    result = pd.Series(
        index=series.index,
        dtype="object",
    )

    result.loc[text.isin(long_values)] = "long"
    result.loc[text.isin(short_values)] = "short"

    return result


def load_setups(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise SystemExit(f"Setup file not found: {path}")

    raw = pd.read_csv(path)

    aliases = {
        "setup_id": [
            "setup_id",
            "id",
            "zone_id",
        ],
        "timestamp": [
            "entry_timestamp",
            "entry_time",
            "timestamp",
            "setup_timestamp",
            "created_at",
        ],
        "direction": [
            "direction",
            "side",
            "trade_direction",
        ],
        "entry": [
            "entry_price",
            "entry",
            "price",
        ],
        "stop": [
            "stop_price",
            "stop",
            "stop_loss",
            "sl",
        ],
        "zone_high": [
            "zone_high",
            "proximal",
            "upper_bound",
        ],
        "zone_low": [
            "zone_low",
            "distal",
            "lower_bound",
        ],
    }

    rename_map: dict[str, str] = {}

    for canonical, possible_names in aliases.items():
        required = canonical in {
            "timestamp",
            "direction",
        }

        column = find_column(
            raw,
            possible_names,
            required=required,
        )

        if column is not None:
            rename_map[column] = canonical

    setups = raw.rename(columns=rename_map).copy()

    if "setup_id" not in setups.columns:
        setups["setup_id"] = [
            f"setup_{index:06d}"
            for index in range(len(setups))
        ]

    setups["timestamp"] = pd.to_datetime(
        setups["timestamp"],
        utc=True,
        errors="coerce",
    )

    setups["direction"] = normalize_direction(
        setups["direction"]
    )

    for column in [
        "entry",
        "stop",
        "zone_high",
        "zone_low",
    ]:
        if column in setups.columns:
            setups[column] = pd.to_numeric(
                setups[column],
                errors="coerce",
            )

    setups = (
        setups.dropna(
            subset=[
                "timestamp",
                "direction",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates("setup_id")
        .reset_index(drop=True)
    )

    return setups


# =============================================================================
# ENTRY AND STOP
# =============================================================================

def determine_entry_and_stop(
    setup: pd.Series,
    candle: pd.Series,
) -> tuple[float, float]:
    direction = setup["direction"]

    if "entry" in setup.index and pd.notna(setup["entry"]):
        entry = float(setup["entry"])
    else:
        entry = float(candle["close"])

    if "stop" in setup.index and pd.notna(setup["stop"]):
        stop = float(setup["stop"])

    elif (
        direction == "long"
        and "zone_low" in setup.index
        and pd.notna(setup["zone_low"])
    ):
        stop = float(setup["zone_low"])

    elif (
        direction == "short"
        and "zone_high" in setup.index
        and pd.notna(setup["zone_high"])
    ):
        stop = float(setup["zone_high"])

    else:
        atr = float(candle["atr"])

        if direction == "long":
            stop = entry - atr
        else:
            stop = entry + atr

    return entry, stop


# =============================================================================
# PRE-ENTRY FEATURES
# =============================================================================

def build_features(
    setup: pd.Series,
    candle: pd.Series,
    history: pd.DataFrame,
) -> dict:
    direction = setup["direction"]
    direction_sign = 1 if direction == "long" else -1

    atr = float(candle["atr"])
    close = float(candle["close"])

    recent_20 = history.tail(20)
    recent_50 = history.tail(50)

    features = {
        "setup_id": setup["setup_id"],
        "timestamp": setup["timestamp"],
        "direction": direction,

        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": close,
        "volume": candle["volume"],

        "spread_price": candle.get(
            "spread_price",
            np.nan,
        ),
        "spread_pips": candle.get(
            "spread_pips",
            np.nan,
        ),

        "atr": atr,
        "atr_percent": (
            atr / close
            if close != 0
            else np.nan
        ),

        "ema_20": candle["ema_20"],
        "ema_50": candle["ema_50"],
        "ema_200": candle["ema_200"],

        "ema_20_50_spread_atr": (
            (candle["ema_20"] - candle["ema_50"])
            / atr
        ),

        "ema_50_200_spread_atr": (
            (candle["ema_50"] - candle["ema_200"])
            / atr
        ),

        "trend_alignment_20_50": int(
            (
                direction == "long"
                and candle["ema_20"] > candle["ema_50"]
            )
            or
            (
                direction == "short"
                and candle["ema_20"] < candle["ema_50"]
            )
        ),

        "trend_alignment_50_200": int(
            (
                direction == "long"
                and candle["ema_50"] > candle["ema_200"]
            )
            or
            (
                direction == "short"
                and candle["ema_50"] < candle["ema_200"]
            )
        ),

        "rsi_14": candle["rsi_14"],

        "return_1": candle["return_1"],
        "return_3": candle["return_3"],
        "return_6": candle["return_6"],
        "return_12": candle["return_12"],
        "return_24": candle["return_24"],

        "directional_return_3": (
            direction_sign * candle["return_3"]
        ),

        "directional_return_12": (
            direction_sign * candle["return_12"]
        ),

        "candle_range_atr": (
            candle["candle_range"] / atr
        ),

        "candle_body_atr": (
            candle["candle_body"] / atr
        ),

        "upper_wick_atr": (
            candle["upper_wick"] / atr
        ),

        "lower_wick_atr": (
            candle["lower_wick"] / atr
        ),

        "distance_to_high_20_atr": (
            candle["rolling_high_20"] - close
        ) / atr,

        "distance_to_low_20_atr": (
            close - candle["rolling_low_20"]
        ) / atr,

        "distance_to_high_50_atr": (
            candle["rolling_high_50"] - close
        ) / atr,

        "distance_to_low_50_atr": (
            close - candle["rolling_low_50"]
        ) / atr,

        "volatility_20": candle["volatility_20"],

        "volume_relative_20": (
            candle["volume"]
            / candle["volume_average_20"]
            if pd.notna(candle["volume_average_20"])
            and candle["volume_average_20"] != 0
            else np.nan
        ),

        "recent_up_candle_ratio_20": float(
            (
                recent_20["close"]
                > recent_20["open"]
            ).mean()
        ),

        "recent_directional_return_20": float(
            direction_sign
            * (
                recent_20["close"].iloc[-1]
                / recent_20["close"].iloc[0]
                - 1
            )
        ),

        "recent_range_average_atr_20": float(
            recent_20["candle_range"].mean()
            / atr
        ),

        "recent_range_expansion": float(
            recent_20["candle_range"].tail(5).mean()
            / recent_20["candle_range"].head(15).mean()
        ),

        "recent_high_break": int(
            close
            >= recent_50["high"].iloc[:-1].max()
        ),

        "recent_low_break": int(
            close
            <= recent_50["low"].iloc[:-1].min()
        ),

        "hour_utc": candle["hour_utc"],
        "weekday": candle["weekday"],
        "month": candle["month"],

        "asian_session": candle["asian_session"],
        "london_session": candle["london_session"],
        "new_york_session": candle["new_york_session"],
        "london_new_york_overlap": (
            candle["london_new_york_overlap"]
        ),
    }

    # Preserve useful V7 setup information.
    excluded_setup_fields = {
        "setup_id",
        "timestamp",
        "direction",
        "entry",
        "stop",
    }

    for column, value in setup.items():
        if column in excluded_setup_fields:
            continue

        if column in features:
            continue

        features[f"setup_{column}"] = value

    return features


# =============================================================================
# POST-ENTRY OUTCOMES
# =============================================================================

def evaluate_future_path(
    direction: str,
    entry: float,
    stop: float,
    future: pd.DataFrame,
) -> dict:
    risk = abs(entry - stop)

    if not np.isfinite(risk) or risk <= 0:
        return {
            "valid_risk": 0,
        }

    if direction == "long":
        favorable_distance = (
            future["high"] - entry
        )

        adverse_distance = (
            entry - future["low"]
        )

        stop_hits = (
            future["low"] <= stop
        )

    else:
        favorable_distance = (
            entry - future["low"]
        )

        adverse_distance = (
            future["high"] - entry
        )

        stop_hits = (
            future["high"] >= stop
        )

    favorable_r = favorable_distance / risk
    adverse_r = adverse_distance / risk

    stop_indices = np.flatnonzero(
        stop_hits.to_numpy()
    )

    first_stop_index = (
        int(stop_indices[0])
        if len(stop_indices)
        else None
    )

    result = {
        "valid_risk": 1,
        "entry_price": entry,
        "stop_price": stop,
        "risk_price": risk,

        "mfe_r": float(
            favorable_r.max()
        ),

        "mae_r": float(
            max(adverse_r.max(), 0)
        ),

        "bars_observed": int(
            len(future)
        ),

        "stop_hit": int(
            first_stop_index is not None
        ),

        "bars_to_stop": (
            first_stop_index + 1
            if first_stop_index is not None
            else np.nan
        ),
    }

    for target_r in TARGET_R_MULTIPLES:
        suffix = target_suffix(target_r)

        if direction == "long":
            target_price = (
                entry + risk * target_r
            )

            target_hits = (
                future["high"] >= target_price
            )

        else:
            target_price = (
                entry - risk * target_r
            )

            target_hits = (
                future["low"] <= target_price
            )

        target_indices = np.flatnonzero(
            target_hits.to_numpy()
        )

        first_target_index = (
            int(target_indices[0])
            if len(target_indices)
            else None
        )

        target_before_stop = (
            first_target_index is not None
            and (
                first_stop_index is None
                or first_target_index < first_stop_index
            )
        )

        result[
            f"target_{suffix}r_price"
        ] = target_price

        result[
            f"hit_{suffix}r_before_stop"
        ] = int(target_before_stop)

        result[
            f"bars_to_{suffix}r"
        ] = (
            first_target_index + 1
            if first_target_index is not None
            else np.nan
        )

    return result


# =============================================================================
# MARKET DATABASE
# =============================================================================

def build_market_database(
    candles: pd.DataFrame,
    setups: pd.DataFrame,
) -> pd.DataFrame:
    candles = add_indicators(candles)

    candle_times = candles["timestamp"]

    rows: list[dict] = []

    for _, setup in setups.iterrows():
        candle_position = int(
            candle_times.searchsorted(
                setup["timestamp"],
                side="left",
            )
        )

        if candle_position >= len(candles):
            continue

        if candle_position < MINIMUM_HISTORY_BARS:
            continue

        candle = candles.iloc[candle_position]

        if pd.isna(candle["atr"]) or candle["atr"] <= 0:
            continue

        history = candles.iloc[
            max(0, candle_position - 200):
            candle_position + 1
        ]

        future = candles.iloc[
            candle_position + 1:
            candle_position + 1 + FORWARD_BARS
        ]

        if future.empty:
            continue

        entry, stop = determine_entry_and_stop(
            setup,
            candle,
        )

        features = build_features(
            setup,
            candle,
            history,
        )

        outcome_data = evaluate_future_path(
            setup["direction"],
            entry,
            stop,
            future,
        )

        rows.append(
            {
                **features,
                **outcome_data,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# REPORTING
# =============================================================================

def build_summary(
    dataset: pd.DataFrame,
) -> dict:
    summary = {
        "pair": PAIR,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),
        "targets": {},
    }

    if dataset.empty:
        return summary

    summary["date_start"] = str(
        dataset["timestamp"].min()
    )

    summary["date_end"] = str(
        dataset["timestamp"].max()
    )

    summary["direction_counts"] = (
        dataset["direction"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary["median_mfe_r"] = float(
        dataset["mfe_r"].median()
    )

    summary["median_mae_r"] = float(
        dataset["mae_r"].median()
    )

    summary["average_spread_pips"] = (
        float(dataset["spread_pips"].mean())
        if "spread_pips" in dataset
        else None
    )

    for target_r in TARGET_R_MULTIPLES:
        suffix = target_suffix(target_r)
        column = f"hit_{suffix}r_before_stop"

        if column not in dataset.columns:
            continue

        summary["targets"][str(target_r)] = {
            "wins": int(dataset[column].sum()),
            "losses": int(
                len(dataset) - dataset[column].sum()
            ),
            "win_rate": float(
                dataset[column].mean()
            ),
        }

    return summary


def write_html_report(
    dataset: pd.DataFrame,
    summary: dict,
    path: Path,
) -> None:
    target_rows = ""

    for target_r, values in summary["targets"].items():
        target_rows += f"""
        <tr>
            <td>{target_r}R</td>
            <td>{values["wins"]}</td>
            <td>{values["losses"]}</td>
            <td>{values["win_rate"]:.1%}</td>
        </tr>
        """

    preview_columns = [
        "timestamp",
        "direction",
        "entry_price",
        "stop_price",
        "risk_price",
        "spread_pips",
        "atr",
        "rsi_14",
        "trend_alignment_20_50",
        "mfe_r",
        "mae_r",
        "hit_1_0r_before_stop",
        "hit_1_25r_before_stop",
        "hit_1_5r_before_stop",
        "hit_2_0r_before_stop",
        "hit_3_0r_before_stop",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in dataset.columns
    ]

    if dataset.empty:
        preview_html = "<p>No historical examples were generated.</p>"
    else:
        preview_html = (
            dataset[preview_columns]
            .tail(30)
            .sort_values(
                "timestamp",
                ascending=False,
            )
            .to_html(
                index=False,
                border=0,
                classes="data-table",
                float_format=lambda value: f"{value:.5f}",
            )
        )

    path.write_text(
        f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>LINQ V9 Market Database</title>

<style>
body {{
    margin: 0;
    background: #f4f6f8;
    color: #18212b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.container {{
    max-width: 1350px;
    margin: 0 auto;
    padding: 32px 20px 60px;
}}

.cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 14px;
}}

.card,
.section {{
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,.05);
}}

.section {{
    margin-top: 20px;
    overflow-x: auto;
}}

.label {{
    color: #687482;
    font-size: 13px;
}}

.value {{
    font-size: 27px;
    font-weight: 700;
    margin-top: 7px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

th {{
    background: #edf1f4;
    text-align: left;
    padding: 9px;
}}

td {{
    padding: 9px;
    border-bottom: 1px solid #e7ebef;
    white-space: nowrap;
}}

.notice {{
    background: #e9f3ff;
    border-left: 5px solid #3178c6;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}}
</style>
</head>

<body>
<div class="container">

<h1>LINQ Market Intelligence Engine — V9 Phase 1</h1>
<p>Historical market database and adaptive-target labels</p>

<div class="notice">
    This phase creates training data only. It does not select or execute trades.
    Targets below 1R are intentionally excluded.
</div>

<div class="cards">

    <div class="card">
        <div class="label">Historical examples</div>
        <div class="value">{summary["rows"]}</div>
    </div>

    <div class="card">
        <div class="label">Dataset columns</div>
        <div class="value">{summary["columns"]}</div>
    </div>

    <div class="card">
        <div class="label">Median favorable movement</div>
        <div class="value">{summary.get("median_mfe_r", float("nan")):.2f}R</div>
    </div>

    <div class="card">
        <div class="label">Median adverse movement</div>
        <div class="value">{summary.get("median_mae_r", float("nan")):.2f}R</div>
    </div>

    <div class="card">
        <div class="label">Average spread</div>
        <div class="value">{summary.get("average_spread_pips", float("nan")):.2f} pips</div>
    </div>

</div>

<div class="section">

<h2>Target reached before stop</h2>

<table>
<thead>
<tr>
    <th>Target</th>
    <th>Wins</th>
    <th>Losses</th>
    <th>Win rate</th>
</tr>
</thead>

<tbody>
{target_rows}
</tbody>
</table>

</div>

<div class="section">
<h2>Latest generated examples</h2>
{preview_html}
</div>

</div>
</body>
</html>
        """,
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the LINQ V9 historical market database."
        )
    )

    parser.add_argument(
        "--candles",
        default="data/cache/EUR_USD_M5.csv",
    )

    parser.add_argument(
        "--setups",
        default=(
            "reports/v7/"
            "EUR_USD_automatic_setups.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default="reports/v9",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    output_directory = Path(arguments.output)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        output_directory
        / f"{PAIR}_market_database.csv"
    )

    summary_path = (
        output_directory
        / f"{PAIR}_market_database_summary.json"
    )

    report_path = (
        output_directory
        / f"{PAIR}_market_database_report.html"
    )

    print(divider())
    print("LINQ MARKET INTELLIGENCE ENGINE — V9 PHASE 1")
    print(divider())

    candle_data = load_candles(
        arguments.candles
    )

    setup_data = load_setups(
        arguments.setups
    )

    print(f"Candles loaded:              {len(candle_data):,}")
    print(f"Candidate setups loaded:     {len(setup_data):,}")
    print(
        "OANDA bid/ask data:         "
        + (
            "YES"
            if candle_data["spread_pips"].notna().any()
            else "NO"
        )
    )

    dataset = build_market_database(
        candle_data,
        setup_data,
    )

    if dataset.empty:
        raise SystemExit(
            "\nNo historical examples were generated.\n"
            "Check whether setup timestamps overlap the candle data."
        )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    summary = build_summary(
        dataset
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    write_html_report(
        dataset,
        summary,
        report_path,
    )

    print()
    print(divider())
    print("HISTORICAL MARKET DATABASE")
    print(divider())

    print(f"Examples generated:          {len(dataset):,}")
    print(f"Dataset columns:             {len(dataset.columns):,}")
    print(
        f"Median favorable movement:   "
        f"{summary['median_mfe_r']:.2f}R"
    )
    print(
        f"Median adverse movement:     "
        f"{summary['median_mae_r']:.2f}R"
    )

    if summary.get("average_spread_pips") is not None:
        print(
            f"Average spread:              "
            f"{summary['average_spread_pips']:.2f} pips"
        )

    print()
    print("TARGET OUTCOMES")
    print(divider("-"))

    for target_r, values in summary["targets"].items():
        print(
            f"{target_r:>4}R before stop:          "
            f"{values['win_rate']:.1%} "
            f"({values['wins']}/{len(dataset)})"
        )

    print()
    print(divider())
    print("FILES SAVED")
    print(divider())

    print(f"Dataset:                     {dataset_path}")
    print(f"Summary:                     {summary_path}")
    print(f"Readable report:             {report_path}")

    print()
    print("Open the report with:")
    print()
    print(f"open {report_path}")
    print()
    print(divider())
    print("V9 Phase 1 completed successfully.")
    print(divider())


if __name__ == "__main__":
    main()
