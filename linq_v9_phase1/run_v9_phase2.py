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
MIN_TRAIN = 40
FORWARD_BARS = 288
TARGETS = (1.0, 1.25, 1.5, 2.0)

STOP_METHODS = {
    "existing_stop": None,
    "atr_0_75": 0.75,
    "atr_1_00": 1.00,
    "atr_1_25": 1.25,
    "atr_1_50": 1.50,
    "atr_2_00": 2.00,
}

SELECTION_THRESHOLDS = (0.50, 0.55, 0.60, 0.65)


def divider(ch="=", n=105):
    return ch * n


def find_col(df, aliases, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(
            f"Missing required column. Expected one of {aliases}. "
            f"Available: {list(df.columns)}"
        )
    return None


def suffix(value):
    return str(value).replace(".", "_")


def load_candles(path):
    raw = pd.read_csv(path)
    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "date"],
        "open": ["open", "o", "mid_o", "mid_open"],
        "high": ["high", "h", "mid_h", "mid_high"],
        "low": ["low", "l", "mid_l", "mid_low"],
        "close": ["close", "c", "mid_c", "mid_close"],
        "volume": ["volume", "tick_volume", "vol"],
        "bid_close": ["bid_close", "bid_c"],
        "ask_close": ["ask_close", "ask_c"],
    }
    rename = {}
    for canonical, names in aliases.items():
        col = find_col(raw, names, required=canonical in {"timestamp", "open", "high", "low", "close"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume", "bid_close", "ask_close"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df:
        df["volume"] = np.nan
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    pc = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - pc).abs(),
            (df["low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
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
        "timestamp": ["timestamp", "entry_timestamp", "entry_time", "setup_timestamp", "created_at"],
        "direction": ["direction", "side", "trade_direction"],
        "entry": ["entry", "entry_price", "price"],
        "stop": ["stop", "stop_price", "stop_loss", "sl"],
        "zone_low": ["zone_low", "distal", "lower_bound"],
        "zone_high": ["zone_high", "proximal", "upper_bound"],
    }
    rename = {}
    for canonical, names in aliases.items():
        col = find_col(raw, names, required=canonical in {"timestamp", "direction"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    if "setup_id" not in df:
        df["setup_id"] = [f"setup_{i:06d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    d = df["direction"].astype(str).str.lower().str.strip()

    normalized_direction = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object",
    )

    normalized_direction.loc[
        d.isin(
            [
                "long",
                "buy",
                "bull",
                "bullish",
                "demand",
                "1",
            ]
        )
    ] = "long"

    normalized_direction.loc[
        d.isin(
            [
                "short",
                "sell",
                "bear",
                "bearish",
                "supply",
                "-1",
            ]
        )
    ] = "short"

    df["direction"] = normalized_direction
    for c in ["entry", "stop", "zone_low", "zone_high"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return (
        df.dropna(subset=["timestamp", "direction"])
        .sort_values("timestamp")
        .drop_duplicates("setup_id")
        .reset_index(drop=True)
    )


def resolve_entry(setup, candle):
    if "entry" in setup.index and pd.notna(setup["entry"]):
        return float(setup["entry"])
    return float(candle["close"])


def resolve_existing_stop(setup):
    if "stop" in setup.index and pd.notna(setup["stop"]):
        return float(setup["stop"])
    if setup["direction"] == "long" and "zone_low" in setup.index and pd.notna(setup["zone_low"]):
        return float(setup["zone_low"])
    if setup["direction"] == "short" and "zone_high" in setup.index and pd.notna(setup["zone_high"]):
        return float(setup["zone_high"])
    return np.nan


def evaluate(direction, entry, stop, future):
    risk = abs(entry - stop)
    if not np.isfinite(risk) or risk <= 0:
        return None

    if direction == "long":
        stop_hit = future["low"] <= stop
        favorable = future["high"] - entry
        adverse = entry - future["low"]
    else:
        stop_hit = future["high"] >= stop
        favorable = entry - future["low"]
        adverse = future["high"] - entry

    stop_idx = np.flatnonzero(stop_hit.to_numpy())
    first_stop = int(stop_idx[0]) if len(stop_idx) else None

    out = {
        "risk_price": risk,
        "mfe_r": float((favorable / risk).max()),
        "mae_r": float((adverse / risk).max()),
        "bars_to_stop": first_stop + 1 if first_stop is not None else np.nan,
    }

    sign = 1 if direction == "long" else -1
    for target in TARGETS:
        price = entry + sign * risk * target
        hit = future["high"] >= price if direction == "long" else future["low"] <= price
        idx = np.flatnonzero(hit.to_numpy())
        first_target = int(idx[0]) if len(idx) else None
        won = first_target is not None and (first_stop is None or first_target < first_stop)
        out[f"hit_{suffix(target)}r"] = int(won)
        out[f"bars_to_{suffix(target)}r"] = first_target + 1 if first_target is not None else np.nan
    return out


def build_stop_dataset(candles, setups, phase1):
    phase1 = phase1.copy()
    phase1["timestamp"] = pd.to_datetime(phase1["timestamp"], utc=True, errors="coerce")
    feature_map = phase1.set_index("setup_id", drop=False).to_dict("index")

    times = candles["timestamp"]
    rows = []

    for _, setup in setups.iterrows():
        pos = int(times.searchsorted(setup["timestamp"], side="left"))
        if pos >= len(candles) or pos < 200:
            continue
        candle = candles.iloc[pos]
        future = candles.iloc[pos + 1 : pos + 1 + FORWARD_BARS]
        if future.empty or pd.isna(candle["atr"]) or candle["atr"] <= 0:
            continue

        entry = resolve_entry(setup, candle)
        existing = resolve_existing_stop(setup)

        base_features = feature_map.get(setup["setup_id"], {})
        clean_features = {
            k: v
            for k, v in base_features.items()
            if not (
                str(k).startswith("hit_")
                or str(k).startswith("bars_to_")
                or k in {
                    "mfe_r", "mae_r", "stop_hit", "risk_price",
                    "entry_price", "stop_price", "valid_risk", "bars_observed"
                }
            )
        }

        for method, atr_multiple in STOP_METHODS.items():
            if method == "existing_stop":
                stop = existing
                if not np.isfinite(stop):
                    continue
            else:
                stop = entry - candle["atr"] * atr_multiple if setup["direction"] == "long" else entry + candle["atr"] * atr_multiple

            result = evaluate(setup["direction"], entry, float(stop), future)
            if result is None:
                continue

            row = dict(clean_features)
            row.update(
                {
                    "setup_id": setup["setup_id"],
                    "timestamp": setup["timestamp"],
                    "direction": setup["direction"],
                    "stop_method": method,
                    "atr_stop_multiple": atr_multiple if atr_multiple is not None else np.nan,
                    "entry_price": entry,
                    "stop_price": float(stop),
                    "stop_distance_atr": abs(entry - float(stop)) / candle["atr"],
                    "spread_pips_at_entry": candle.get("spread_pips", np.nan),
                    "market_ema20_50_atr": (candle["ema20"] - candle["ema50"]) / candle["atr"],
                    "market_ema50_200_atr": (candle["ema50"] - candle["ema200"]) / candle["atr"],
                    "market_rsi": candle["rsi"],
                    "market_ret3": candle["ret3"],
                    "market_ret12": candle["ret12"],
                    "market_range_atr": candle["range_atr"],
                    "market_hour_utc": candle["hour_utc"],
                    "market_weekday": candle["weekday"],
                }
            )
            row.update(result)
            rows.append(row)

    return pd.DataFrame(rows)


def make_model(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
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
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                    ]
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("prep", ColumnTransformer(transformers)),
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


def walk_forward(group, target_col):
    group = group.sort_values("timestamp").reset_index(drop=True)
    excluded = {
        "setup_id", "timestamp", "entry_price", "stop_price",
        "risk_price", "mfe_r", "mae_r", "bars_to_stop"
    }
    excluded.update(c for c in group.columns if c.startswith("hit_") or c.startswith("bars_to_"))
    feature_cols = [c for c in group.columns if c not in excluded]
    X = group[feature_cols].copy()
    y = group[target_col].astype(int)
    probs = np.full(len(group), np.nan)

    for i in range(MIN_TRAIN, len(group)):
        train_y = y.iloc[:i]
        if train_y.nunique() < 2:
            probs[i] = float(train_y.mean())
            continue
        model = make_model(X.iloc[:i])
        model.fit(X.iloc[:i], train_y)
        probs[i] = model.predict_proba(X.iloc[i:i+1])[0, 1]

    result = group.copy()
    result["probability_1r"] = probs
    return result


def metric_row(method, wf, threshold):
    valid = wf["probability_1r"].notna()
    data = wf.loc[valid].copy()
    chosen = data[data["probability_1r"] >= threshold]
    if data.empty:
        return None
    auc = np.nan
    if data["hit_1_0r"].nunique() > 1:
        auc = float(roc_auc_score(data["hit_1_0r"], data["probability_1r"]))
    brier = float(brier_score_loss(data["hit_1_0r"], data["probability_1r"]))
    if chosen.empty:
        wins = 0
        win_rate = np.nan
        total_r = 0.0
        expectancy = np.nan
        pf = np.nan
    else:
        wins = int(chosen["hit_1_0r"].sum())
        losses = int(len(chosen) - wins)
        total_r = float(wins - losses)
        win_rate = wins / len(chosen)
        expectancy = total_r / len(chosen)
        pf = wins / losses if losses else np.inf
    return {
        "stop_method": method,
        "threshold": threshold,
        "walk_forward_rows": int(len(data)),
        "base_win_rate": float(data["hit_1_0r"].mean()),
        "auc": auc,
        "brier": brier,
        "trades": int(len(chosen)),
        "wins": wins,
        "win_rate": win_rate,
        "total_r": total_r,
        "expectancy_r": expectancy,
        "profit_factor": pf,
        "median_stop_atr": float(chosen["stop_distance_atr"].median()) if len(chosen) else np.nan,
        "median_mae_r": float(chosen["mae_r"].median()) if len(chosen) else np.nan,
        "median_mfe_r": float(chosen["mfe_r"].median()) if len(chosen) else np.nan,
    }


def build_report(summary_df, stop_stats, best, path):
    best_html = (
        f"<h2>Best current research configuration</h2>"
        f"<p><strong>{best['stop_method']}</strong> at probability threshold "
        f"<strong>{best['threshold']:.0%}</strong></p>"
        f"<p>{int(best['trades'])} trades · {best['win_rate']:.1%} win rate · "
        f"{best['expectancy_r']:+.3f}R expectancy · {best['total_r']:+.1f}R total</p>"
        if best is not None
        else "<p>No configuration met the minimum trade requirement.</p>"
    )
    summary_html = summary_df.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
    stats_html = stop_stats.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LINQ V9 Phase 2</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f6f8;color:#18212b;margin:0}}
.container{{max-width:1400px;margin:auto;padding:30px 20px}}
.section{{background:white;padding:20px;border-radius:12px;margin:18px 0;overflow:auto;box-shadow:0 2px 10px rgba(0,0,0,.05)}}
.notice{{background:#fff1d6;border-left:5px solid #d89900;padding:15px;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#edf1f4}}th,td{{padding:9px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}
</style></head><body><div class="container">
<h1>LINQ Market Intelligence Engine — V9 Phase 2</h1>
<p>Walk-forward opportunity selection and stop-distance research</p>
<div class="notice">Research only. The sample remains small. Stop choice is evaluated out of sample, but no live-trading conclusion should be made yet.</div>
<div class="section">{best_html}</div>
<div class="section"><h2>Walk-forward configurations</h2>{summary_html}</div>
<div class="section"><h2>Raw stop behavior</h2>{stats_html}</div>
</div></body></html>""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", default="data/cache/EUR_USD_M5.csv")
    parser.add_argument("--setups", default="reports/v7/EUR_USD_automatic_setups.csv")
    parser.add_argument("--phase1", default="reports/v9/EUR_USD_market_database.csv")
    parser.add_argument("--output", default="reports/v9_phase2")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(divider())
    print("LINQ MARKET INTELLIGENCE ENGINE — V9 PHASE 2")
    print(divider())

    c = load_candles(args.candles)
    s = load_setups(args.setups)
    p1 = pd.read_csv(args.phase1)

    stop_dataset = build_stop_dataset(c, s, p1)
    if stop_dataset.empty:
        raise SystemExit("No stop-analysis rows were generated.")

    stop_dataset_path = out / f"{PAIR}_stop_analysis_dataset.csv"
    stop_dataset.to_csv(stop_dataset_path, index=False)

    stop_stats = (
        stop_dataset.groupby("stop_method")
        .agg(
            examples=("setup_id", "size"),
            stop_atr_median=("stop_distance_atr", "median"),
            win_rate_1r=("hit_1_0r", "mean"),
            win_rate_1_25r=("hit_1_25r", "mean"),
            win_rate_1_5r=("hit_1_5r", "mean"),
            win_rate_2r=("hit_2_0r", "mean"),
            median_mfe_r=("mfe_r", "median"),
            median_mae_r=("mae_r", "median"),
        )
        .reset_index()
    )

    wf_all = []
    metric_rows = []

    for method, group in stop_dataset.groupby("stop_method"):
        wf = walk_forward(group, "hit_1_0r")
        wf_all.append(wf)
        for threshold in SELECTION_THRESHOLDS:
            row = metric_row(method, wf, threshold)
            if row:
                metric_rows.append(row)

    wf_predictions = pd.concat(wf_all, ignore_index=True)
    summary_df = pd.DataFrame(metric_rows).sort_values(
        ["expectancy_r", "win_rate", "trades"],
        ascending=[False, False, False],
        na_position="last",
    )

    eligible = summary_df[
        (summary_df["trades"] >= 8)
        & summary_df["expectancy_r"].notna()
    ]
    best = eligible.iloc[0].to_dict() if not eligible.empty else None

    predictions_path = out / f"{PAIR}_walk_forward_stop_predictions.csv"
    summary_path = out / f"{PAIR}_stop_selection_summary.csv"
    json_path = out / f"{PAIR}_phase2_summary.json"
    report_path = out / f"{PAIR}_phase2_report.html"

    wf_predictions.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "pair": PAIR,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stop_methods": list(STOP_METHODS),
        "minimum_train_rows": MIN_TRAIN,
        "selection_thresholds": list(SELECTION_THRESHOLDS),
        "best_configuration": best,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    build_report(summary_df, stop_stats, best, report_path)

    print(f"Candles loaded:                  {len(c):,}")
    print(f"Setups loaded:                   {len(s):,}")
    print(f"Stop-analysis rows:              {len(stop_dataset):,}")
    print(f"Stop methods tested:             {stop_dataset['stop_method'].nunique()}")

    print("\nRAW 1R WIN RATE BY STOP")
    print(divider("-"))
    for _, row in stop_stats.sort_values("win_rate_1r", ascending=False).iterrows():
        print(
            f"{row['stop_method']:<18} "
            f"1R={row['win_rate_1r']:.1%}  "
            f"1.25R={row['win_rate_1_25r']:.1%}  "
            f"median stop={row['stop_atr_median']:.2f} ATR"
        )

    print("\nBEST WALK-FORWARD CONFIGURATION")
    print(divider("-"))
    if best:
        print(f"Stop method:                     {best['stop_method']}")
        print(f"Probability threshold:           {best['threshold']:.0%}")
        print(f"Trades:                          {int(best['trades'])}")
        print(f"Wins:                            {int(best['wins'])}")
        print(f"Win rate:                        {best['win_rate']:.1%}")
        print(f"Total return at 1:1:             {best['total_r']:+.1f}R")
        print(f"Expectancy:                      {best['expectancy_r']:+.3f}R")
        print(f"Profit factor:                   {best['profit_factor']:.3f}")
        print(f"Walk-forward AUC:                {best['auc']:.3f}")
    else:
        print("No configuration produced at least 8 selected trades.")

    print("\nFILES SAVED")
    print(divider("-"))
    print(f"Stop dataset:                    {stop_dataset_path}")
    print(f"Walk-forward predictions:        {predictions_path}")
    print(f"Configuration summary:           {summary_path}")
    print(f"Readable report:                 {report_path}")
    print(f"\nOpen with:\nopen {report_path}")


if __name__ == "__main__":
    main()
