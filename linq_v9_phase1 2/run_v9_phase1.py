from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIR = "EUR_USD"
TARGETS = (1.0, 1.25, 1.5, 2.0, 3.0)
LOOKBACK_BARS = 100
FORWARD_BARS = 288

def pick(df, names, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise ValueError(f"Missing column; expected one of {names}")
    return None

def candles(path):
    raw = pd.read_csv(path)
    rename = {}
    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "date"],
        "open": ["open", "o", "mid_o"],
        "high": ["high", "h", "mid_h"],
        "low": ["low", "l", "mid_l"],
        "close": ["close", "c", "mid_c"],
        "volume": ["volume", "tick_volume", "vol"],
    }
    for canonical, names in aliases.items():
        c = pick(raw, names, required=canonical != "volume")
        if c is not None:
            rename[c] = canonical
    df = raw.rename(columns=rename).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df:
        df["volume"] = np.nan
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def setups(path):
    raw = pd.read_csv(path)
    aliases = {
        "setup_id": ["setup_id", "id", "zone_id"],
        "timestamp": ["entry_timestamp", "entry_time", "timestamp", "setup_timestamp", "created_at"],
        "direction": ["direction", "side", "trade_direction"],
        "entry": ["entry_price", "entry", "price"],
        "stop": ["stop_price", "stop", "stop_loss", "sl"],
        "zone_high": ["zone_high", "proximal", "upper_bound"],
        "zone_low": ["zone_low", "distal", "lower_bound"],
    }
    rename = {}
    for canonical, names in aliases.items():
        c = pick(raw, names, required=canonical in {"timestamp", "direction"})
        if c is not None:
            rename[c] = canonical
    df = raw.rename(columns=rename).copy()
    if "setup_id" not in df:
        df["setup_id"] = [f"setup_{i:06d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    text = df["direction"].astype(str).str.lower().str.strip()
    df["direction"] = np.where(text.isin(["long", "buy", "bull", "bullish", "demand", "1"]), "long",
                       np.where(text.isin(["short", "sell", "bear", "bearish", "supply", "-1"]), "short", np.nan))
    for c in ["entry", "stop", "zone_high", "zone_low"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp", "direction"]).sort_values("timestamp").drop_duplicates("setup_id").reset_index(drop=True)

def indicators(df):
    x = df.copy()
    pc = x["close"].shift()
    tr = pd.concat([(x["high"]-x["low"]), (x["high"]-pc).abs(), (x["low"]-pc).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    x["rsi"] = 100 - 100/(1 + gain/loss.replace(0, np.nan))
    x["ret1"] = x["close"].pct_change()
    x["ret3"] = x["close"].pct_change(3)
    x["ret12"] = x["close"].pct_change(12)
    x["range"] = x["high"] - x["low"]
    x["body"] = (x["close"] - x["open"]).abs()
    x["high20"] = x["high"].rolling(20).max()
    x["low20"] = x["low"].rolling(20).min()
    x["vol20"] = x["ret1"].rolling(20).std()
    x["hour"] = x["timestamp"].dt.hour
    x["weekday"] = x["timestamp"].dt.dayofweek
    return x

def entry_stop(setup, candle):
    entry = float(setup["entry"]) if "entry" in setup and pd.notna(setup["entry"]) else float(candle["close"])
    if "stop" in setup and pd.notna(setup["stop"]):
        stop = float(setup["stop"])
    elif setup["direction"] == "long" and "zone_low" in setup and pd.notna(setup["zone_low"]):
        stop = float(setup["zone_low"])
    elif setup["direction"] == "short" and "zone_high" in setup and pd.notna(setup["zone_high"]):
        stop = float(setup["zone_high"])
    else:
        atr = float(candle["atr"]) if pd.notna(candle["atr"]) else float(candle["range"])
        stop = entry - atr if setup["direction"] == "long" else entry + atr
    return entry, stop

def outcomes(direction, entry, stop, future):
    risk = abs(entry-stop)
    if not np.isfinite(risk) or risk <= 0:
        return {"valid_risk": 0}
    sign = 1 if direction == "long" else -1
    favorable = future["high"]-entry if direction == "long" else entry-future["low"]
    adverse = entry-future["low"] if direction == "long" else future["high"]-entry
    stop_hit = future["low"] <= stop if direction == "long" else future["high"] >= stop
    stop_idx = np.flatnonzero(stop_hit.to_numpy())
    first_stop = int(stop_idx[0]) if len(stop_idx) else None
    result = {
        "valid_risk": 1, "entry_price": entry, "stop_price": stop, "risk_price": risk,
        "mfe_r": float((favorable/risk).max()),
        "mae_r": float((adverse/risk).max()),
        "bars_to_stop": first_stop+1 if first_stop is not None else np.nan,
    }
    for target_r in TARGETS:
        target = entry + sign*risk*target_r
        hit = future["high"] >= target if direction == "long" else future["low"] <= target
        idx = np.flatnonzero(hit.to_numpy())
        first = int(idx[0]) if len(idx) else None
        suffix = str(target_r).replace(".", "_")
        result[f"hit_{suffix}r_before_stop"] = int(first is not None and (first_stop is None or first < first_stop))
        result[f"bars_to_{suffix}r"] = first+1 if first is not None else np.nan
    return result

def build(c, s):
    c = indicators(c)
    times = c["timestamp"].astype("int64").to_numpy()
    rows = []
    for _, setup in s.iterrows():
        pos = int(np.searchsorted(times, int(setup["timestamp"].value), side="left"))
        if pos >= len(c) or pos < 50:
            continue
        candle = c.iloc[pos]
        future = c.iloc[pos+1:pos+1+FORWARD_BARS]
        if future.empty or pd.isna(candle["atr"]):
            continue
        entry, stop = entry_stop(setup, candle)
        atr = float(candle["atr"])
        row = {
            "setup_id": setup["setup_id"], "timestamp": setup["timestamp"], "direction": setup["direction"],
            "close": candle["close"], "atr": atr, "atr_pct": atr/candle["close"],
            "ema_spread_atr": (candle["ema20"]-candle["ema50"])/atr,
            "trend_alignment": int((setup["direction"]=="long" and candle["ema20"]>candle["ema50"]) or
                                   (setup["direction"]=="short" and candle["ema20"]<candle["ema50"])),
            "rsi": candle["rsi"], "return_1": candle["ret1"], "return_3": candle["ret3"],
            "return_12": candle["ret12"], "range_atr": candle["range"]/atr, "body_atr": candle["body"]/atr,
            "distance_high20_atr": (candle["high20"]-candle["close"])/atr,
            "distance_low20_atr": (candle["close"]-candle["low20"])/atr,
            "volatility_20": candle["vol20"], "hour_utc": candle["hour"], "weekday": candle["weekday"],
            "london_session": int(7 <= candle["hour"] <= 11),
            "new_york_session": int(12 <= candle["hour"] <= 16),
            "session_overlap": int(12 <= candle["hour"] <= 15),
        }
        for col, val in setup.items():
            if col not in {"setup_id", "timestamp", "direction", "entry", "stop"} and col not in row:
                row[f"setup_{col}"] = val
        row.update(outcomes(setup["direction"], entry, stop, future))
        rows.append(row)
    return pd.DataFrame(rows)

def report(df, summary, path):
    target_rows = "".join(
        f"<tr><td>{r}R</td><td>{v['wins']}</td><td>{v['win_rate']:.1%}</td></tr>"
        for r, v in summary["targets"].items()
    )
    preview = df.tail(25).to_html(index=False, border=0) if len(df) else "<p>No rows generated.</p>"
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LINQ V9</title>
<style>body{{font-family:Arial;background:#f4f6f8;padding:24px}}section{{background:white;padding:20px;margin:16px 0;border-radius:10px;overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}</style></head>
<body><h1>LINQ Market Intelligence Engine — V9 Phase 1</h1>
<section><h2>Dataset</h2><p>{summary['rows']} historical examples · {summary['columns']} columns</p></section>
<section><h2>Target outcomes before stop</h2><table><tr><th>Target</th><th>Wins</th><th>Win rate</th></tr>{target_rows}</table></section>
<section><h2>Latest rows</h2>{preview}</section></body></html>""", encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candles", default="data/cache/EUR_USD_M5.csv")
    p.add_argument("--setups", default="reports/v7/EUR_USD_automatic_setups.csv")
    p.add_argument("--output", default="reports/v9")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    c, s = candles(a.candles), setups(a.setups)
    df = build(c, s)
    dataset_path = out/f"{PAIR}_market_database.csv"
    summary_path = out/f"{PAIR}_market_database_summary.json"
    report_path = out/f"{PAIR}_market_database_report.html"
    df.to_csv(dataset_path, index=False)
    summary = {"pair":PAIR, "created_at_utc":datetime.now(timezone.utc).isoformat(),
               "rows":int(len(df)), "columns":int(len(df.columns)), "targets":{}}
    for r in TARGETS:
        col = f"hit_{str(r).replace('.','_')}r_before_stop"
        if col in df:
            summary["targets"][str(r)] = {"wins":int(df[col].sum()), "win_rate":float(df[col].mean())}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(df, summary, report_path)
    print("="*90)
    print("LINQ V9 PHASE 1 COMPLETE")
    print("="*90)
    print(f"Candles loaded:       {len(c):,}")
    print(f"Setups loaded:        {len(s):,}")
    print(f"Examples generated:   {len(df):,}")
    for r, v in summary["targets"].items():
        print(f"{r:>4}R win rate:       {v['win_rate']:.1%}")
    print(f"\nDataset: {dataset_path}")
    print(f"Report:  {report_path}")
    print(f"\nOpen with:\nopen {report_path}")

if __name__ == "__main__":
    main()
