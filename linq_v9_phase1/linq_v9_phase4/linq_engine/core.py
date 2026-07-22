from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import Phase4Config

LEAKAGE = {
    "timestamp","entry_price","stop_price","target_price","risk_price",
    "mfe_r","mae_r","stop_hit","valid_risk","bars_observed","result_r",
    "gross_result_r","net_result_r","exit_reason","bars_held","exit_price",
    "spread_cost_r","slippage_cost_r","total_cost_r","selected",
    "probability_1r","actual_win","holdout"
}

def find_col(df, aliases, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    if required:
        raise ValueError(f"Missing one of {aliases}. Available: {list(df.columns)}")
    return None

def load_candles(path):
    raw = pd.read_csv(path)
    aliases = {
        "timestamp":["timestamp","time","datetime","date"],
        "open":["open","o","mid_open","mid_o"],
        "high":["high","h","mid_high","mid_h"],
        "low":["low","l","mid_low","mid_l"],
        "close":["close","c","mid_close","mid_c"],
        "bid_close":["bid_close","bid_c"],
        "ask_close":["ask_close","ask_c"],
    }
    rename = {}
    for canonical, choices in aliases.items():
        col = find_col(raw, choices, canonical in {"timestamp","open","high","low","close"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in [x for x in aliases if x != "timestamp" and x in df]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    pc = df["close"].shift()
    tr = pd.concat([(df["high"]-df["low"]), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    d = df["close"].diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["ret3"] = df["close"].pct_change(3)
    df["ret12"] = df["close"].pct_change(12)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"]
    df["hour_utc"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["spread_pips"] = ((df["ask_close"]-df["bid_close"])/0.0001) if {"bid_close","ask_close"}.issubset(df.columns) else np.nan
    return df

def load_setups(path):
    raw = pd.read_csv(path)
    aliases = {
        "setup_id":["setup_id","id","zone_id"],
        "timestamp":["timestamp","entry_timestamp","entry_time","setup_timestamp","created_at"],
        "direction":["direction","side","trade_direction"],
        "entry":["entry","entry_price","price"],
    }
    rename = {}
    for canonical, choices in aliases.items():
        col = find_col(raw, choices, canonical in {"timestamp","direction"})
        if col is not None:
            rename[col] = canonical
    df = raw.rename(columns=rename).copy()
    if "setup_id" not in df:
        df["setup_id"] = [f"setup_{i:06d}" for i in range(len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    d = df["direction"].astype(str).str.lower().str.strip()
    norm = pd.Series(pd.NA, index=df.index, dtype="object")
    norm.loc[d.isin(["long","buy","bull","bullish","demand","1"])] = "long"
    norm.loc[d.isin(["short","sell","bear","bearish","supply","-1"])] = "short"
    df["direction"] = norm
    if "entry" in df:
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
    return df.dropna(subset=["timestamp","direction"]).sort_values("timestamp").drop_duplicates("setup_id").reset_index(drop=True)

def load_phase1(path):
    df = pd.read_csv(path)
    if "setup_id" not in df:
        raise ValueError("Phase 1 database must contain setup_id.")
    return df

def clean_features(row):
    out = {}
    for k, v in row.items():
        s = str(k)
        if k in LEAKAGE or s.startswith("hit_") or s.startswith("bars_to_") or s.startswith("probability"):
            continue
        out[k] = v
    return out

def evaluate(direction, entry, stop, future, spread_pips, cfg):
    risk = abs(entry-stop)
    if not np.isfinite(risk) or risk <= 0 or future.empty:
        return None
    target = entry + risk if direction == "long" else entry - risk
    if direction == "long":
        target_hits = future["high"] >= target
        stop_hits = future["low"] <= stop
        fav = future["high"] - entry
        adv = entry - future["low"]
    else:
        target_hits = future["low"] <= target
        stop_hits = future["high"] >= stop
        fav = entry - future["low"]
        adv = future["high"] - entry
    ti = np.flatnonzero(target_hits.to_numpy())
    si = np.flatnonzero(stop_hits.to_numpy())
    ft = int(ti[0]) if len(ti) else None
    fs = int(si[0]) if len(si) else None
    won = ft is not None and (fs is None or ft < fs)
    if won:
        gross, bar, reason, exit_price = cfg.target_r, ft, "target", target
    elif fs is not None:
        gross, bar, reason, exit_price = -1.0, fs, "stop", stop
    else:
        last = float(future["close"].iloc[-1])
        signed = last-entry if direction=="long" else entry-last
        gross, bar, reason, exit_price = float(np.clip(signed/risk,-1,cfg.target_r)), len(future)-1, "time_exit", last
    spread = float(spread_pips) if np.isfinite(spread_pips) else 0.0
    spread_r = spread*cfg.pip_size/risk
    slip_r = 2*cfg.slippage_pips_each_side*cfg.pip_size/risk
    cost = spread_r + slip_r
    return {
        "actual_win":int(won),"gross_result_r":gross,"net_result_r":gross-cost,
        "exit_reason":reason,"bars_held":bar+1,"exit_price":exit_price,
        "target_price":target,"risk_price":risk,"mfe_r":float((fav/risk).max()),
        "mae_r":float((adv/risk).max()),"spread_cost_r":spread_r,
        "slippage_cost_r":slip_r,"total_cost_r":cost
    }

def build_dataset(candles, setups, phase1, cfg):
    fmap = phase1.set_index("setup_id", drop=False).to_dict("index")
    times = candles["timestamp"]
    rows = []
    for _, setup in setups.iterrows():
        pos = int(times.searchsorted(setup["timestamp"], side="left"))
        if pos >= len(candles) or pos < 200:
            continue
        candle = candles.iloc[pos]
        atr = float(candle["atr"]) if pd.notna(candle["atr"]) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue
        spread = candle.get("spread_pips", np.nan)
        if cfg.maximum_spread_pips is not None and np.isfinite(spread) and spread > cfg.maximum_spread_pips:
            continue
        future = candles.iloc[pos+1:pos+1+cfg.forward_bars]
        if future.empty:
            continue
        entry = float(setup["entry"]) if "entry" in setup.index and pd.notna(setup["entry"]) else float(candle["close"])
        stop = entry-cfg.stop_atr*atr if setup["direction"]=="long" else entry+cfg.stop_atr*atr
        outcome = evaluate(setup["direction"], entry, stop, future, spread, cfg)
        if outcome is None:
            continue
        row = clean_features(fmap.get(setup["setup_id"], {}))
        row.update({
            "setup_id":setup["setup_id"],"timestamp":setup["timestamp"],"direction":setup["direction"],
            "entry_price":entry,"stop_price":stop,"stop_distance_atr":cfg.stop_atr,
            "spread_pips_at_entry":spread,"atr_at_entry":atr,
            "market_ema20_50_atr":(candle["ema20"]-candle["ema50"])/atr,
            "market_ema50_200_atr":(candle["ema50"]-candle["ema200"])/atr,
            "market_rsi":candle["rsi"],"market_ret3":candle["ret3"],
            "market_ret12":candle["ret12"],"market_range_atr":candle["range_atr"],
            "market_hour_utc":int(candle["hour_utc"]),"market_weekday":int(candle["weekday"]),
        })
        row.update(outcome)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

def feature_columns(df):
    excluded = LEAKAGE | {"setup_id","strategy"}
    return [c for c in df.columns if c not in excluded]

def make_model(X):
    numeric = X.select_dtypes(include=["number","bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    tr = []
    if numeric:
        tr.append(("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric))
    if categorical:
        tr.append(("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))]), categorical))
    return Pipeline([("prep",ColumnTransformer(tr)),("model",LogisticRegression(max_iter=3000,class_weight="balanced",C=0.25,random_state=42))])

def predict_holdout(dataset, holdout_start, cfg):
    data = dataset.copy()
    cols = feature_columns(data)
    X = data[cols].copy()
    y = data["actual_win"].astype(int)
    probs = np.full(len(data), np.nan)
    hold = np.zeros(len(data), dtype=bool)
    hold[holdout_start:] = True
    for i in range(holdout_start, len(data)):
        if i < cfg.minimum_training_rows:
            continue
        train_y = y.iloc[:i]
        if train_y.nunique() < 2:
            probs[i] = float(train_y.mean())
        else:
            model = make_model(X.iloc[:i])
            model.fit(X.iloc[:i], train_y)
            probs[i] = model.predict_proba(X.iloc[i:i+1])[0,1]
    data["holdout"] = hold
    data["probability_1r"] = probs
    data["selected"] = data["holdout"] & data["probability_1r"].notna() & (data["probability_1r"] >= cfg.probability_threshold)
    return data

def max_dd(values):
    arr = np.asarray(list(values), dtype=float)
    if len(arr)==0: return np.nan
    eq = np.concatenate([[0.0],np.cumsum(arr)])
    peaks = np.maximum.accumulate(eq)
    return float((peaks-eq).max())

def losing_streak(values):
    longest=current=0
    for v in values:
        if v < 0:
            current += 1; longest=max(longest,current)
        else: current=0
    return longest

def summarize(name, trades, prediction_rows=None):
    wins = int((trades["net_result_r"]>0).sum())
    losses = int((trades["net_result_r"]<0).sum())
    gp = float(trades.loc[trades["net_result_r"]>0,"net_result_r"].sum())
    gl = abs(float(trades.loc[trades["net_result_r"]<0,"net_result_r"].sum()))
    auc=brier=np.nan
    if prediction_rows is not None:
        valid = prediction_rows.dropna(subset=["probability_1r"])
        if len(valid):
            brier=float(brier_score_loss(valid["actual_win"],valid["probability_1r"]))
            if valid["actual_win"].nunique()>1:
                auc=float(roc_auc_score(valid["actual_win"],valid["probability_1r"]))
    return {
        "strategy":name,"trades":len(trades),"wins":wins,"losses":losses,
        "win_rate":wins/len(trades) if len(trades) else np.nan,
        "gross_r":float(trades["gross_result_r"].sum()) if len(trades) else 0.0,
        "cost_r":float(trades["total_cost_r"].sum()) if len(trades) else 0.0,
        "net_r":float(trades["net_result_r"].sum()) if len(trades) else 0.0,
        "expectancy_r":float(trades["net_result_r"].mean()) if len(trades) else np.nan,
        "profit_factor":gp/gl if gl>0 else np.inf if gp>0 else np.nan,
        "max_drawdown_r":max_dd(trades["net_result_r"]),
        "longest_losing_streak":losing_streak(trades["net_result_r"]),
        "auc":auc,"brier":brier
    }

def equity_curve(trades):
    df = trades.sort_values("timestamp").copy()
    df["trade_number"] = np.arange(1,len(df)+1)
    df["equity_r"] = df["net_result_r"].cumsum()
    peaks = np.maximum.accumulate(np.concatenate([[0.0],df["equity_r"].to_numpy()]))[1:]
    df["drawdown_r"] = peaks-df["equity_r"]
    return df

def build_report(summaries, selected, benchmark, equity, calibration, holdout_start, cfg, path):
    def table(df):
        if df is None or df.empty: return "<p>No rows available.</p>"
        return df.to_html(index=False,border=0,float_format=lambda x:f"{x:.3f}")
    cards=""
    for x in summaries:
        cards += f"<div class='card'><h3>{x['strategy']}</h3><div class='metric'>{x['trades']} trades</div><p>{x['win_rate']:.1%} win rate</p><p>{x['net_r']:+.2f}R net</p><p>{x['expectancy_r']:+.3f}R expectancy</p><p>PF {x['profit_factor']:.2f} · DD {x['max_drawdown_r']:.2f}R</p></div>"
    cols=[c for c in ["timestamp","setup_id","direction","probability_1r","entry_price","stop_price","target_price","spread_pips_at_entry","gross_result_r","total_cost_r","net_result_r","exit_reason","bars_held"] if c in selected]
    path.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>LINQ V9 Phase 4</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f5f7;color:#18212b;margin:0}}.container{{max-width:1450px;margin:auto;padding:30px 20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}.card,.section{{background:white;padding:20px;border-radius:12px;box-shadow:0 2px 10px rgba(0,0,0,.05);margin:18px 0;overflow:auto}}.card{{margin:0}}.metric{{font-size:28px;font-weight:700}}.locked{{background:#eaf4ff;border-left:5px solid #3977b8;padding:16px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th{{background:#edf1f4}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}}</style></head><body><div class='container'>
<h1>LINQ Market Intelligence Engine — V9 Phase 4</h1><p>Frozen final-holdout validation</p>
<div class='locked'><strong>Frozen:</strong> threshold {cfg.probability_threshold:.0%} · stop {cfg.stop_atr:.2f} ATR · target {cfg.target_r:.2f}R · last {cfg.holdout_trades} setups held out · slippage {cfg.slippage_pips_each_side:.2f} pip per side · holdout begins {holdout_start}</div>
<div class='grid'>{cards}</div>
<div class='section'><h2>Strategy comparison</h2>{table(pd.DataFrame(summaries))}</div>
<div class='section'><h2>Selected ledger</h2>{table(selected[cols])}</div>
<div class='section'><h2>Equity and drawdown</h2>{table(equity[[c for c in ["trade_number","timestamp","net_result_r","equity_r","drawdown_r"] if c in equity]])}</div>
<div class='section'><h2>Calibration</h2>{table(calibration)}</div>
<div class='section'><h2>All holdout setups</h2>{table(benchmark[cols])}</div>
</div></body></html>""",encoding="utf-8")
