from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

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
