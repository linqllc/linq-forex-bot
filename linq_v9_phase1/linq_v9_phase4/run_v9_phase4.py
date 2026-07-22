from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from linq_engine.config import Phase4Config
from linq_engine.core import load_candles, load_setups, load_phase1, build_dataset, predict_holdout, summarize, equity_curve, build_report

def divider(ch="=", n=108): return ch*n

def calibration_table(predictions):
    valid=predictions[predictions["holdout"] & predictions["probability_1r"].notna()].copy()
    bins=[0,.50,.60,.65,.70,.80,.90,1.000001]
    labels=["<50%","50–59%","60–64%","65–69%","70–79%","80–89%","90%+"]
    valid["confidence_band"]=pd.cut(valid["probability_1r"],bins=bins,labels=labels,right=False,include_lowest=True)
    return valid.groupby("confidence_band",observed=False).agg(predictions=("setup_id","size"),average_probability=("probability_1r","mean"),actual_win_rate=("actual_win","mean")).reset_index()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candles",default="data/cache/EUR_USD_M5.csv")
    p.add_argument("--setups",default="reports/v7/EUR_USD_automatic_setups.csv")
    p.add_argument("--phase1",default="reports/v9/EUR_USD_market_database.csv")
    p.add_argument("--output",default="reports/v9_phase4")
    p.add_argument("--holdout-trades",type=int,default=20)
    p.add_argument("--slippage-pips",type=float,default=.10)
    p.add_argument("--maximum-spread-pips",type=float,default=None)
    a=p.parse_args()
    cfg=Phase4Config(holdout_trades=a.holdout_trades,slippage_pips_each_side=a.slippage_pips,maximum_spread_pips=a.maximum_spread_pips)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    print(divider()); print("LINQ MARKET INTELLIGENCE ENGINE — V9 PHASE 4"); print(divider())
    print("FROZEN RULES")
    print(f"Probability threshold:           {cfg.probability_threshold:.0%}")
    print(f"Stop:                            {cfg.stop_atr:.2f} ATR")
    print(f"Target:                          {cfg.target_r:.2f}R")
    print(f"Reserved holdout setups:         {cfg.holdout_trades}")
    print(f"Slippage per side:               {cfg.slippage_pips_each_side:.2f} pips")
    candles=load_candles(a.candles); setups=load_setups(a.setups); phase1=load_phase1(a.phase1)
    dataset=build_dataset(candles,setups,phase1,cfg)
    if len(dataset)<=cfg.holdout_trades: raise SystemExit(f"Only {len(dataset)} usable setups; cannot reserve {cfg.holdout_trades}.")
    start_i=len(dataset)-cfg.holdout_trades
    holdout_start=dataset.iloc[start_i]["timestamp"]
    predictions=predict_holdout(dataset,start_i,cfg)
    holdout=predictions[predictions["holdout"]].copy()
    selected=holdout[holdout["selected"]].copy()
    summaries=[summarize("model_selected",selected,holdout),summarize("all_holdout_setups",holdout)]
    equity=equity_curve(selected); calibration=calibration_table(predictions)
    paths={
        "dataset":out/f"{cfg.pair}_phase4_dataset.csv",
        "predictions":out/f"{cfg.pair}_phase4_predictions.csv",
        "selected":out/f"{cfg.pair}_phase4_selected_trades.csv",
        "summary":out/f"{cfg.pair}_phase4_summary.csv",
        "json":out/f"{cfg.pair}_phase4_summary.json",
        "report":out/f"{cfg.pair}_phase4_report.html",
    }
    dataset.to_csv(paths["dataset"],index=False); predictions.to_csv(paths["predictions"],index=False); selected.to_csv(paths["selected"],index=False); pd.DataFrame(summaries).to_csv(paths["summary"],index=False)
    paths["json"].write_text(json.dumps({"pair":cfg.pair,"created_at_utc":datetime.now(timezone.utc).isoformat(),"holdout_start":str(holdout_start),"frozen_rules":cfg.__dict__,"summaries":summaries},indent=2,default=str),encoding="utf-8")
    build_report(summaries,selected,holdout,equity,calibration,holdout_start,cfg,paths["report"])
    print("\nDATA"); print(divider("-"))
    print(f"Candles loaded:                  {len(candles):,}")
    print(f"Usable setups:                   {len(dataset):,}")
    print(f"Training setups before holdout:  {start_i}")
    print(f"Final holdout setups:            {len(holdout)}")
    print(f"Holdout begins:                  {holdout_start}")
    print("\nFINAL HOLDOUT RESULTS"); print(divider("-"))
    for x in summaries:
        print(f"\n{x['strategy']}")
        print(f"  Trades:                        {x['trades']}")
        print(f"  Wins / losses:                 {x['wins']} / {x['losses']}")
        print(f"  Win rate:                      {x['win_rate']:.1%}")
        print(f"  Gross R:                       {x['gross_r']:+.3f}R")
        print(f"  Execution cost:                -{x['cost_r']:.3f}R")
        print(f"  Net R:                         {x['net_r']:+.3f}R")
        print(f"  Net expectancy:                {x['expectancy_r']:+.3f}R")
        print(f"  Profit factor:                 {x['profit_factor']:.3f}")
        print(f"  Maximum drawdown:              {x['max_drawdown_r']:.3f}R")
        print(f"  Longest losing streak:         {x['longest_losing_streak']}")
        if x["strategy"]=="model_selected":
            print(f"  Holdout AUC:                   {x['auc']:.3f}")
            print(f"  Holdout Brier:                 {x['brier']:.3f}")
    print("\nFILES SAVED"); print(divider("-"))
    for k,v in paths.items(): print(f"{k.title():<32}{v}")
    print(f"\nOpen with:\nopen {paths['report']}")
    print("\n"+divider()); print("V9 Phase 4 completed successfully."); print(divider())

if __name__=="__main__": main()
