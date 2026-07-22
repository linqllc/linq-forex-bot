from __future__ import annotations
import math
from typing import Iterable
import pandas as pd
from src.metrics import performance_summary
from src.strategy import run_strategy

def reward_grid(start: float, stop: float, step: float) -> list[float]:
    values=[]; current=start
    while current <= stop + 1e-9:
        values.append(round(current, 4)); current += step
    return values

def optimize_reward_to_risk(candles: pd.DataFrame, instrument: str, config: dict, ratios: Iterable[float]) -> tuple[pd.DataFrame, dict]:
    rows=[]
    for ratio in ratios:
        trades=run_strategy(candles, instrument, config, reward_to_risk=float(ratio))
        summary=performance_summary(trades); summary["reward_to_risk"]=float(ratio); rows.append(summary)
    results=pd.DataFrame(rows)
    if results.empty: return results, {}
    min_trades=int(config.get("optimizer",{}).get("minimum_trades",20))
    eligible=results[results["trades"]>=min_trades].copy()
    if eligible.empty: eligible=results.copy()
    finite_pf=eligible["profit_factor"].replace([math.inf,-math.inf],10.0).clip(upper=10.0)
    eligible["selection_score"]=eligible["expectancy_r"]*100 + finite_pf*2 + eligible["max_drawdown_r"]*0.25
    overall=eligible.sort_values(["selection_score","trades"],ascending=False).iloc[0]
    profit=eligible.sort_values(["net_r","expectancy_r"],ascending=False).iloc[0]
    expectancy=eligible.sort_values(["expectancy_r","profit_factor"],ascending=False).iloc[0]
    wins=eligible.sort_values(["win_rate","net_r"],ascending=False).iloc[0]
    selection={
      "instrument":instrument,"minimum_trades_rule":min_trades,
      "best_overall_ratio":float(overall["reward_to_risk"]),
      "best_profit_ratio":float(profit["reward_to_risk"]),
      "best_expectancy_ratio":float(expectancy["reward_to_risk"]),
      "best_win_rate_ratio":float(wins["reward_to_risk"]),
      "best_overall_expectancy_r":float(overall["expectancy_r"]),
      "best_overall_profit_factor":float(overall["profit_factor"]),
      "best_overall_net_r":float(overall["net_r"]),
      "best_overall_max_drawdown_r":float(overall["max_drawdown_r"]),
      "best_overall_trades":int(overall["trades"])
    }
    return results.sort_values("reward_to_risk"), selection
