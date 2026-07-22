from __future__ import annotations
import math
from typing import Iterable, Callable
import numpy as np
import pandas as pd


def reward_grid(start: float, stop: float, step: float) -> list[float]:
    return [round(x, 4) for x in np.arange(start, stop + step/2, step)]


def evaluate_ratio(setups: pd.DataFrame, ratio: float) -> dict:
    if setups.empty:
        return {"reward_to_risk":ratio,"trades":0,"win_rate":0.0,"expectancy_r":0.0,"profit_factor":0.0,"max_drawdown_r":0.0,"net_r":0.0}
    results = np.where(setups["max_favorable_r"].to_numpy(float) >= ratio, ratio, setups["final_r"].clip(lower=-1.0).to_numpy(float))
    wins = results[results > 0]; losses = results[results < 0]
    equity = np.cumsum(results); peaks = np.maximum.accumulate(np.r_[0.0,equity])[:-1]; dd = equity - peaks
    gross_loss = abs(losses.sum())
    return {
        "reward_to_risk":float(ratio), "trades":int(len(results)), "win_rate":float((results>0).mean()),
        "expectancy_r":float(results.mean()), "average_win_r":float(wins.mean()) if len(wins) else 0.0,
        "average_loss_r":float(losses.mean()) if len(losses) else 0.0,
        "profit_factor":float(wins.sum()/gross_loss) if gross_loss else float("inf"),
        "max_drawdown_r":float(dd.min()) if len(dd) else 0.0, "net_r":float(results.sum())
    }


def optimize_setups(setups: pd.DataFrame, ratios: Iterable[float], minimum_trades: int = 20, progress_callback: Callable[[int, int, str], None] | None = None) -> tuple[pd.DataFrame, dict]:
    ratio_list = list(ratios)
    rows = []
    total = max(1, len(ratio_list))
    for idx, r in enumerate(ratio_list, start=1):
        rows.append(evaluate_ratio(setups, r))
        if progress_callback:
            progress_callback(idx, total, f"testing {r:.2f}R")
    results = pd.DataFrame(rows)
    eligible = results[results.trades >= minimum_trades].copy()
    if eligible.empty: eligible = results.copy()
    pf = eligible.profit_factor.replace([math.inf,-math.inf],10.0).clip(upper=10.0)
    eligible["selection_score"] = eligible.expectancy_r*100 + pf*2 + eligible.max_drawdown_r*0.25
    overall = eligible.sort_values(["selection_score","trades"], ascending=False).iloc[0]
    return results, {"best_overall_ratio":float(overall.reward_to_risk),"best_overall_expectancy_r":float(overall.expectancy_r),"best_overall_profit_factor":float(overall.profit_factor),"best_overall_net_r":float(overall.net_r),"best_overall_max_drawdown_r":float(overall.max_drawdown_r),"best_overall_trades":int(overall.trades)}


def chronological_split(setups: pd.DataFrame, train_fraction: float, validation_fraction: float):
    if setups.empty or "entry_time" not in setups.columns:
        empty = setups.copy()
        return empty, empty.copy(), empty.copy()
    ordered = setups.sort_values("entry_time").reset_index(drop=True)
    n=len(ordered); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()
