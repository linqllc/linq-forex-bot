from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Config


def simulate(df: pd.DataFrame, setups: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if setups.empty:
        return setups.copy()

    cost = (cfg.spread_pips + cfg.slippage_pips) * cfg.pip_size
    trades = []

    for setup in setups.to_dict("records"):
        i = int(setup["entry_index"])
        direction = setup["direction"]
        raw_entry = float(setup["entry"])
        entry = raw_entry + cost if direction == "long" else raw_entry - cost
        stop = float(setup["stop"])
        target = float(setup["target"])
        initial_risk = entry-stop if direction == "long" else stop-entry
        if initial_risk <= 0:
            continue

        exit_price = float(df.iloc[min(i+cfg.max_holding_bars, len(df)-1)].close)
        exit_index = min(i+cfg.max_holding_bars, len(df)-1)
        exit_reason = "timeout"

        for j in range(i+1, min(i+cfg.max_holding_bars+1, len(df))):
            bar = df.iloc[j]
            stop_hit = bar.low <= stop if direction == "long" else bar.high >= stop
            target_hit = bar.high >= target if direction == "long" else bar.low <= target

            if stop_hit and target_hit:
                if cfg.intrabar_policy == "target_first":
                    stop_hit = False
                else:
                    target_hit = False

            if stop_hit:
                exit_price, exit_index, exit_reason = stop, j, "stop"
                break
            if target_hit:
                exit_price, exit_index, exit_reason = target, j, "target"
                break

        pnl = exit_price-entry if direction == "long" else entry-exit_price
        r_multiple = pnl/initial_risk
        setup.update({
            "executed_entry": entry,
            "exit_index": exit_index,
            "exit_time": df.iloc[exit_index].time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "r_multiple": r_multiple,
            "won": r_multiple > 0,
        })
        trades.append(setup)

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": None, "expectancy_r": None,
            "profit_factor": None, "max_drawdown_r": None, "total_r": 0.0,
        }
    r = trades["r_multiple"].astype(float)
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "trades": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins/losses) if losses > 0 else None,
        "max_drawdown_r": float(drawdown.min()),
        "total_r": float(r.sum()),
    }


def grouped_reports(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {}
    outputs = {}
    for name, col in {
        "freshness": "prior_touches",
        "depth_rank": "depth_rank",
        "zone_score": "zone_score",
        "reinforcement": "reinforcements",
        "direction": "direction",
    }.items():
        rows = []
        for value, group in trades.groupby(col, dropna=False):
            row = {col: value, **summarize(group)}
            rows.append(row)
        outputs[name] = pd.DataFrame(rows)
    return outputs
