from __future__ import annotations
import numpy as np
import pandas as pd


def performance_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "net_r": 0.0,
        }

    r = trades["result_r"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    equity = r.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": int(len(trades)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "average_win_r": float(wins.mean()) if len(wins) else 0.0,
        "average_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(profit_factor),
        "max_drawdown_r": float(drawdown.min()) if len(drawdown) else 0.0,
        "net_r": float(r.sum()),
    }
