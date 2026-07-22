from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


def label_outcomes(candles: pd.DataFrame, setups: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    if setups.empty:
        return setups.copy()

    labeled = setups.copy()
    outcomes: list[dict] = []

    for setup in labeled.itertuples(index=False):
        entry_idx = int(setup.entry_index)
        entry = float(setup.entry_price)
        risk = float(setup.atr) * cfg.stop_atr
        if not np.isfinite(risk) or risk <= 0:
            outcomes.append({"outcome": "invalid", "r_multiple": np.nan})
            continue

        if setup.direction == "long":
            stop = entry - risk
            target = entry + risk * cfg.target_r
        else:
            stop = entry + risk
            target = entry - risk * cfg.target_r

        mfe_r = 0.0
        mae_r = 0.0
        result = "timeout"
        exit_price = float(candles.iloc[min(entry_idx + cfg.max_holding_bars, len(candles)-1)]["close"])
        bars_held = 0

        end = min(entry_idx + cfg.max_holding_bars + 1, len(candles))
        for j in range(entry_idx + 1, end):
            bar = candles.iloc[j]
            bars_held = j - entry_idx

            if setup.direction == "long":
                favorable = (float(bar["high"]) - entry) / risk
                adverse = (entry - float(bar["low"])) / risk
                stop_hit = float(bar["low"]) <= stop
                target_hit = float(bar["high"]) >= target
            else:
                favorable = (entry - float(bar["low"])) / risk
                adverse = (float(bar["high"]) - entry) / risk
                stop_hit = float(bar["high"]) >= stop
                target_hit = float(bar["low"]) <= target

            mfe_r = max(mfe_r, favorable)
            mae_r = max(mae_r, adverse)

            if stop_hit and target_hit:
                if cfg.intrabar_policy == "target_first":
                    result, exit_price = "target", target
                else:
                    result, exit_price = "stop", stop
                break
            if stop_hit:
                result, exit_price = "stop", stop
                break
            if target_hit:
                result, exit_price = "target", target
                break

        if result == "target":
            r_multiple = cfg.target_r
        elif result == "stop":
            r_multiple = -1.0
        else:
            r_multiple = (
                (exit_price - entry) / risk
                if setup.direction == "long"
                else (entry - exit_price) / risk
            )

        outcomes.append(
            {
                "stop_price": stop,
                "target_price": target,
                "outcome": result,
                "r_multiple": r_multiple,
                "mfe_r": mfe_r,
                "mae_r": mae_r,
                "bars_held": bars_held,
                "hit_1r": mfe_r >= 1.0,
                "hit_target": result == "target",
            }
        )

    return pd.concat([labeled.reset_index(drop=True), pd.DataFrame(outcomes)], axis=1)
