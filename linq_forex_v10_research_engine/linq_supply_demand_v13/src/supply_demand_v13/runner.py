from __future__ import annotations
from pathlib import Path
import json
from .config import Config
from .data import load_candles
from .features import add_features
from .engine import build_setups
from .backtest import simulate, summarize, grouped_reports


def run(csv_path: str | Path, report_dir: str | Path, cfg: Config) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = load_candles(csv_path)
    featured = add_features(raw, cfg)
    zones, events, setups = build_setups(featured, cfg)
    trades = simulate(featured, setups, cfg)
    summary = {
        "engine_version": "13.0-supply-demand",
        "instrument": cfg.instrument,
        "candles": len(raw),
        "zones": len(zones),
        "zone_events": len(events),
        "setups": len(setups),
        "performance": summarize(trades),
        "causality": {
            "h1_pivots_delayed_until_confirmation": True,
            "completed_h1_context_only": True,
            "entries_use_current_completed_m5_close": True,
            "outcomes_start_next_m5_bar": True,
            "intrabar_policy": cfg.intrabar_policy,
        },
        "costs": {
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
        },
    }

    featured.to_csv(report_dir / f"{cfg.instrument}_v13_features.csv", index=False)
    zones.to_csv(report_dir / f"{cfg.instrument}_v13_zones.csv", index=False)
    events.to_csv(report_dir / f"{cfg.instrument}_v13_zone_events.csv", index=False)
    setups.to_csv(report_dir / f"{cfg.instrument}_v13_setups.csv", index=False)
    trades.to_csv(report_dir / f"{cfg.instrument}_v13_trades.csv", index=False)
    for name, report in grouped_reports(trades).items():
        report.to_csv(report_dir / f"{cfg.instrument}_v13_by_{name}.csv", index=False)
    (report_dir / f"{cfg.instrument}_v13_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
