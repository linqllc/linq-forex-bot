from __future__ import annotations
import argparse
import json
from supply_demand_v13.config import Config
from supply_demand_v13.runner import run


def main():
    p = argparse.ArgumentParser(description="LINQ V13 Supply/Demand research engine")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--candles", required=True)
    p.add_argument("--report-dir", default="reports/v13_supply_demand")
    p.add_argument("--spread-pips", type=float, default=0.8)
    p.add_argument("--slippage-pips", type=float, default=0.2)
    p.add_argument("--minimum-score", type=int, default=8)
    p.add_argument("--minimum-rr", type=float, default=1.5)
    p.add_argument("--max-prior-touches", type=int, default=3)
    args = p.parse_args()

    cfg = Config(
        instrument=args.instrument,
        spread_pips=args.spread_pips,
        slippage_pips=args.slippage_pips,
        minimum_zone_score=args.minimum_score,
        minimum_rr=args.minimum_rr,
        max_prior_touches=args.max_prior_touches,
    )
    print(json.dumps(run(args.candles, args.report_dir, cfg), indent=2))


if __name__ == "__main__":
    main()
