# LINQ Supply/Demand V13

Python/OANDA-candle research engine replacing the Pine prototype.

## Install inside your existing research-engine virtual environment

```bash
cd ~/Desktop/linq_forex_bot/linq_forex_v10_research_engine
unzip ~/Downloads/linq_supply_demand_v13.zip -d v13_build
cp -R v13_build/linq_supply_demand_v13/src/supply_demand_v13 src/
cp v13_build/linq_supply_demand_v13/run_v13_supply_demand.py .
cp v13_build/linq_supply_demand_v13/tests/test_engine.py tests/test_v13_supply_demand.py
python3 -m pip install -e .
python3 -m py_compile run_v13_supply_demand.py src/supply_demand_v13/*.py
python3 -m pytest -q tests/test_v13_supply_demand.py
```

## Run on your existing EUR/USD OANDA M5 history

```bash
python3 run_v13_supply_demand.py \
  --instrument EUR_USD \
  --candles "/Users/rgoode/Desktop/linq_forex_bot/linq_forex_bot_v5/linq_forex_bot_v7/data/cache/EUR_USD_M5.csv" \
  --report-dir reports/v13_supply_demand \
  --spread-pips 0.8 \
  --slippage-pips 0.2 \
  --minimum-score 8 \
  --minimum-rr 1.5 \
  --max-prior-touches 3
```

## Reports

- features
- zones
- zone events: creation, merge/reinforcement, touches, invalidation, expiry
- accepted setups
- executed trades
- grouped results by freshness/touches, depth rank, score, reinforcement, direction
- summary JSON with execution costs and causality declarations

## Important

This first build uses completed M5 OHLC bars. It models spread/slippage and starts outcome evaluation on the following candle. When both stop and target occur inside one M5 candle, it defaults to `stop_first`. True broker-grade reconstruction requires lower-timeframe bid/ask or tick data.
