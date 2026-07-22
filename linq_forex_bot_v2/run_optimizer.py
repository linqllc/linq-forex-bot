from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.config import load_config
from src.demo_data import generate_demo_data
from src.indicators import add_indicators
from src.market_structure import add_market_structure
from src.oanda_client import OandaClient
from src.optimizer import optimize_reward_to_risk, reward_grid
from src.session import add_session_columns, calculate_opening_ranges

def prepare(candles, config):
    candles=add_session_columns(candles,config['timezone'])
    candles=add_indicators(candles,config['impulse']['atr_period'],config['impulse']['lookback_bodies'])
    candles=add_market_structure(candles,config['structure']['fast_ema'],config['structure']['slow_ema'],config['structure']['swing_window'])
    return calculate_opening_ranges(candles,config['session']['opening_range_start'],config['session']['opening_range_end'])

def main():
    parser=argparse.ArgumentParser(description='LINQ pair-specific reward/risk optimizer')
    parser.add_argument('--instruments',nargs='+',default=None)
    parser.add_argument('--days',type=int,default=180)
    parser.add_argument('--min-r',type=float,default=None); parser.add_argument('--max-r',type=float,default=None); parser.add_argument('--step',type=float,default=None)
    parser.add_argument('--demo',action='store_true'); parser.add_argument('--config',default='config/strategy.yaml')
    args=parser.parse_args(); load_dotenv(); config=load_config(args.config)
    instruments=[x.upper() for x in (args.instruments or config['instruments'])]
    oc=config['optimizer']; ratios=reward_grid(args.min_r or oc['minimum_r'],args.max_r or oc['maximum_r'],args.step or oc['step_r'])
    reports=Path('reports/optimizer'); reports.mkdir(parents=True,exist_ok=True)
    client=None if args.demo else OandaClient(); selections=[]
    for instrument in instruments:
        if args.demo: candles=generate_demo_data(args.days); source='synthetic_demo'
        else:
            end=datetime.now(timezone.utc); start=end-timedelta(days=args.days)
            candles=client.get_candles(instrument,start,end,granularity=config['granularity']); source='oanda'
        results,selection=optimize_reward_to_risk(prepare(candles,config),instrument,config,ratios)
        selection.update({'data_source':source,'days_requested':args.days})
        results.to_csv(reports/f'{instrument}_{source}_ratio_results.csv',index=False)
        (reports/f'{instrument}_{source}_best_ratio.json').write_text(json.dumps(selection,indent=2))
        selections.append(selection); print(json.dumps(selection,indent=2))
    pd.DataFrame(selections).to_csv(reports/'pair_ratio_summary.csv',index=False)
    (reports/'pair_ratio_summary.json').write_text(json.dumps(selections,indent=2))
    print(f'Saved optimizer reports to {reports}')
if __name__=='__main__': main()
