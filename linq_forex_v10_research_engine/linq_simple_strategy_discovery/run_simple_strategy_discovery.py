
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting to PeriodArray/Index representation will drop timezone information."
)


@dataclass(frozen=True)
class Config:
    instrument: str = "EUR_USD"
    pip_size: float = 0.0001
    spread_pips: float = 0.8
    slippage_pips: float = 0.2
    commission_r: float = 0.0
    risk_per_trade_pct: float = 0.5
    max_holding_bars: int = 96
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    min_train_trades: int = 60
    min_validation_trades: int = 20
    min_test_trades: int = 20
    minimum_locked_profit_factor: float = 1.10
    top_n: int = 25
    intrabar_policy: str = "stop_first"


@dataclass(frozen=True)
class StrategySpec:
    signal: str
    direction: str
    trend_filter: str
    session: str
    volatility_filter: str
    stop_atr: float
    target_r: float
    max_holding_bars: int

    @property
    def complexity(self) -> int:
        return (
            1
            + int(self.direction != "both")
            + int(self.trend_filter != "none")
            + int(self.session != "all")
            + int(self.volatility_filter != "none")
        )

    @property
    def name(self) -> str:
        return (
            f"{self.signal}|{self.direction}|trend={self.trend_filter}|"
            f"session={self.session}|vol={self.volatility_filter}|"
            f"stop={self.stop_atr:.2f}ATR|target={self.target_r:.2f}R|"
            f"hold={self.max_holding_bars}"
        )


def _find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_candles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = _find_column(df.columns, ["timestamp", "time", "datetime", "date"])
    if ts is None:
        raise ValueError("CSV needs a timestamp/time/datetime/date column.")

    mapping = {}
    # Supports conventional OHLC names and OANDA midpoint candle exports.
    for canonical, aliases in {
        "open": ["open", "o", "mid_open", "mid_o", "bid_open", "ask_open"],
        "high": ["high", "h", "mid_high", "mid_h", "bid_high", "ask_high"],
        "low": ["low", "l", "mid_low", "mid_l", "bid_low", "ask_low"],
        "close": ["close", "c", "mid_close", "mid_c", "bid_close", "ask_close"],
        "volume": ["volume", "tick_volume", "vol"],
    }.items():
        col = _find_column(df.columns, aliases)
        if col is not None:
            mapping[col] = canonical

    required = {"open", "high", "low", "close"}
    found = set(mapping.values())
    missing = required - found
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    df = df.rename(columns=mapping)
    df["timestamp"] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    if len(df) < 1000:
        raise ValueError("At least 1,000 candles are recommended.")
    return df


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def supertrend_direction(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(1, index=df.index, dtype=int)

    for i in range(1, len(df)):
        if pd.isna(a.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            continue
        if upper.iloc[i] < final_upper.iloc[i - 1] or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower.iloc[i] > final_lower.iloc[i - 1] or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == -1 and df["close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = 1
        elif direction.iloc[i - 1] == 1 and df["close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return direction


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, 14)
    out["atr_pct"] = out["atr"] / out["close"]
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)

    macd = ema(out["close"], 12) - ema(out["close"], 26)
    out["macd"] = macd
    out["macd_signal"] = ema(macd, 9)

    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std(ddof=0)
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std

    out["donchian_high"] = out["high"].rolling(20).max().shift(1)
    out["donchian_low"] = out["low"].rolling(20).min().shift(1)
    out["supertrend_dir"] = supertrend_direction(out)

    hour = out["timestamp"].dt.hour
    out["session"] = np.select(
        [
            hour.between(0, 6),
            hour.between(7, 10),
            hour.between(11, 15),
            hour.between(16, 20),
        ],
        ["asia", "london_open", "new_york", "late_us"],
        default="rollover",
    )

    median_atr = out["atr_pct"].rolling(500, min_periods=100).median()
    out["high_vol"] = out["atr_pct"] >= median_atr
    out["low_vol"] = out["atr_pct"] < median_atr
    return out


def signal_series(df: pd.DataFrame, signal: str) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    if signal == "ema_cross_20_50":
        long_sig = (df["ema20"] > df["ema50"]) & (df["ema20"].shift(1) <= df["ema50"].shift(1))
        short_sig = (df["ema20"] < df["ema50"]) & (df["ema20"].shift(1) >= df["ema50"].shift(1))
    elif signal == "macd_cross":
        long_sig = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        short_sig = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    elif signal == "rsi_reversal":
        long_sig = (df["rsi14"] > 30) & (df["rsi14"].shift(1) <= 30)
        short_sig = (df["rsi14"] < 70) & (df["rsi14"].shift(1) >= 70)
    elif signal == "bollinger_reentry":
        long_sig = (close > df["bb_lower"]) & (close.shift(1) <= df["bb_lower"].shift(1))
        short_sig = (close < df["bb_upper"]) & (close.shift(1) >= df["bb_upper"].shift(1))
    elif signal == "donchian_breakout":
        long_sig = close > df["donchian_high"]
        short_sig = close < df["donchian_low"]
    elif signal == "supertrend_flip":
        long_sig = (df["supertrend_dir"] == 1) & (df["supertrend_dir"].shift(1) == -1)
        short_sig = (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) == 1)
    else:
        raise ValueError(f"Unknown signal: {signal}")
    return long_sig.fillna(False), short_sig.fillna(False)


def apply_filters(
    df: pd.DataFrame,
    long_sig: pd.Series,
    short_sig: pd.Series,
    spec: StrategySpec,
) -> tuple[pd.Series, pd.Series]:
    if spec.direction == "long":
        short_sig[:] = False
    elif spec.direction == "short":
        long_sig[:] = False

    if spec.trend_filter == "ema200":
        long_sig &= df["close"] > df["ema200"]
        short_sig &= df["close"] < df["ema200"]
    elif spec.trend_filter == "ema50_200":
        long_sig &= df["ema50"] > df["ema200"]
        short_sig &= df["ema50"] < df["ema200"]

    if spec.session != "all":
        mask = df["session"] == spec.session
        long_sig &= mask
        short_sig &= mask

    if spec.volatility_filter == "high":
        long_sig &= df["high_vol"]
        short_sig &= df["high_vol"]
    elif spec.volatility_filter == "low":
        long_sig &= df["low_vol"]
        short_sig &= df["low_vol"]

    return long_sig, short_sig


def simulate(
    df: pd.DataFrame,
    spec: StrategySpec,
    cfg: Config,
) -> pd.DataFrame:
    long_sig, short_sig = signal_series(df, spec.signal)
    long_sig, short_sig = apply_filters(df, long_sig.copy(), short_sig.copy(), spec)

    signal_indices = np.flatnonzero((long_sig | short_sig).to_numpy())
    trades = []
    next_allowed = 0
    # Mid-candle data require a half-spread plus slippage on each side.
    # Entry and exit are therefore priced at executable ask/bid levels.
    half_cost = (cfg.spread_pips / 2 + cfg.slippage_pips) * cfg.pip_size

    for i in signal_indices:
        if i < next_allowed or i + 1 >= len(df):
            continue
        direction = 1 if bool(long_sig.iloc[i]) else -1
        entry_i = i + 1
        raw_entry = float(df["open"].iloc[entry_i])
        entry = raw_entry + direction * half_cost
        a = float(df["atr"].iloc[i])
        if not math.isfinite(a) or a <= 0:
            continue

        stop_distance = spec.stop_atr * a
        stop = entry - direction * stop_distance
        target = entry + direction * stop_distance * spec.target_r

        exit_i = min(entry_i + spec.max_holding_bars, len(df) - 1)
        outcome_r = None
        reason = "time"

        for j in range(entry_i, exit_i + 1):
            high = float(df["high"].iloc[j])
            low = float(df["low"].iloc[j])
            if direction == 1:
                # Long positions exit by selling at bid.
                hit_stop = low - half_cost <= stop
                hit_target = high - half_cost >= target
            else:
                # Short positions exit by buying at ask.
                hit_stop = high + half_cost >= stop
                hit_target = low + half_cost <= target

            if hit_stop and hit_target:
                if cfg.intrabar_policy == "target_first":
                    outcome_r, reason = spec.target_r - cfg.commission_r, "target"
                else:
                    outcome_r, reason = -1.0 - cfg.commission_r, "stop"
                exit_i = j
                break
            if hit_stop:
                outcome_r, reason, exit_i = -1.0 - cfg.commission_r, "stop", j
                break
            if hit_target:
                outcome_r, reason, exit_i = spec.target_r - cfg.commission_r, "target", j
                break

        if outcome_r is None:
            exit_price = float(df["close"].iloc[exit_i]) - direction * half_cost
            outcome_r = direction * (exit_price - entry) / stop_distance
            outcome_r -= cfg.commission_r

        trades.append(
            {
                "strategy": spec.name,
                "signal_time": df["timestamp"].iloc[i],
                "entry_time": df["timestamp"].iloc[entry_i],
                "exit_time": df["timestamp"].iloc[exit_i],
                "direction": "long" if direction == 1 else "short",
                "entry": entry,
                "stop": stop,
                "target": target,
                "r": float(outcome_r),
                "exit_reason": reason,
            }
        )
        next_allowed = exit_i + 1

    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": np.nan, "expectancy_r": np.nan,
            "profit_factor": np.nan, "max_drawdown_r": np.nan,
            "total_r": 0.0, "monthly_consistency": np.nan,
        }

    r = trades["r"].astype(float)
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    equity = r.cumsum()
    dd = equity - equity.cummax()
    monthly = trades.assign(month=trades["exit_time"].dt.to_period("M")).groupby("month")["r"].sum()

    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "total_r": float(r.sum()),
        "monthly_consistency": float((monthly > 0).mean()) if len(monthly) else np.nan,
    }


def split_ranges(df: pd.DataFrame, cfg: Config) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    n = len(df)
    train_end = int(n * cfg.train_fraction)
    valid_end = int(n * (cfg.train_fraction + cfg.validation_fraction))
    return {
        "train": (df["timestamp"].iloc[0], df["timestamp"].iloc[train_end - 1]),
        "validation": (df["timestamp"].iloc[train_end], df["timestamp"].iloc[valid_end - 1]),
        "test": (df["timestamp"].iloc[valid_end], df["timestamp"].iloc[-1]),
    }


def slice_trades(trades: pd.DataFrame, bounds: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    if trades.empty:
        return trades
    start, end = bounds
    return trades[(trades["signal_time"] >= start) & (trades["signal_time"] <= end)].copy()


def discovery_score(row: dict, complexity: int) -> float:
    """Rank only with discovery and validation data.

    The locked test is intentionally excluded so it remains untouched until
    finalists have already been selected.
    """
    train_exp = row.get("train_expectancy_r", np.nan)
    valid_exp = row.get("validation_expectancy_r", np.nan)
    valid_pf = row.get("validation_profit_factor", 0.0)
    valid_consistency = row.get("validation_monthly_consistency", 0.0)
    valid_dd = abs(min(row.get("validation_max_drawdown_r", 0.0), 0.0))
    stability_gap = abs(train_exp - valid_exp)
    return (
        30 * min(train_exp, valid_exp)
        + 12 * min(valid_pf, 3.0)
        + 12 * valid_consistency
        - 0.8 * valid_dd
        - 10 * stability_gap
        - 2.5 * complexity
    )


def locked_test_pass(record: dict, cfg: Config) -> bool:
    return bool(
        record.get("test_trades", 0) >= cfg.min_test_trades
        and record.get("test_expectancy_r", -np.inf) > 0
        and record.get("test_profit_factor", 0.0) >= cfg.minimum_locked_profit_factor
    )

def generate_specs(cfg: Config) -> list[StrategySpec]:
    signals = [
        "ema_cross_20_50",
        "macd_cross",
        "rsi_reversal",
        "bollinger_reentry",
        "donchian_breakout",
        "supertrend_flip",
    ]
    directions = ["both", "long", "short"]
    trends = ["none", "ema200", "ema50_200"]
    sessions = ["all", "asia", "london_open", "new_york"]
    vols = ["none", "high"]
    stops = [1.0, 1.5, 2.0]
    targets = [1.0, 1.5, 2.0, 3.0]

    return [
        StrategySpec(s, d, t, se, v, sl, tp, cfg.max_holding_bars)
        for s, d, t, se, v, sl, tp in itertools.product(
            signals, directions, trends, sessions, vols, stops, targets
        )
    ]


def plain_english(spec: StrategySpec) -> str:
    direction = {
        "both": "Take both buy and sell signals",
        "long": "Take buy signals only",
        "short": "Take sell signals only",
    }[spec.direction]
    trend = {
        "none": "without an additional trend filter",
        "ema200": "only in the direction of price versus the 200 EMA",
        "ema50_200": "only when the 50 EMA and 200 EMA trend agree",
    }[spec.trend_filter]
    session = "at any time" if spec.session == "all" else f"during the {spec.session.replace('_', ' ')} session"
    vol = "" if spec.volatility_filter == "none" else " and only when volatility is above its rolling median"
    return (
        f"{direction} from {spec.signal.replace('_', ' ')} {trend}, {session}{vol}. "
        f"Enter at the next candle open, use a {spec.stop_atr:.2f}-ATR stop, "
        f"target {spec.target_r:.2f}R, and close after {spec.max_holding_bars} bars if neither is hit."
    )


def run(csv_path: str | Path, report_dir: str | Path, cfg: Config) -> dict:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(load_candles(csv_path))
    ranges = split_ranges(df, cfg)
    rows = []

    specs = generate_specs(cfg)
    print(f"Testing {len(specs):,} simple strategies on {len(df):,} candles...")
    print("Locked-test data will not be used to rank or select finalists.")

    # Stage 1: discovery + validation only.
    for count, spec in enumerate(specs, 1):
        trades = simulate(df, spec, cfg)
        record = {"strategy": spec.name, "complexity": spec.complexity, **asdict(spec)}

        for split in ("train", "validation"):
            m = metrics(slice_trades(trades, ranges[split]))
            for k, v in m.items():
                record[f"{split}_{k}"] = v

        eligible = bool(
            record["train_trades"] >= cfg.min_train_trades
            and record["validation_trades"] >= cfg.min_validation_trades
            and record["train_expectancy_r"] > 0
            and record["validation_expectancy_r"] > 0
        )
        record["discovery_eligible"] = eligible
        record["discovery_score"] = discovery_score(record, spec.complexity) if eligible else -9999.0
        rows.append(record)

        if count % 500 == 0:
            print(f"  completed {count:,}/{len(specs):,}")

    ranking = pd.DataFrame(rows).sort_values(
        ["discovery_eligible", "discovery_score", "validation_expectancy_r"],
        ascending=[False, False, False],
    )
    ranking.to_csv(report_dir / f"{cfg.instrument}_discovery_ranking.csv", index=False)

    # Stage 2: freeze the shortlist, then open the locked test exactly once.
    finalists = ranking[ranking["discovery_eligible"]].head(cfg.top_n).copy()
    recommendations = []
    locked_rows = []

    for discovery_rank, (_, row) in enumerate(finalists.iterrows(), 1):
        spec = StrategySpec(
            signal=row["signal"],
            direction=row["direction"],
            trend_filter=row["trend_filter"],
            session=row["session"],
            volatility_filter=row["volatility_filter"],
            stop_atr=float(row["stop_atr"]),
            target_r=float(row["target_r"]),
            max_holding_bars=int(row["max_holding_bars"]),
        )
        trades = simulate(df, spec, cfg)
        test_m = metrics(slice_trades(trades, ranges["test"]))
        test_record = row.to_dict()
        for k, v in test_m.items():
            test_record[f"test_{k}"] = v
        test_record["discovery_rank"] = discovery_rank
        test_record["locked_pass"] = locked_test_pass(test_record, cfg)
        locked_rows.append(test_record)

        safe_name = str(discovery_rank).zfill(2)
        trades.to_csv(report_dir / f"finalist_{safe_name}_trades.csv", index=False)
        recommendations.append(
            {
                "discovery_rank": discovery_rank,
                "strategy": spec.name,
                "plain_english": plain_english(spec),
                "complexity": spec.complexity,
                "discovery_score": float(row["discovery_score"]),
                "locked_pass": bool(test_record["locked_pass"]),
                "train": {
                    "trades": int(row["train_trades"]),
                    "expectancy_r": float(row["train_expectancy_r"]),
                    "profit_factor": float(row["train_profit_factor"]),
                    "max_drawdown_r": float(row["train_max_drawdown_r"]),
                    "monthly_consistency": float(row["train_monthly_consistency"]),
                },
                "validation": {
                    "trades": int(row["validation_trades"]),
                    "expectancy_r": float(row["validation_expectancy_r"]),
                    "profit_factor": float(row["validation_profit_factor"]),
                    "max_drawdown_r": float(row["validation_max_drawdown_r"]),
                    "monthly_consistency": float(row["validation_monthly_consistency"]),
                },
                "locked_test": {
                    "trades": int(test_m["trades"]),
                    "expectancy_r": float(test_m["expectancy_r"]),
                    "profit_factor": float(test_m["profit_factor"]),
                    "max_drawdown_r": float(test_m["max_drawdown_r"]),
                    "monthly_consistency": float(test_m["monthly_consistency"]),
                },
            }
        )

    locked_df = pd.DataFrame(locked_rows)
    if not locked_df.empty:
        locked_df.to_csv(report_dir / f"{cfg.instrument}_finalist_locked_test.csv", index=False)

    passing = [r for r in recommendations if r["locked_pass"]]
    summary = {
        "engine": "LINQ Simple Strategy Discovery 1.1",
        "instrument": cfg.instrument,
        "candles": len(df),
        "strategies_tested": len(specs),
        "discovery_eligible_strategies": int(ranking["discovery_eligible"].sum()),
        "finalists_opened_on_locked_test": len(recommendations),
        "locked_test_passes": len(passing),
        "split_ranges": {
            k: {"start": str(v[0]), "end": str(v[1])} for k, v in ranges.items()
        },
        "cost_assumptions": {
            "spread_pips": cfg.spread_pips,
            "slippage_pips_each_side": cfg.slippage_pips,
            "commission_r": cfg.commission_r,
            "pricing": "mid OHLC converted to executable bid/ask using half spread plus slippage per side",
        },
        "qualification_gates": {
            "min_train_trades": cfg.min_train_trades,
            "min_validation_trades": cfg.min_validation_trades,
            "min_test_trades": cfg.min_test_trades,
            "minimum_locked_profit_factor": cfg.minimum_locked_profit_factor,
            "positive_expectancy_required_in_each_stage": True,
        },
        "recommendations_in_frozen_discovery_order": recommendations,
        "warning": (
            "Historical profitability does not guarantee future profitability. "
            "Locked-test results were excluded from candidate ranking and selection. "
            "Passing finalists still require parameter-neighbor, walk-forward, and demo validation."
        ),
    }
    (report_dir / f"{cfg.instrument}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print("\n" + "=" * 100)
    print("LINQ SIMPLE STRATEGY DISCOVERY 1.1 — RESULTS")
    print("=" * 100)
    print(f"Strategies tested: {len(specs):,}")
    print(f"Discovery/validation eligible: {int(ranking['discovery_eligible'].sum()):,}")
    print(f"Frozen finalists evaluated on locked test: {len(recommendations):,}")
    print(f"Locked-test passes: {len(passing):,}")

    if recommendations:
        best = recommendations[0]
        print("\nTOP DISCOVERY-RANKED FINALIST")
        print("-" * 100)
        print(best["plain_english"])
        print(
            f"Locked test: pass={best['locked_pass']}, trades={best['locked_test']['trades']}, "
            f"expectancy={best['locked_test']['expectancy_r']:.3f}R, "
            f"PF={best['locked_test']['profit_factor']:.3f}, "
            f"DD={best['locked_test']['max_drawdown_r']:.2f}R, "
            f"positive months={best['locked_test']['monthly_consistency']:.1%}"
        )
    else:
        print("\nNo strategy passed the discovery and validation gates.")

    if recommendations and not passing:
        print("\nNo frozen finalist passed the locked-test gates. Do not force a winner.")

    print(f"\nReports saved to: {report_dir}")
    return summary

def main() -> None:
    p = argparse.ArgumentParser(
        description="Find simple, historically robust trading rules from OHLC data."
    )
    p.add_argument("--candles", required=True, help="CSV containing timestamp and OHLC columns.")
    p.add_argument("--instrument", default="EUR_USD")
    p.add_argument("--report-dir", default="reports/simple_discovery")
    p.add_argument("--pip-size", type=float, default=0.0001)
    p.add_argument("--spread-pips", type=float, default=0.8)
    p.add_argument("--slippage-pips", type=float, default=0.2)
    p.add_argument("--max-holding-bars", type=int, default=96)
    p.add_argument("--min-train-trades", type=int, default=60)
    p.add_argument("--min-validation-trades", type=int, default=20)
    p.add_argument("--min-test-trades", type=int, default=20)
    p.add_argument("--minimum-locked-profit-factor", type=float, default=1.10)
    p.add_argument("--top-n", type=int, default=25)
    args = p.parse_args()

    cfg = Config(
        instrument=args.instrument,
        pip_size=args.pip_size,
        spread_pips=args.spread_pips,
        slippage_pips=args.slippage_pips,
        max_holding_bars=args.max_holding_bars,
        min_train_trades=args.min_train_trades,
        min_validation_trades=args.min_validation_trades,
        min_test_trades=args.min_test_trades,
        minimum_locked_profit_factor=args.minimum_locked_profit_factor,
        top_n=args.top_n,
    )
    run(args.candles, args.report_dir, cfg)


if __name__ == "__main__":
    main()
