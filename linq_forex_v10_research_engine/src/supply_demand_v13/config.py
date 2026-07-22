from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    instrument: str = "EUR_USD"
    pip_size: float = 0.0001

    atr_period: int = 14
    h1_pivot_left: int = 2
    h1_pivot_right: int = 2
    h1_ema_fast: int = 20
    h1_ema_slow: int = 50

    impulse_bars: int = 3
    impulse_atr_min: float = 1.50
    bos_lookback: int = 12
    min_directional_share: float = 2 / 3
    min_body_range_ratio: float = 0.55
    max_overlap_ratio: float = 0.55
    min_close_location: float = 0.65
    base_search_bars: int = 4

    min_zone_atr: float = 0.10
    max_zone_atr: float = 1.25
    merge_overlap_fraction: float = 0.35
    merge_gap_atr: float = 0.15
    max_zone_age_bars: int = 400
    max_prior_touches: int = 3
    meaningful_penetration: float = 0.10

    slowdown_earlier: int = 3
    slowdown_recent: int = 2
    slowdown_recent_avg_max: float = 0.35
    slowdown_recent_single_max: float = 1.30
    allow_mixed_candles: bool = True

    confirmation_window: int = 2
    min_confirmation_body_atr: float = 0.20
    require_micro_break: bool = False
    micro_break_lookback: int = 3

    stop_buffer_atr: float = 0.10
    minimum_rr: float = 1.50
    max_holding_bars: int = 96
    intrabar_policy: str = "stop_first"

    spread_pips: float = 0.8
    slippage_pips: float = 0.2

    score_impulse: int = 3
    score_bos: int = 3
    score_clean: int = 2
    score_h1_alignment: int = 2
    score_fresh: int = 2
    score_first_reuse: int = 1
    score_reinforcement: int = 1
    minimum_zone_score: int = 8
