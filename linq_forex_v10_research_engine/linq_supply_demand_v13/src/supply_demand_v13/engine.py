from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from .config import Config


@dataclass
class Zone:
    zone_id: str
    direction: int
    created_index: int
    created_time: str
    top: float
    bottom: float
    base_score: int
    reinforcements: int = 0
    touches: int = 0
    last_touch_index: int = -10**9
    active: bool = True
    invalidated_index: int | None = None
    expired_index: int | None = None


def _overlap_fraction(a_top, a_bottom, b_top, b_bottom) -> float:
    overlap = max(0.0, min(a_top, b_top) - max(a_bottom, b_bottom))
    smaller = max(min(a_top-a_bottom, b_top-b_bottom), 1e-12)
    return overlap / smaller


def _gap(a_top, a_bottom, b_top, b_bottom) -> float:
    if a_bottom > b_top:
        return a_bottom - b_top
    if b_bottom > a_top:
        return b_bottom - a_top
    return 0.0


def _impulse_quality(df: pd.DataFrame, i: int, direction: int, cfg: Config) -> dict:
    start = i - cfg.impulse_bars + 1
    if start < 1:
        return {"valid": False}

    w = df.iloc[start:i+1]
    atr = float(df.iloc[i]["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return {"valid": False}

    displacement = (
        float(df.iloc[i]["close"] - df.iloc[start-1]["close"])
        if direction == 1 else
        float(df.iloc[start-1]["close"] - df.iloc[i]["close"])
    )
    directional = (
        (w["close"] > w["open"]).mean()
        if direction == 1 else
        (w["close"] < w["open"]).mean()
    )
    body_ratio = float(w["body_range"].mean())

    overlaps = []
    for j in range(start + 1, i + 1):
        a, b = df.iloc[j-1], df.iloc[j]
        ov = max(0.0, min(a.high, b.high) - max(a.low, b.low))
        smaller = max(min(a.high-a.low, b.high-b.low), 1e-12)
        overlaps.append(ov / smaller)
    overlap = float(np.mean(overlaps)) if overlaps else 0.0

    if direction == 1:
        close_location = float(((w["close"] - w["low"]) / w["range"]).mean())
        prior = float(df.iloc[max(0, start-cfg.bos_lookback):start]["high"].max())
        bos = float(df.iloc[i]["close"]) > prior
    else:
        close_location = float(((w["high"] - w["close"]) / w["range"]).mean())
        prior = float(df.iloc[max(0, start-cfg.bos_lookback):start]["low"].min())
        bos = float(df.iloc[i]["close"]) < prior

    clean = (
        directional >= cfg.min_directional_share and
        body_ratio >= cfg.min_body_range_ratio and
        overlap <= cfg.max_overlap_ratio and
        close_location >= cfg.min_close_location
    )
    return {
        "valid": displacement >= cfg.impulse_atr_min * atr and bos and clean,
        "displacement_atr": displacement / atr,
        "directional_share": directional,
        "body_range": body_ratio,
        "overlap": overlap,
        "close_location": close_location,
        "bos": bos,
        "clean": clean,
        "start": start,
    }


def _base(df: pd.DataFrame, impulse_start: int, direction: int, cfg: Config):
    lo = max(0, impulse_start - cfg.base_search_bars)
    for j in range(impulse_start - 1, lo - 1, -1):
        row = df.iloc[j]
        opposite = row.close < row.open if direction == 1 else row.close > row.open
        if opposite:
            top = max(row.open, row.close) if direction == 1 else row.high
            bottom = row.low if direction == 1 else min(row.open, row.close)
            return j, float(top), float(bottom)
    return None


def _slowing(df: pd.DataFrame, i: int, direction: int, cfg: Config) -> tuple[bool, dict]:
    r0 = i - cfg.slowdown_recent
    e0 = r0 - cfg.slowdown_earlier
    if e0 < 0:
        return False, {}

    earlier = df.iloc[e0:r0]
    recent = df.iloc[r0:i]
    opp_e = earlier[earlier["close"] < earlier["open"]] if direction == 1 else earlier[earlier["close"] > earlier["open"]]
    opp_r = recent[recent["close"] < recent["open"]] if direction == 1 else recent[recent["close"] > recent["open"]]

    needed_e = 1 if cfg.allow_mixed_candles else cfg.slowdown_earlier
    needed_r = 1 if cfg.allow_mixed_candles else cfg.slowdown_recent
    if len(opp_e) < needed_e or len(opp_r) < needed_r:
        return False, {"earlier_count": len(opp_e), "recent_count": len(opp_r)}

    eavg = float(opp_e["body"].mean())
    ravg = float(opp_r["body"].mean())
    rmax = float(opp_r["body"].max())
    ok = (
        eavg > 0 and
        ravg <= eavg * cfg.slowdown_recent_avg_max and
        rmax <= eavg * cfg.slowdown_recent_single_max
    )
    return ok, {
        "earlier_avg_body": eavg,
        "recent_avg_body": ravg,
        "recent_max_body": rmax,
        "recent_avg_ratio": ravg / eavg if eavg else np.nan,
        "recent_max_ratio": rmax / eavg if eavg else np.nan,
    }


def build_setups(df: pd.DataFrame, cfg: Config):
    zones: list[Zone] = []
    zone_events: list[dict] = []
    setups: list[dict] = []
    last_impulse = {1: -10**9, -1: -10**9}

    warmup = max(100, cfg.bos_lookback + cfg.impulse_bars + cfg.base_search_bars)

    for i in range(warmup, len(df) - cfg.max_holding_bars - 1):
        row = df.iloc[i]
        atr = float(row.atr)
        if not np.isfinite(atr) or atr <= 0:
            continue

        # Invalidate/expire/touch existing zones.
        for z in zones:
            if not z.active:
                continue
            if i - z.created_index > cfg.max_zone_age_bars:
                z.active = False
                z.expired_index = i
                zone_events.append({"zone_id": z.zone_id, "index": i, "time": row.time, "event": "expired"})
                continue
            invalid = row.close < z.bottom if z.direction == 1 else row.close > z.top
            if invalid:
                z.active = False
                z.invalidated_index = i
                zone_events.append({"zone_id": z.zone_id, "index": i, "time": row.time, "event": "invalidated"})
                continue

            width = max(z.top-z.bottom, 1e-12)
            inside = (
                row.low <= z.top-width*cfg.meaningful_penetration and row.high >= z.bottom
                if z.direction == 1 else
                row.high >= z.bottom+width*cfg.meaningful_penetration and row.low <= z.top
            )
            if inside and i - z.last_touch_index > 1:
                z.touches += 1
                z.last_touch_index = i
                zone_events.append({
                    "zone_id": z.zone_id, "index": i, "time": row.time,
                    "event": "touch", "touch_number": z.touches,
                })

        # Detect impulses and create/merge zones.
        for direction in (1, -1):
            q = _impulse_quality(df, i, direction, cfg)
            if not q.get("valid") or i - last_impulse[direction] <= 1:
                continue
            last_impulse[direction] = i
            base = _base(df, q["start"], direction, cfg)
            if base is None:
                continue
            base_i, top, bottom = base
            width = top-bottom
            if not (cfg.min_zone_atr*atr <= width <= cfg.max_zone_atr*atr):
                continue

            merge = None
            for z in zones:
                if z.active and z.direction == direction:
                    if (_overlap_fraction(top, bottom, z.top, z.bottom) >= cfg.merge_overlap_fraction or
                        _gap(top, bottom, z.top, z.bottom) <= cfg.merge_gap_atr*atr):
                        merge = z
                        break

            base_score = cfg.score_impulse + cfg.score_bos + cfg.score_clean
            if merge:
                merge.top = max(merge.top, top)
                merge.bottom = min(merge.bottom, bottom)
                merge.base_score = max(merge.base_score, base_score)
                merge.reinforcements += 1
                zone_events.append({
                    "zone_id": merge.zone_id, "index": i, "time": row.time,
                    "event": "reinforced", "reinforcements": merge.reinforcements,
                })
            else:
                zid = f"{cfg.instrument}-{('demand' if direction == 1 else 'supply')}-{pd.Timestamp(row.time).isoformat()}"
                z = Zone(
                    zone_id=zid, direction=direction, created_index=i,
                    created_time=pd.Timestamp(row.time).isoformat(),
                    top=top, bottom=bottom, base_score=base_score,
                )
                zones.append(z)
                zone_events.append({
                    "zone_id": zid, "index": i, "time": row.time,
                    "event": "created", **q,
                })

        # Confirm setups against recently touched zones.
        for z in zones:
            if not z.active or z.direction != int(row.h1_direction):
                continue
            if not (0 <= i-z.last_touch_index <= cfg.confirmation_window):
                continue

            prior_touches = max(z.touches - 1, 0)
            if prior_touches > cfg.max_prior_touches:
                continue

            directional = row.close > row.open if z.direction == 1 else row.close < row.open
            outside = row.close > z.top if z.direction == 1 else row.close < z.bottom
            body_ok = row.body >= cfg.min_confirmation_body_atr * atr
            if not (directional and outside and body_ok):
                continue

            if cfg.require_micro_break:
                if z.direction == 1:
                    micro_ok = row.close > df.iloc[i-cfg.micro_break_lookback:i]["high"].max()
                else:
                    micro_ok = row.close < df.iloc[i-cfg.micro_break_lookback:i]["low"].min()
                if not micro_ok:
                    continue

            slowing_ok, slowing = _slowing(df, i, z.direction, cfg)
            if not slowing_ok:
                continue

            score = z.base_score + cfg.score_h1_alignment
            score += cfg.score_fresh if prior_touches == 0 else cfg.score_first_reuse if prior_touches == 1 else 0
            score += min(z.reinforcements, 2) * cfg.score_reinforcement
            if score < cfg.minimum_zone_score:
                continue

            # Depth rank among active aligned zones at this instant.
            peers = [x for x in zones if x.active and x.direction == z.direction]
            if z.direction == 1:
                peers.sort(key=lambda x: x.bottom)
            else:
                peers.sort(key=lambda x: x.top, reverse=True)
            rank = peers.index(z) + 1

            stop = z.bottom - cfg.stop_buffer_atr*atr if z.direction == 1 else z.top + cfg.stop_buffer_atr*atr
            target = float(row.last_high) if z.direction == 1 else float(row.last_low)
            risk = row.close-stop if z.direction == 1 else stop-row.close
            reward = target-row.close if z.direction == 1 else row.close-target
            rr = reward/risk if risk > 0 else np.nan
            if not np.isfinite(rr) or rr < cfg.minimum_rr:
                continue

            setup_id = f"{z.zone_id}-entry-{pd.Timestamp(row.time).isoformat()}"
            setups.append({
                "setup_id": setup_id,
                "zone_id": z.zone_id,
                "entry_index": i,
                "timestamp": row.time,
                "direction": "long" if z.direction == 1 else "short",
                "entry": float(row.close),
                "stop": float(stop),
                "target": float(target),
                "planned_rr": float(rr),
                "zone_top": z.top,
                "zone_bottom": z.bottom,
                "zone_score": score,
                "depth_rank": rank,
                "prior_touches": prior_touches,
                "reinforcements": z.reinforcements,
                **slowing,
            })
            # One setup per zone touch.
            z.last_touch_index = -10**9

    zones_df = pd.DataFrame([asdict(z) for z in zones])
    events_df = pd.DataFrame(zone_events)
    setups_df = pd.DataFrame(setups)
    return zones_df, events_df, setups_df
