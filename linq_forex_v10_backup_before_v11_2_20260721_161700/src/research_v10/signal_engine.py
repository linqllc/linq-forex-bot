from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ResearchConfig


@dataclass
class ActiveImpulse:
    impulse_index: int
    impulse_timestamp: pd.Timestamp
    direction: str
    impulse_low: float
    impulse_high: float
    impulse_size: float
    impulse_atr: float
    impulse_body_ratio: float
    bull_fvg_atr: float
    bear_fvg_atr: float
    expires_after_index: int


@dataclass
class OpenTrade:
    setup_id: str
    instrument: str
    direction: str
    impulse_index: int
    entry_index: int
    impulse_timestamp: pd.Timestamp
    timestamp: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    risk_price: float
    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


class CausalSignalEngine:
    """
    Bar-by-bar signal detector.

    This engine never searches beyond the current candle. Each call to
    process_bar() sees only the current completed candle and state created
    by earlier candles.
    """

    def __init__(
        self,
        config: ResearchConfig,
        retrace_threshold: float = 0.25,
        required_session: str = "london_open",
    ):
        self.config = config
        self.retrace_threshold = float(retrace_threshold)
        self.required_session = required_session
        self.active_impulses: list[ActiveImpulse] = []

    def _register_impulse(
        self,
        index: int,
        candle: pd.Series,
    ) -> None:
        if not np.isfinite(candle.get("atr", np.nan)):
            return

        atr = float(candle["atr"])
        if atr <= 0:
            return

        candle_open = float(candle["open"])
        candle_close = float(candle["close"])
        candle_low = float(candle["low"])
        candle_high = float(candle["high"])

        bullish = candle_close > candle_open
        bearish = candle_close < candle_open

        displaced = (
            float(candle["range_atr"])
            >= self.config.displacement_atr_min
            and float(candle["body_ratio"])
            >= self.config.displacement_body_ratio_min
        )

        if not displaced or not (bullish or bearish):
            return

        impulse_size = candle_high - candle_low
        if impulse_size <= 0:
            return

        direction = "long" if bullish else "short"

        self.active_impulses.append(
            ActiveImpulse(
                impulse_index=index,
                impulse_timestamp=pd.Timestamp(candle["timestamp"]),
                direction=direction,
                impulse_low=candle_low,
                impulse_high=candle_high,
                impulse_size=impulse_size,
                impulse_atr=impulse_size / atr,
                impulse_body_ratio=float(candle["body_ratio"]),
                bull_fvg_atr=float(candle.get("bull_fvg_atr", np.nan)),
                bear_fvg_atr=float(candle.get("bear_fvg_atr", np.nan)),
                expires_after_index=index + self.config.max_pullback_bars,
            )
        )

    def _signal_from_impulse(
        self,
        impulse: ActiveImpulse,
        index: int,
        candle: pd.Series,
    ) -> dict[str, Any] | None:
        low = float(candle["low"])
        high = float(candle["high"])
        close = float(candle["close"])

        if impulse.direction == "long":
            if low < impulse.impulse_low:
                return {
                    "_remove": True,
                    "_reason": "impulse_invalidated",
                }

            shallow = (
                impulse.impulse_high
                - self.config.pullback_min * impulse.impulse_size
            )
            deep = (
                impulse.impulse_high
                - self.config.pullback_max * impulse.impulse_size
            )

            touched = low <= shallow and high >= deep
            if not touched:
                return None

            entry_price = min(shallow, max(deep, close))
            retrace_fraction = (
                impulse.impulse_high - entry_price
            ) / impulse.impulse_size

        else:
            if high > impulse.impulse_high:
                return {
                    "_remove": True,
                    "_reason": "impulse_invalidated",
                }

            shallow = (
                impulse.impulse_low
                + self.config.pullback_min * impulse.impulse_size
            )
            deep = (
                impulse.impulse_low
                + self.config.pullback_max * impulse.impulse_size
            )

            touched = high >= shallow and low <= deep
            if not touched:
                return None

            entry_price = max(shallow, min(deep, close))
            retrace_fraction = (
                entry_price - impulse.impulse_low
            ) / impulse.impulse_size

        session = str(candle["session"])

        signal_passed = (
            retrace_fraction <= self.retrace_threshold + 1e-12
            and session == self.required_session
        )

        timestamp = pd.Timestamp(candle["timestamp"])

        signal = {
            "setup_id": (
                f"{self.config.instrument}-"
                f"{impulse.direction}-"
                f"{timestamp.isoformat()}"
            ),
            "instrument": self.config.instrument,
            "direction": impulse.direction,
            "impulse_index": impulse.impulse_index,
            "entry_index": index,
            "impulse_timestamp": impulse.impulse_timestamp,
            "timestamp": timestamp,
            "entry_price": float(entry_price),
            "impulse_atr": impulse.impulse_atr,
            "impulse_body_ratio": impulse.impulse_body_ratio,
            "retrace_fraction": float(retrace_fraction),
            "pullback_bars": index - impulse.impulse_index,
            "atr": float(candle["atr"]),
            "ema_fast": float(candle["ema_fast"]),
            "ema_slow": float(candle["ema_slow"]),
            "ema_slow_slope_12": float(
                candle["ema_slow_slope_12"]
            ),
            "ema_distance_atr": float(candle["ema_distance_atr"]),
            "return_12": float(candle["return_12"]),
            "return_48": float(candle["return_48"]),
            "realized_vol_48": float(candle["realized_vol_48"]),
            "hour_utc": float(candle["hour_utc"]),
            "weekday": int(candle["weekday"]),
            "session": session,
            "distance_prev_high_atr": float(
                candle["distance_prev_high_atr"]
            ),
            "distance_prev_low_atr": float(
                candle["distance_prev_low_atr"]
            ),
            "bull_fvg_atr": impulse.bull_fvg_atr,
            "bear_fvg_atr": impulse.bear_fvg_atr,
            "trend_aligned": bool(
                entry_price > float(candle["ema_slow"])
                and float(candle["ema_slow_slope_12"]) > 0
                if impulse.direction == "long"
                else entry_price < float(candle["ema_slow"])
                and float(candle["ema_slow_slope_12"]) < 0
            ),
            "signal_passed": bool(signal_passed),
            "accepted": bool(signal_passed),
            "rejection_reason": (
                ""
                if signal_passed
                else "locked_rule_failed"
            ),
            "locked_rule": (
                f"retrace_fraction le {self.retrace_threshold} "
                f"AND session eq {self.required_session}"
            ),
            "_remove": True,
            "_reason": "pullback_touched",
        }

        return signal

    def process_bar(
        self,
        index: int,
        candle: pd.Series,
    ) -> list[dict[str, Any]]:
        """
        Process exactly one completed candle.

        Existing impulses are evaluated first. The current candle can then
        become a new impulse for future candles, but never for itself.
        """
        signals: list[dict[str, Any]] = []
        surviving: list[ActiveImpulse] = []

        for impulse in self.active_impulses:
            if index > impulse.expires_after_index:
                continue

            result = self._signal_from_impulse(
                impulse=impulse,
                index=index,
                candle=candle,
            )

            if result is None:
                surviving.append(impulse)
                continue

            remove = bool(result.pop("_remove", False))
            result.pop("_reason", None)

            if "setup_id" in result:
                signals.append(result)

            if not remove:
                surviving.append(impulse)

        self.active_impulses = surviving

        self._register_impulse(
            index=index,
            candle=candle,
        )

        return signals
