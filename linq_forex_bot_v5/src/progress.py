from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import timedelta


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float('inf'):
        return "calculating..."
    return str(timedelta(seconds=max(0, int(seconds))))


@dataclass
class ProgressBar:
    label: str
    total: int
    width: int = 28
    min_interval: float = 0.15

    def __post_init__(self) -> None:
        self.total = max(1, int(self.total))
        self.start = time.perf_counter()
        self.last_draw = 0.0
        self.current = 0
        self.finished = False

    def update(self, current: int, detail: str = "", force: bool = False) -> None:
        if self.finished:
            return
        self.current = max(0, min(int(current), self.total))
        now = time.perf_counter()
        if not force and self.current < self.total and now - self.last_draw < self.min_interval:
            return
        self.last_draw = now
        elapsed = max(now - self.start, 1e-9)
        fraction = self.current / self.total
        rate = self.current / elapsed
        eta = (self.total - self.current) / rate if rate > 0 and self.current > 0 else None
        filled = int(round(self.width * fraction))
        bar = "█" * filled + "░" * (self.width - filled)
        suffix = f" | {detail}" if detail else ""
        line = (
            f"\r{self.label:<25} [{bar}] {fraction*100:6.2f}% "
            f"| elapsed {_fmt_seconds(elapsed)} | ETA {_fmt_seconds(eta)}{suffix}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()
        if self.current >= self.total:
            self.finished = True
            sys.stdout.write("\n")
            sys.stdout.flush()

    def advance(self, amount: int = 1, detail: str = "") -> None:
        self.update(self.current + amount, detail=detail)

    def close(self, detail: str = "complete") -> None:
        # Always draw the final state so the completion detail is visible even
        # when the last regular update already reached 100%.
        self.finished = False
        self.update(self.total, detail=detail, force=True)


class StageTimer:
    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        print(f"▶ {label}...")

    def done(self, detail: str = "") -> None:
        elapsed = time.perf_counter() - self.start
        extra = f" | {detail}" if detail else ""
        print(f"✓ {self.label} completed in {_fmt_seconds(elapsed)}{extra}")
