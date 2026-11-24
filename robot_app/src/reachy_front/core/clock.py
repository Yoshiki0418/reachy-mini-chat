"""時計。判断ロジックは時刻を直接読まず、これを受け取る（テストで時間を進められるように）。"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """単調増加の秒。起点は不定（差だけに意味がある）。"""
        ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()


class ManualClock:
    """テスト用。``advance`` で進めるまで止まっている。"""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("時計は戻せない")
        self._t += seconds


__all__ = ["Clock", "ManualClock", "SystemClock"]
