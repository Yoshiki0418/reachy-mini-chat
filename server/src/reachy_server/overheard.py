"""場の文脈バッファ: 自分宛てではない（ignore / backchannel）発話を、短時間だけメモリに持つ。

研究室の一員として「いま何の話をしているか」を理解する助けにする。保存はしない。設計書 §2.2。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class OverheardBuffer:
    def __init__(
        self, ttl_s: float = 600.0, max_items: int = 20, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.ttl_s = ttl_s
        self._items: deque[tuple[float, str]] = deque(maxlen=max_items)
        self._clock = clock

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0

    def add(self, text: str) -> None:
        text = text.strip()
        if self.enabled and text:
            self._items.append((self._clock(), text))

    def recent(self) -> tuple[str, ...]:
        """期限内のものを古い順に返す（期限切れはここで捨てる）。"""
        now = self._clock()
        while self._items and now - self._items[0][0] > self.ttl_s:
            self._items.popleft()
        return tuple(text for _, text in self._items)

    def clear(self) -> None:
        self._items.clear()
