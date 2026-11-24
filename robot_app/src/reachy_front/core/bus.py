"""asyncio の pub/sub イベントバス（設計書 §3.1）。

- 購読者ごとに専用のキューとタスクを持つ。**1 人の購読者の例外・遅さは、発行者にも他の購読者にも
  影響しない**（例外はログに残して次のイベントへ）。購読者ごとのイベントの順序は、発行順のまま
- ``subscribe(Event, ...)`` で全イベント（記録用）。型を指定すると、その型と子クラスを受け取る
- キューがあふれたら、その購読者の**古いイベントから捨てる**（ロボットが固まるより、記録や表示が
  欠けるほうがよい）。捨てた数は ``dropped`` で見られる
- ``trace_id``: 発行時に付いていなければ、いま処理中のイベントの ``trace_id`` を引き継ぐ。
  なので、あるイベントの購読者が発行したイベントは、同じ連鎖として追える
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, SystemClock
from .events import Event

log = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None] | None]

_current_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "reachy_front_trace_id", default=None
)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def current_trace_id() -> str | None:
    """いま処理しているイベントの ``trace_id``（なければ None）。"""
    return _current_trace.get()


@dataclass
class Subscription:
    name: str
    event_type: type[Event]
    handler: Handler
    queue: deque[Event]
    wake: asyncio.Event
    max_queue: int
    dropped: int = 0
    task: asyncio.Task[None] | None = None
    busy: bool = False
    _idle: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self._idle.set()


class EventBus:
    def __init__(self, clock: Clock | None = None, *, max_queue: int = 1000) -> None:
        self._clock = clock or SystemClock()
        self._max_queue = max_queue
        self._subs: list[Subscription] = []
        self._closed = False

    # ------------------------------------------------------------------ 購読

    def subscribe(
        self,
        event_type: type[Event],
        handler: Handler,
        *,
        name: str | None = None,
        max_queue: int | None = None,
    ) -> Subscription:
        """``handler``（同期でも async でも）を登録する。動いている event loop の中で呼ぶ。"""
        if self._closed:
            raise RuntimeError("バスは閉じている")
        sub = Subscription(
            name=name or getattr(handler, "__qualname__", repr(handler)),
            event_type=event_type,
            handler=handler,
            queue=deque(),
            wake=asyncio.Event(),
            max_queue=max_queue or self._max_queue,
        )
        sub.task = asyncio.get_running_loop().create_task(self._run(sub), name=f"bus:{sub.name}")
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
            self._stop(sub)

    # ------------------------------------------------------------------ 発行

    def publish(self, event: Event) -> Event:
        """イベントを配る（待たない）。時刻と ``trace_id`` を補った Event を返す。"""
        if self._closed:
            return event
        update: dict[str, Any] = {}
        if event.ts is None:
            update["ts"] = self._clock.now()
        if event.trace_id is None and (tid := _current_trace.get()) is not None:
            update["trace_id"] = tid
        stamped = event.model_copy(update=update) if update else event
        for sub in self._subs:
            if isinstance(stamped, sub.event_type):
                self._enqueue(sub, stamped)
        return stamped

    def start_trace(self, event: Event) -> Event:
        """新しい ``trace_id`` で発行する（連鎖の起点。入室・発話の終わりなど）。"""
        return self.publish(event.model_copy(update={"trace_id": new_trace_id()}))

    @staticmethod
    def _stop(sub: Subscription) -> None:
        """タスクを止め、``drain`` で待っている人を起こす（待ち続けないように）。"""
        if sub.task is not None:
            sub.task.cancel()
        sub.queue.clear()
        sub.busy = False
        sub._idle.set()  # pyright: ignore[reportPrivateUsage]

    @staticmethod
    def _enqueue(sub: Subscription, event: Event) -> None:
        if len(sub.queue) >= sub.max_queue:
            sub.queue.popleft()
            sub.dropped += 1
            if sub.dropped == 1 or sub.dropped % 100 == 0:
                log.warning("購読者 %s のキューがあふれた（捨てた累計 %d）", sub.name, sub.dropped)
        sub.queue.append(event)
        sub._idle.clear()  # pyright: ignore[reportPrivateUsage]
        sub.wake.set()

    # ------------------------------------------------------------------ 実行

    async def _run(self, sub: Subscription) -> None:
        while True:
            await sub.wake.wait()
            while sub.queue:
                event = sub.queue.popleft()
                sub.busy = True
                token = _current_trace.set(event.trace_id)
                try:
                    result = sub.handler(event)
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("購読者 %s が %s で失敗", sub.name, type(event).__name__)
                finally:
                    _current_trace.reset(token)
                    sub.busy = False
            sub.wake.clear()
            sub._idle.set()  # pyright: ignore[reportPrivateUsage]

    async def drain(self, timeout: float | None = None) -> None:
        """いま溜まっているイベントを全員が処理し終えるまで待つ（テストと終了処理用）。

        購読者が処理中に発行したイベントも、静まるまで待つ。
        """

        async def settle() -> None:
            while True:
                for sub in list(self._subs):
                    await sub._idle.wait()  # pyright: ignore[reportPrivateUsage]
                if all(not s.queue and not s.busy for s in self._subs):
                    return

        await asyncio.wait_for(settle(), timeout)

    async def close(self) -> None:
        """購読者のタスクを止める。以降の発行は無視される。"""
        self._closed = True
        tasks = [s.task for s in self._subs if s.task is not None]
        for s in self._subs:
            self._stop(s)
        await asyncio.gather(*tasks, return_exceptions=True)
        self._subs.clear()


__all__ = ["EventBus", "Handler", "Subscription", "current_trace_id", "new_trace_id"]
