"""World: イベントバスのイベントを畳み込み、読み取り専用のスナップショットを渡す。"""

from __future__ import annotations

from ..core.bus import EventBus, Subscription
from ..core.events import Event
from .reducer import reduce
from .state import WorldState


class World:
    def __init__(self, initial: WorldState | None = None) -> None:
        self._state = initial or WorldState()

    def apply(self, event: Event) -> WorldState:
        self._state = reduce(self._state, event)
        return self._state

    def snapshot(self) -> WorldState:
        """いまの状況。不変なので、受け取った後に変わることはない。"""
        return self._state

    def attach(self, bus: EventBus) -> Subscription:
        """バスの全イベントを畳み込む購読を登録する（動いている event loop の中で呼ぶ）。"""

        def on_event(event: Event) -> None:
            self.apply(event)

        return bus.subscribe(Event, on_event, name="world")


__all__ = ["World"]
