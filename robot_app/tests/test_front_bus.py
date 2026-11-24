"""イベントバス: 購読者の隔離・順序・trace_id・あふれ。"""

from __future__ import annotations

import asyncio
import logging

import pytest

from reachy_front.core import EventBus, ManualClock, current_trace_id
from reachy_front.core import events as ev
from reachy_front.core.events import Event


@pytest.fixture
async def bus() -> EventBus:  # pyright: ignore[reportInvalidTypeForm]
    b = EventBus(ManualClock(5.0))
    yield b  # type: ignore[misc]
    await b.close()


async def test_subscribers_get_events_of_their_type_and_subclass(bus: EventBus) -> None:
    only_turn: list[Event] = []
    everything: list[Event] = []
    bus.subscribe(ev.TurnEnded, only_turn.append)
    bus.subscribe(Event, everything.append)
    bus.publish(ev.TurnEnded(turn_id="t1"))
    bus.publish(ev.ServerConnectionChanged(connected=True))
    await bus.drain(1)
    assert [type(e) for e in only_turn] == [ev.TurnEnded]
    assert [type(e) for e in everything] == [ev.TurnEnded, ev.ServerConnectionChanged]


async def test_publish_stamps_the_time_and_keeps_an_explicit_one(bus: EventBus) -> None:
    got: list[Event] = []
    bus.subscribe(Event, got.append)
    bus.publish(ev.TurnEnded(turn_id="a"))
    bus.publish(ev.TurnEnded(turn_id="b", ts=1.0))
    await bus.drain(1)
    assert [e.ts for e in got] == [5.0, 1.0]


async def test_order_is_preserved_per_subscriber(bus: EventBus) -> None:
    got: list[int] = []

    async def slow(e: Event) -> None:
        assert isinstance(e, ev.AudioChunkReceived)
        await asyncio.sleep(0)
        got.append(e.seq)

    bus.subscribe(ev.AudioChunkReceived, slow)
    for i in range(20):
        bus.publish(ev.AudioChunkReceived(turn_id="t", seq=i, nbytes=1))
    await bus.drain(1)
    assert got == list(range(20))


async def test_a_failing_subscriber_does_not_affect_others_or_later_events(
    bus: EventBus, caplog: pytest.LogCaptureFixture
) -> None:
    good: list[str] = []
    seen_by_bad: list[str] = []

    def bad(e: Event) -> None:
        assert isinstance(e, ev.TurnEnded)
        seen_by_bad.append(e.turn_id)
        raise RuntimeError("boom")

    bus.subscribe(ev.TurnEnded, bad, name="bad")
    bus.subscribe(ev.TurnEnded, lambda e: good.append(e.turn_id), name="good")  # type: ignore[attr-defined]
    with caplog.at_level(logging.ERROR):
        bus.publish(ev.TurnEnded(turn_id="a"))
        bus.publish(ev.TurnEnded(turn_id="b"))
        await bus.drain(1)
    assert good == ["a", "b"]
    assert seen_by_bad == ["a", "b"]  # 失敗しても次のイベントを受け取る
    assert "bad" in caplog.text


async def test_a_slow_subscriber_does_not_block_the_publisher_or_others(bus: EventBus) -> None:
    release = asyncio.Event()
    fast: list[str] = []

    async def stuck(_e: Event) -> None:
        await release.wait()

    bus.subscribe(ev.TurnEnded, stuck, name="stuck")
    bus.subscribe(ev.TurnEnded, lambda e: fast.append(e.turn_id), name="fast")  # type: ignore[attr-defined]
    for i in range(5):
        bus.publish(ev.TurnEnded(turn_id=str(i)))  # 待たされない
    await asyncio.sleep(0.05)
    assert fast == ["0", "1", "2", "3", "4"]
    release.set()
    await bus.drain(1)


async def test_trace_id_is_inherited_by_events_published_from_a_handler(bus: EventBus) -> None:
    out: list[Event] = []

    def on_speech_end(e: Event) -> None:
        assert current_trace_id() == e.trace_id
        bus.publish(ev.FloorDecided(utt_id="u1", decision="respond"))

    bus.subscribe(ev.SpeechEnded, on_speech_end)
    bus.subscribe(ev.FloorDecided, out.append)
    root = bus.start_trace(ev.SpeechEnded(utt_id="u1", duration_s=1.0))
    other = bus.start_trace(ev.SpeechEnded(utt_id="u2", duration_s=1.0))
    await bus.drain(1)
    assert root.trace_id and root.trace_id != other.trace_id
    assert [e.trace_id for e in out] == [root.trace_id, other.trace_id]
    assert current_trace_id() is None  # 処理の外には漏れない


async def test_an_event_without_a_trace_stays_without_one(bus: EventBus) -> None:
    got: list[Event] = []
    bus.subscribe(Event, got.append)
    bus.publish(ev.TurnEnded(turn_id="a"))
    await bus.drain(1)
    assert got[0].trace_id is None


async def test_overflow_drops_the_oldest_and_counts(bus: EventBus) -> None:
    got: list[str] = []
    gate = asyncio.Event()

    async def handler(e: Event) -> None:
        await gate.wait()
        got.append(e.turn_id)  # type: ignore[attr-defined]

    sub = bus.subscribe(ev.TurnEnded, handler, max_queue=3)
    bus.publish(ev.TurnEnded(turn_id="0"))
    await asyncio.sleep(0.02)  # 0 は処理中（gate 待ち）
    for i in range(1, 7):
        bus.publish(ev.TurnEnded(turn_id=str(i)))
    gate.set()
    await bus.drain(1)
    assert got == ["0", "4", "5", "6"]  # 古い 1〜3 が捨てられた
    assert sub.dropped == 3


async def test_unsubscribe_and_close_stop_delivery() -> None:
    bus = EventBus()
    got: list[Event] = []
    sub = bus.subscribe(Event, got.append)
    bus.publish(ev.TurnEnded(turn_id="a"))
    await bus.drain(1)
    bus.unsubscribe(sub)
    bus.publish(ev.TurnEnded(turn_id="b"))
    await asyncio.sleep(0.02)
    assert len(got) == 1
    await bus.close()
    bus.publish(ev.TurnEnded(turn_id="c"))  # 閉じた後の発行は無視（例外にしない）
    with pytest.raises(RuntimeError):
        bus.subscribe(Event, got.append)


async def test_drain_waits_for_events_published_by_handlers(bus: EventBus) -> None:
    got: list[str] = []

    async def first(_e: Event) -> None:
        await asyncio.sleep(0.01)
        bus.publish(ev.TurnEnded(turn_id="second"))

    bus.subscribe(ev.SpeechEnded, first)
    bus.subscribe(ev.TurnEnded, lambda e: got.append(e.turn_id))  # type: ignore[attr-defined]
    bus.publish(ev.SpeechEnded(utt_id="u", duration_s=1))
    await bus.drain(1)
    assert got == ["second"]
