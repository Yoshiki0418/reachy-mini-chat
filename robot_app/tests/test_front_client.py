"""サーバークライアントと FakeServer の契約テスト（メモリ上の通信路、実機・ネットワークなし）。"""

from __future__ import annotations

import asyncio
import random
import struct
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from pydantic import SecretStr

from reachy_front.client import AudioClip, LinkConfig, ServerClient
from reachy_front.core import EventBus
from reachy_front.core import events as ev
from reachy_front.core.events import Event
from reachy_protocol.conformance import ConversationChecker
from reachy_protocol.messages import (
    Error,
    FloorDecision,
    MemoryOp,
    Message,
    Motion,
    ToolCall,
    UtteranceEnd,
    UtteranceFeatures,
    Welcome,
)
from reachy_protocol.testing.fake_robot import default_context
from reachy_protocol.testing.fake_server import FakeServer, FakeServerConfig
from reachy_protocol.testing.tap import MonitoredTransport
from reachy_protocol.transport import Transport, memory_pair

TOKEN = "test-token-0123456789"


def config(**kw: Any) -> LinkConfig:
    base: dict[str, Any] = {
        "robot_id": "reachy-test",
        "token": SecretStr(TOKEN),
        "backoff_initial_s": 0.02,
        "backoff_max_s": 0.1,
        "ping_interval_s": 0.05,
        "pong_timeout_s": 0.1,
        "handshake_timeout_s": 0.5,
        "connect_timeout_s": 0.5,
    }
    return LinkConfig(**{**base, **kw})


class Harness:
    """FakeServer につなぐ connect 関数と、観測用の記録。"""

    def __init__(self, server: FakeServer | None = None) -> None:
        self.server = server or FakeServer(FakeServerConfig(token=TOKEN))
        self.checker = ConversationChecker()
        self.connects = 0
        self.fail_next = 0
        self.serve_tasks: list[asyncio.Task[None]] = []

    async def connect(self) -> Transport:
        self.connects += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ConnectionRefusedError
        robot_side, server_side = memory_pair()
        self.serve_tasks.append(asyncio.create_task(self.server.serve(server_side)))
        return MonitoredTransport(robot_side, self.checker, "robot")


@pytest.fixture
async def bus() -> AsyncIterator[EventBus]:
    b = EventBus()
    yield b
    await b.close()


def record(bus: EventBus) -> list[Event]:
    got: list[Event] = []
    bus.subscribe(Event, got.append)
    return got


async def until(cond: Callable[[], bool], timeout: float = 2.0) -> None:
    async def poll() -> None:
        while not cond():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


def make_client(h: Harness, bus: EventBus, **kw: Any) -> ServerClient:
    return ServerClient(h.connect, bus, config(**kw), rng=random.Random(0))


def pcm(n: int = 1600) -> bytes:
    return struct.pack(f"<{n}f", *([0.0] * n))


def features() -> UtteranceFeatures:
    return UtteranceFeatures(duration_s=1.0, eye_contact_ratio=1.0, people_count=1)


async def talk(client: ServerClient, utt_id: str = "u1", chunks: int = 3) -> None:
    assert await client.begin_utterance(utt_id, doa_rad=1.5)
    for _ in range(chunks):
        assert await client.send_audio(utt_id, pcm())
    assert await client.end_utterance(UtteranceEnd(utt_id=utt_id, features=features()))


# ---------------------------------------------------------------- 接続


async def test_handshake_publishes_connection_and_follows_the_contract(bus: EventBus) -> None:
    got = record(bus)
    h = Harness()
    client = make_client(h, bus)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    assert client.connected and client.server_version == "fake-0.1"
    await bus.drain(1)
    assert [e for e in got if isinstance(e, ev.ServerConnectionChanged)] == [
        ev.ServerConnectionChanged(connected=True).model_copy(update={"ts": got[0].ts})
    ]
    await client.close()
    assert h.checker.violations == []


async def test_wrong_token_is_fatal_and_not_retried(bus: EventBus) -> None:
    h = Harness(FakeServer(FakeServerConfig(token="another-token-0123456789")))
    got = record(bus)
    client = make_client(h, bus)
    client.start()
    await until(lambda: client.fatal is not None)
    await asyncio.sleep(0.2)  # バックオフの何倍も待っても、繋ぎ直さない
    assert client.fatal == "unauthorized"
    assert h.connects == 1
    await bus.drain(1)
    assert not [e for e in got if isinstance(e, ev.ServerConnectionChanged)]
    await client.close()


async def test_incompatible_protocol_version_is_fatal(bus: EventBus) -> None:
    class OldServer(FakeServer):
        async def _handle(self, ep: Any, msg: Message) -> bool:
            from reachy_protocol.messages import Hello

            if isinstance(msg, Hello):
                await ep.send(
                    Welcome.model_validate({"protocol_version": "9.0.0", "server_version": "x"})
                )
                return True
            return await super()._handle(ep, msg)

    h = Harness(OldServer(FakeServerConfig(token=TOKEN)))
    client = make_client(h, bus)
    client.start()
    await until(lambda: client.fatal is not None)
    assert client.fatal and client.fatal.startswith("incompatible_version")
    await asyncio.sleep(0.1)
    assert h.connects == 1
    await client.close()


async def test_reconnects_with_backoff_after_connection_failures(bus: EventBus) -> None:
    h = Harness()
    h.fail_next = 3
    client = make_client(h, bus)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 3)
    assert h.connects == 4 and client.attempts == 4
    await client.close()


async def test_a_dropped_connection_ends_open_turns_and_reconnects(bus: EventBus) -> None:
    got = record(bus)
    server = FakeServer(FakeServerConfig(token=TOKEN, sentence_interval=5.0))
    h = Harness(server)
    client = make_client(h, bus)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client)
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: client.audio.qsize() >= 1)  # 1 文目が届いた（2 文目は 5 秒待ち）
    h.serve_tasks[0].cancel()  # サーバーが落ちる
    await until(lambda: h.connects >= 2 and client.connected)
    await bus.drain(1)
    kinds = [type(e).__name__ for e in got]
    assert "TurnEnded" in kinds
    ended = next(e for e in got if isinstance(e, ev.TurnEnded))
    assert ended.cancelled is True
    changes = [e.connected for e in got if isinstance(e, ev.ServerConnectionChanged)]
    assert changes == [True, False, True]
    assert client.audio.empty()  # 古い音声は捨てる
    await client.close()


async def test_missing_pong_drops_the_connection_and_reconnects(bus: EventBus) -> None:
    class Mute(FakeServer):
        async def _handle(self, ep: Any, msg: Message) -> bool:
            from reachy_protocol.messages import Ping

            if isinstance(msg, Ping):
                return True  # pong を返さない
            return await super()._handle(ep, msg)

    h = Harness(Mute(FakeServerConfig(token=TOKEN)))
    client = make_client(h, bus, ping_interval_s=0.03, pong_timeout_s=0.05)
    client.start()
    await until(lambda: h.connects >= 2, timeout=3)
    await client.close()


async def test_sends_return_false_while_disconnected(bus: EventBus) -> None:
    h = Harness()
    client = make_client(h, bus)  # start しない
    assert await client.begin_utterance("u1") is False
    assert await client.send_audio("u1", pcm()) is False
    assert await client.cancel_turn("t1") is False


# ---------------------------------------------------------------- 会話


async def test_full_turn_through_the_bus_and_audio_queue(bus: EventBus) -> None:
    got = record(bus)
    h = Harness()
    client = make_client(h, bus)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client, chunks=3)
    await until(lambda: any(isinstance(e, ev.TranscriptReceived) for e in got))
    await bus.drain(1)
    transcript = next(e for e in got if isinstance(e, ev.TranscriptReceived))
    assert (transcript.utt_id, transcript.text) == ("u1", "こんにちは")
    # 発話の音声はチャンクの順に全部届いている
    assert len(h.server.log.audio["u1"]) == 3 * len(pcm())
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: any(isinstance(e, ev.TurnEnded) for e in got))
    await bus.drain(1)
    sentences = [e for e in got if isinstance(e, ev.SentenceReceived)]
    assert [(s.idx, s.text) for s in sentences] == [
        (0, "こんにちは！"),
        (1, "今日も来てくれてうれしいな。"),
    ]
    assert sentences[0].emotion == "happy"
    clips: list[AudioClip] = [client.audio.get_nowait() for _ in range(client.audio.qsize())]
    assert [c.idx for c in clips] == [0, 1]
    assert all(c.wav.startswith(b"RIFF") for c in clips)
    ended = next(e for e in got if isinstance(e, ev.TurnEnded))
    assert ended.cancelled is False
    assert sum(isinstance(e, ev.AudioChunkReceived) for e in got) == 2
    await client.close()
    assert h.checker.finish() == []  # 順序規約（hello→welcome、発話、ターン）の違反なし


async def test_cancel_stops_the_turn_and_drops_its_audio(bus: EventBus) -> None:
    got = record(bus)
    h = Harness(FakeServer(FakeServerConfig(token=TOKEN, sentence_interval=1.0)))
    client = make_client(h, bus)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client)
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: client.audio.qsize() >= 1)
    turn_id = client.audio.get_nowait().turn_id
    assert await client.cancel_turn(turn_id, "barge_in")
    await until(lambda: any(isinstance(e, ev.TurnEnded) for e in got))
    await bus.drain(1)
    ended = next(e for e in got if isinstance(e, ev.TurnEnded))
    assert (ended.turn_id, ended.cancelled) == (turn_id, True)
    assert client.audio.empty()  # 取り消した後に届いた音声も入らない
    await client.close()


async def test_cancel_purges_queued_clips_of_that_turn_only(bus: EventBus) -> None:
    h = Harness()
    client = make_client(h, bus)
    for turn, idx in (("t1", 0), ("t2", 0), ("t1", 1)):
        client.audio.put_nowait(AudioClip(turn, idx, b"x"))
    await client.cancel_turn("t1")
    assert [(c.turn_id, c.idx) for c in [client.audio.get_nowait()]] == [("t2", 0)]
    assert client.audio.empty()


async def test_audio_queue_is_bounded(bus: EventBus) -> None:
    from reachy_protocol.channel import BinaryFrame
    from reachy_protocol.framing import BinaryHeader

    client = make_client(Harness(), bus, max_audio_queue=3)
    for i in range(6):
        header = BinaryHeader(kind="audio_out", ref="t", seq=i, format="wav_pcm16")
        client._on_audio(BinaryFrame(header, b"x"))  # pyright: ignore[reportPrivateUsage]
    assert [client.audio.get_nowait().idx for _ in range(3)] == [3, 4, 5]  # 古い順に捨てる


# ---------------------------------------------------------------- サーバーからの依頼


async def test_memory_op_and_tool_call_use_registered_handlers(bus: EventBus) -> None:
    server = FakeServer(
        FakeServerConfig(
            token=TOKEN, memory_ops=[("set_name", "さくら")], tool_calls=[("lab_presence", {})]
        )
    )
    h = Harness(server)
    client = make_client(h, bus)
    seen: list[str] = []

    async def on_memory(msg: Message) -> None:
        assert isinstance(msg, MemoryOp)
        seen.append(f"op:{msg.op}:{msg.value}")
        await client.reply_memory_ack(msg.op_id, "applied")

    async def on_tool(msg: Message) -> None:
        assert isinstance(msg, ToolCall)
        seen.append(f"tool:{msg.name}")
        await client.reply_tool_result(msg.call_id, ok=True, result=["さくら"])

    client.on_message(MemoryOp, on_memory)
    client.on_message(ToolCall, on_tool)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client)
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: len(server.log.acks) == 1 and len(server.log.tool_results) == 1)
    assert sorted(seen) == ["op:set_name:さくら", "tool:lab_presence"]
    assert server.log.acks[0].result == "applied"
    assert server.log.tool_results[0].result == ["さくら"]
    await client.close()


async def test_unhandled_requests_get_a_default_refusal_so_the_server_never_waits(
    bus: EventBus,
) -> None:
    server = FakeServer(
        FakeServerConfig(
            token=TOKEN, memory_ops=[("add_fact", "好きな色は青")], tool_calls=[("x", {})]
        )
    )
    h = Harness(server)
    client = make_client(h, bus)  # 処理を登録しない
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client)
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: server.log.acks and server.log.tool_results)  # type: ignore[arg-type]
    assert server.log.acks[0].result == "rejected"
    assert server.log.tool_results[0].ok is False
    await client.close()


async def test_a_failing_handler_does_not_kill_the_connection(bus: EventBus) -> None:
    server = FakeServer(FakeServerConfig(token=TOKEN, motion="nod"))
    h = Harness(server)
    client = make_client(h, bus)
    calls: list[str] = []

    async def bad(msg: Message) -> None:
        assert isinstance(msg, Motion)
        calls.append(msg.name)
        raise RuntimeError("boom")

    client.on_message(Motion, bad)
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await talk(client)
    await client.decide_floor(
        FloorDecision(utt_id="u1", decision="respond", context=default_context())
    )
    await until(lambda: calls == ["nod"])
    assert client.connected
    await talk(client, "u2")  # つづけて使える
    await client.close()


async def test_server_error_is_logged_not_fatal(bus: EventBus) -> None:
    client = make_client(Harness(), bus)
    client._dispatch(Error(code="timeout", message="slow"))  # pyright: ignore[reportPrivateUsage]
    assert client.fatal is None
