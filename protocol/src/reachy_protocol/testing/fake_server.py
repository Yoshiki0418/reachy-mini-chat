"""プロトコル準拠の偽サーバー。ロボット側のテストで相手役にする。

STT・LLM・TTS の代わりに、設定どおりの文字起こしと返答を返す。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from ..channel import BinaryFrame, Endpoint
from ..framing import BinaryHeader
from ..messages import (
    Emotion,
    Error,
    FloorDecision,
    Hello,
    MemoryAck,
    MemoryOp,
    Message,
    Motion,
    Ping,
    Pong,
    ProactiveRequest,
    Sentence,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCancel,
    TurnContext,
    TurnEnd,
    TurnStart,
    UtteranceEnd,
    Welcome,
)
from ..transport import Transport, TransportClosed
from ..version import is_compatible

MemoryOpSpec = tuple[str, str | None]  # (op, value)


def _default_transcribe(_utt_id: str, _pcm: bytes) -> str:
    return "こんにちは"


@dataclass
class FakeServerConfig:
    token: str = "test-token"
    server_version: str = "fake-0.1"
    transcribe: Callable[[str, bytes], str] = _default_transcribe
    reply: Sequence[str] = ("こんにちは！", "今日も来てくれてうれしいな。")
    emotions: Sequence[Emotion] = ("happy", "neutral")
    motion: str | None = "nod"
    memory_ops: Sequence[MemoryOpSpec] = ()
    """各ターンの最初の文の後に発行する記憶操作。person_id は context.person.id を使う。"""
    tool_calls: Sequence[tuple[str, dict[str, JsonValue]]] = ()
    """各ターンの冒頭でロボットに依頼するツール呼び出し（結果を最大 1 秒待つ）。"""
    sentence_interval: float = 0.0
    """文と文の間の待ち時間。取消のテストで使う。"""


@dataclass
class FakeServerLog:
    received: list[Message] = field(default_factory=list[Message])
    audio: dict[str, bytes] = field(default_factory=dict[str, bytes])
    contexts: list[TurnContext] = field(default_factory=list[TurnContext])
    acks: list[MemoryAck] = field(default_factory=list[MemoryAck])
    tool_results: list[ToolResult] = field(default_factory=list[ToolResult])


def fake_wav(text: str) -> bytes:
    """中身は検査されないので、文ごとに区別できるダミーの WAV バイト列。"""
    return b"RIFF" + text.encode("utf-8")


class FakeServer:
    def __init__(self, config: FakeServerConfig | None = None) -> None:
        self.config = config or FakeServerConfig()
        self.log = FakeServerLog()
        self._turn_seq = 0
        self._op_seq = 0
        self._call_seq = 0
        self._cancel: dict[str, asyncio.Event] = {}
        self._tool_waiters: dict[str, asyncio.Future[ToolResult]] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    async def serve(self, transport: Transport) -> None:
        """1 接続ぶんの処理。相手が閉じるまで動く。"""
        ep = Endpoint(transport, "server")
        try:
            while True:
                item = await ep.recv()
                if isinstance(item, BinaryFrame):
                    if item.header.kind == "audio_in":
                        ref = item.header.ref
                        self.log.audio[ref] = self.log.audio.get(ref, b"") + item.payload
                    continue
                self.log.received.append(item)
                if not await self._handle(ep, item):
                    break
        except TransportClosed:
            pass
        finally:
            for task in self._tasks:
                task.cancel()
            for task in self._tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await ep.close()

    async def _handle(self, ep: Endpoint, msg: Message) -> bool:
        """False を返すと接続を閉じる。"""
        if isinstance(msg, Hello):
            if not is_compatible(msg.protocol_version):
                await ep.send(Error(code="incompatible_version", message=msg.protocol_version))
                return False
            if msg.token.get_secret_value() != self.config.token:
                await ep.send(Error(code="unauthorized"))
                return False
            await ep.send(
                Welcome(server_version=self.config.server_version, models={"stt": "fake"})
            )
        elif isinstance(msg, UtteranceEnd):
            pcm = self.log.audio.get(msg.utt_id, b"")
            await ep.send(
                Transcript(utt_id=msg.utt_id, text=self.config.transcribe(msg.utt_id, pcm))
            )
        elif isinstance(msg, FloorDecision):
            if msg.decision == "respond" and msg.context is not None:
                self._spawn(ep, msg.utt_id, msg.context)
        elif isinstance(msg, ProactiveRequest):
            self._spawn(ep, msg.intent_id, msg.context)
        elif isinstance(msg, TurnCancel):
            event = self._cancel.get(msg.turn_id)
            if event is not None:
                event.set()
        elif isinstance(msg, MemoryAck):
            self.log.acks.append(msg)
        elif isinstance(msg, ToolResult):
            self.log.tool_results.append(msg)
            waiter = self._tool_waiters.pop(msg.call_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(msg)
        elif isinstance(msg, Ping):
            await ep.send(Pong(nonce=msg.nonce))
        return True

    def _spawn(self, ep: Endpoint, ref: str, context: TurnContext) -> None:
        self.log.contexts.append(context)
        self._turn_seq += 1
        turn_id = f"t{self._turn_seq}"
        self._cancel[turn_id] = asyncio.Event()
        task = asyncio.create_task(self._turn(ep, turn_id, ref, context))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _turn(self, ep: Endpoint, turn_id: str, ref: str, context: TurnContext) -> None:
        cfg = self.config
        cancel = self._cancel[turn_id]
        await ep.send(TurnStart(turn_id=turn_id, ref=ref))
        for name, args in cfg.tool_calls:
            await self._call_tool(ep, name, args)
        for idx, text in enumerate(cfg.reply):
            if cancel.is_set():
                break
            emotion = cfg.emotions[idx % len(cfg.emotions)] if cfg.emotions else "neutral"
            await ep.send(Sentence(turn_id=turn_id, idx=idx, text=text, emotion=emotion))
            header = BinaryHeader(kind="audio_out", ref=turn_id, seq=idx, format="wav_pcm16")
            await ep.send_frame(header, fake_wav(text))
            if idx == 0:
                if cfg.motion:
                    await ep.send(Motion(turn_id=turn_id, name=cfg.motion))
                await self._send_memory_ops(ep, context)
            if cfg.sentence_interval > 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(cancel.wait(), cfg.sentence_interval)
        await ep.send(TurnEnd(turn_id=turn_id, status="cancelled" if cancel.is_set() else "ok"))

    async def _send_memory_ops(self, ep: Endpoint, context: TurnContext) -> None:
        if context.person is None:
            return
        for op, value in self.config.memory_ops:
            self._op_seq += 1
            await ep.send(
                MemoryOp.model_validate(
                    {
                        "op_id": f"o{self._op_seq}",
                        "person_id": context.person.id,
                        "op": op,
                        "value": value,
                    }
                )
            )

    async def _call_tool(self, ep: Endpoint, name: str, args: dict[str, JsonValue]) -> None:
        self._call_seq += 1
        call_id = f"c{self._call_seq}"
        waiter: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
        self._tool_waiters[call_id] = waiter
        await ep.send(ToolCall(call_id=call_id, name=name, args=args))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(waiter, 1.0)
