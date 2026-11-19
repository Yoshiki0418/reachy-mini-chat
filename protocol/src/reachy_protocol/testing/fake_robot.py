"""偽ロボット。サーバー側のテストで相手役にし、汎用シナリオの操作役も務める。

受信は裏のタスクで行い、memory.op / tool.call / ping には自動で応答する。
それ以外のメッセージはキューに入り、``say()`` や ``next_turn()`` が順に読む。
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue, SecretStr

from ..channel import BinaryFrame, Endpoint, Incoming
from ..framing import BinaryHeader
from ..messages import (
    Error,
    FloorDecision,
    Hello,
    MemoryAck,
    MemoryOp,
    Motion,
    PersonProfile,
    Ping,
    Pong,
    ProactiveRequest,
    Sentence,
    Situation,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCancel,
    TurnContext,
    TurnEnd,
    TurnStart,
    UtteranceBegin,
    UtteranceEnd,
    UtteranceFeatures,
    Welcome,
)
from ..transport import Transport, TransportClosed
from ..version import PROTOCOL_VERSION

MemoryPolicy = Callable[[MemoryOp], tuple[Literal["applied", "rejected"], str | None]]
ToolHandler = Callable[[dict[str, JsonValue]], JsonValue]


def accept_all(_op: MemoryOp) -> tuple[Literal["applied", "rejected"], str | None]:
    return "applied", None


def default_context(
    *, person_id: str | None = "p-test", name: str | None = "テスト"
) -> TurnContext:
    person = (
        PersonProfile(id=person_id, name=name, familiarity="regular", visit_count=5)
        if person_id
        else None
    )
    return TurnContext(
        person=person,
        situation=Situation(people_count=1, local_time=datetime(2026, 11, 19, 9, 0, tzinfo=UTC)),
    )


def silence_pcm(seconds: float = 0.5, sr: int = 16000) -> bytes:
    """float32 little-endian の無音。"""
    n = int(seconds * sr)
    return struct.pack(f"<{n}f", *([0.0] * n))


@dataclass
class TurnResult:
    turn_id: str
    ref: str
    status: str
    sentences: list[Sentence] = field(default_factory=list[Sentence])
    audio: dict[int, bytes] = field(default_factory=dict[int, bytes])
    motions: list[str] = field(default_factory=list[str])
    memory_ops: list[MemoryOp] = field(default_factory=list[MemoryOp])

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.sentences)


class FakeRobotClient:
    def __init__(
        self,
        transport: Transport,
        *,
        robot_id: str = "reachy-test",
        token: str = "test-token",
        memory_policy: MemoryPolicy = accept_all,
        tools: dict[str, ToolHandler] | None = None,
    ) -> None:
        self.ep = Endpoint(transport, "robot")
        self.robot_id = robot_id
        self.token = token
        self.memory_policy = memory_policy
        self.tools = tools or {}
        self.received: list[Incoming] = []
        self.memory_ops: list[MemoryOp] = []
        self.tool_calls: list[ToolCall] = []
        self._queue: asyncio.Queue[Incoming | None] = asyncio.Queue()
        self._pongs: dict[str, asyncio.Future[None]] = {}
        self._seq = 0
        self._reader: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ ライフサイクル

    async def __aenter__(self) -> FakeRobotClient:
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self.ep.close()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader

    def _new_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    async def _read_loop(self) -> None:
        try:
            while True:
                item = await self.ep.recv()
                self.received.append(item)
                if isinstance(item, MemoryOp):
                    self.memory_ops.append(item)
                    result, reason = self.memory_policy(item)
                    await self.ep.send(MemoryAck(op_id=item.op_id, result=result, reason=reason))
                    await self._queue.put(item)  # ターン結果にも載せる
                elif isinstance(item, ToolCall):
                    self.tool_calls.append(item)
                    await self.ep.send(self._run_tool(item))
                elif isinstance(item, Ping):
                    await self.ep.send(Pong(nonce=item.nonce))
                elif isinstance(item, Pong):
                    waiter = self._pongs.pop(item.nonce, None)
                    if waiter is not None and not waiter.done():
                        waiter.set_result(None)
                else:
                    await self._queue.put(item)
        except TransportClosed:
            await self._queue.put(None)

    def _run_tool(self, call: ToolCall) -> ToolResult:
        handler = self.tools.get(call.name)
        if handler is None:
            return ToolResult(call_id=call.call_id, ok=False, error=f"unknown tool {call.name}")
        try:
            return ToolResult(call_id=call.call_id, ok=True, result=handler(call.args))
        except Exception as e:  # ツールの失敗は結果として返す
            return ToolResult(call_id=call.call_id, ok=False, error=str(e))

    async def _next(self, timeout: float) -> Incoming:
        item = await asyncio.wait_for(self._queue.get(), timeout)
        if item is None:
            await self._queue.put(None)
            raise TransportClosed
        return item

    # ------------------------------------------------------------ 操作

    async def connect(
        self, *, protocol_version: str = PROTOCOL_VERSION, timeout: float = 2.0
    ) -> Welcome | Error:
        """hello を送り、welcome か error を返す。"""
        await self.ep.send(
            Hello(
                robot_id=self.robot_id,
                token=SecretStr(self.token),
                protocol_version=protocol_version,
            )
        )
        while True:
            item = await self._next(timeout)
            if isinstance(item, Welcome | Error):
                return item

    async def say(
        self,
        pcm: bytes | None = None,
        *,
        features: UtteranceFeatures | None = None,
        chunk_bytes: int = 3200,
        sr: int = 16000,
        timeout: float = 2.0,
    ) -> Transcript:
        """1 発話ぶんの音声を送り、文字起こしを待つ。"""
        pcm = silence_pcm() if pcm is None else pcm
        utt_id = self._new_id("u")
        await self.ep.send(UtteranceBegin(utt_id=utt_id))
        for seq, start in enumerate(range(0, len(pcm), chunk_bytes)):
            header = BinaryHeader(kind="audio_in", ref=utt_id, seq=seq, format="pcm_f32le", sr=sr)
            await self.ep.send_frame(header, pcm[start : start + chunk_bytes])
        features = features or UtteranceFeatures(
            duration_s=len(pcm) / 4 / sr, eye_contact_ratio=1.0, people_count=1
        )
        await self.ep.send(UtteranceEnd(utt_id=utt_id, features=features))
        while True:
            item = await self._next(timeout)
            if isinstance(item, Transcript) and item.utt_id == utt_id:
                return item
            if isinstance(item, Error) and item.ref == utt_id:
                raise AssertionError(f"文字起こしがエラーになった: {item}")

    async def decide(
        self,
        utt_id: str,
        decision: Literal["respond", "backchannel", "ignore"],
        context: TurnContext | None = None,
    ) -> None:
        if decision == "respond" and context is None:
            context = default_context()
        await self.ep.send(FloorDecision(utt_id=utt_id, decision=decision, context=context))

    async def request_proactive(
        self,
        kind: Literal["greet", "engage", "see_off"] = "greet",
        context: TurnContext | None = None,
        strategy: str | None = None,
    ) -> str:
        intent_id = self._new_id("i")
        await self.ep.send(
            ProactiveRequest(
                intent_id=intent_id,
                kind=kind,
                strategy=strategy,
                context=context or default_context(),
            )
        )
        return intent_id

    async def wait_turn_start(self, timeout: float = 2.0) -> TurnStart:
        while True:
            item = await self._next(timeout)
            if isinstance(item, TurnStart):
                return item

    async def next_turn(self, timeout: float = 5.0, start: TurnStart | None = None) -> TurnResult:
        """次のターンを turn.end まで集める。"""
        start = start or await self.wait_turn_start(timeout)
        result = TurnResult(turn_id=start.turn_id, ref=start.ref, status="?")
        while True:
            item = await self._next(timeout)
            if isinstance(item, Sentence) and item.turn_id == start.turn_id:
                result.sentences.append(item)
            elif isinstance(item, BinaryFrame) and item.header.ref == start.turn_id:
                result.audio[item.header.seq] = item.payload
            elif isinstance(item, Motion) and item.turn_id == start.turn_id:
                result.motions.append(item.name)
            elif isinstance(item, MemoryOp):
                result.memory_ops.append(item)
            elif isinstance(item, TurnEnd) and item.turn_id == start.turn_id:
                result.status = item.status
                return result

    async def cancel(self, turn_id: str) -> None:
        await self.ep.send(TurnCancel(turn_id=turn_id, reason="barge_in"))

    async def expect_silence(self, seconds: float = 0.3) -> None:
        """指定時間、サーバーから何も（自動応答の対象以外）届かないことを確かめる。"""
        try:
            item = await self._next(seconds)
        except TimeoutError:
            return
        raise AssertionError(f"沈黙を期待したが {type(item).__name__} が届いた: {item}")

    async def ping(self, timeout: float = 2.0) -> None:
        nonce = self._new_id("n")
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pongs[nonce] = waiter
        await self.ep.send(Ping(nonce=nonce))
        await asyncio.wait_for(waiter, timeout)
