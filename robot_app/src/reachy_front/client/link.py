"""サーバーとの接続の状態機械（設計書 §3.1・§2）。

``ServerClient`` は 1 本の接続を保ち、切れたら指数バックオフで繋ぎ直す。受け取ったものは
イベントバスのイベント（``TranscriptReceived`` など）と、音声のキュー、登録した処理に振り分ける。

- 通信路は ``connect()``（``Transport`` を返す関数）で差し込む。本番は WebSocket（:mod:`.ws`）、
  テストはメモリ上の ``FakeServer``
- **認証の失敗・版の不一致は繰り返さない**（トークンが違うのに叩き続けない）。``fatal`` に理由が残る
- 切れたとき、終わっていないターンには ``TurnEnded(cancelled=True)`` を出し、再生待ちの音声は捨てる
  （繋ぎ直した後に古い音声が流れないように）
- 送信系は、繋がっていなければ ``False`` を返して何もしない（音声の送信を例外で止めない）
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from pydantic import JsonValue, SecretStr

from reachy_protocol.channel import BinaryFrame, Endpoint
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    Error,
    FloorDecision,
    Hello,
    MemoryAck,
    MemoryOp,
    Message,
    PersonForget,
    Ping,
    Pong,
    ProactiveRequest,
    Sentence,
    SessionClose,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCancel,
    TurnEnd,
    TurnStart,
    UtteranceBegin,
    UtteranceEnd,
    Welcome,
)
from reachy_protocol.transport import Transport, TransportClosed
from reachy_protocol.version import is_compatible

from ..core import events as ev
from ..core.bus import EventBus

log = logging.getLogger(__name__)

Connect = Callable[[], Awaitable[Transport]]
Handler = Callable[[Message], Awaitable[None]]

_FATAL_ERRORS = {"unauthorized", "incompatible_version"}


@dataclass(frozen=True)
class AudioClip:
    """サーバーが合成した 1 文の音声（WAV）。再生は expression が行う。"""

    turn_id: str
    idx: int
    wav: bytes


@dataclass
class LinkConfig:
    robot_id: str
    token: SecretStr
    connect_timeout_s: float = 5.0
    handshake_timeout_s: float = 5.0
    ping_interval_s: float = 5.0
    pong_timeout_s: float = 3.0
    backoff_initial_s: float = 0.5
    backoff_max_s: float = 15.0
    max_audio_queue: int = 64
    capabilities: list[str] = field(default_factory=list[str])


class NotConnected(Exception):
    pass


class ServerClient:
    def __init__(
        self,
        connect: Connect,
        bus: EventBus,
        config: LinkConfig,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._connect = connect
        self._bus = bus
        self._cfg = config
        self._rng = rng or random.Random()
        self._ep: Endpoint | None = None
        self._send_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._handlers: dict[str, Handler] = {}
        self._handler_tasks: set[asyncio.Task[Any]] = set()
        self._pongs: dict[str, asyncio.Event] = {}
        self._seq: dict[str, int] = {}
        self._open_turns: set[str] = set()
        self._cancelled: deque[str] = deque(maxlen=64)
        self.audio: asyncio.Queue[AudioClip] = asyncio.Queue()
        self.connected = False
        self.fatal: str | None = None
        """認証の失敗・版の不一致の理由。入ったら繋ぎ直さない。"""
        self.server_version: str | None = None
        self.attempts = 0
        self.ready = asyncio.Event()
        """初めて welcome を受けたら立つ（テストと起動待ち用）。"""

    # ---------------------------------------------------------------- 起動と終了

    def start(self) -> None:
        """接続の維持を始める（動いている event loop の中で呼ぶ）。"""
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run(), name="server-client")

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for t in list(self._handler_tasks):
            t.cancel()
        await asyncio.gather(*self._handler_tasks, return_exceptions=True)

    def on_message(self, msg_type: type[Message], handler: Handler) -> None:
        """``MemoryOp`` / ``ToolCall`` / ``Motion`` など、イベントにしないメッセージの処理を登録する。"""
        self._handlers[msg_type.model_fields["type"].default] = handler

    # ---------------------------------------------------------------- 接続の維持

    async def _run(self) -> None:
        delay = self._cfg.backoff_initial_s
        while self.fatal is None:
            self.attempts += 1
            try:
                welcomed = await self._session()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("サーバーとの接続でエラー")
                welcomed = False
            if self.fatal is not None:
                break
            if welcomed:
                delay = self._cfg.backoff_initial_s  # 一度つながったら、バックオフは最初から
            wait = delay * self._rng.uniform(0.8, 1.2)
            log.info("サーバーに繋ぎ直す（%.1f 秒後）", wait)
            await asyncio.sleep(wait)
            delay = min(delay * 2, self._cfg.backoff_max_s)

    async def _session(self) -> bool:
        """1 回の接続。welcome まで進んだら True。"""
        try:
            transport = await asyncio.wait_for(self._connect(), self._cfg.connect_timeout_s)
        except (OSError, TimeoutError, TransportClosed) as e:
            log.warning("サーバーに繋がらない: %s", type(e).__name__)
            return False
        ep = Endpoint(transport, "robot")
        welcomed = False
        tasks: list[asyncio.Task[None]] = []
        try:
            await ep.send(
                Hello(
                    robot_id=self._cfg.robot_id,
                    token=self._cfg.token,
                    capabilities=self._cfg.capabilities,
                )
            )
            welcome = await asyncio.wait_for(
                self._expect_welcome(ep), self._cfg.handshake_timeout_s
            )
            if welcome is None:
                return False
            welcomed = True
            self._ep = ep
            self.server_version = welcome.server_version
            self.connected = True
            self.ready.set()
            self._bus.publish(ev.ServerConnectionChanged(connected=True))
            tasks = [
                asyncio.create_task(self._heartbeat(ep), name="client-heartbeat"),
                asyncio.create_task(self._reader(ep), name="client-reader"),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (TransportClosed, TimeoutError):
            pass
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._on_disconnected(ep, was_connected=welcomed)
        return welcomed

    async def _expect_welcome(self, ep: Endpoint) -> Welcome | None:
        while True:
            try:
                item = await ep.recv()
            except ValueError:
                log.warning("welcome の前に不正なフレームを受けた")
                return None
            if isinstance(item, Welcome):
                if not is_compatible(item.protocol_version):
                    self.fatal = f"incompatible_version: {item.protocol_version}"
                    log.error("サーバーのプロトコル版が合わない: %s", item.protocol_version)
                    return None
                return item
            if isinstance(item, Error):
                if item.code in _FATAL_ERRORS:
                    self.fatal = item.code
                    log.error("サーバーに拒否された: %s", item.code)
                return None
            # welcome の前の他のメッセージは無視

    async def _on_disconnected(self, ep: Endpoint, *, was_connected: bool) -> None:
        with contextlib.suppress(Exception):
            await ep.close()
        if self._ep is ep:
            self._ep = None
        self.connected = False
        self._seq.clear()
        self._pongs.clear()
        # 繋ぎ直した後に、古いターンの音声が流れないようにする
        while not self.audio.empty():
            self.audio.get_nowait()
        for turn_id in sorted(self._open_turns):
            self._bus.publish(ev.TurnEnded(turn_id=turn_id, cancelled=True))
        self._open_turns.clear()
        if was_connected:
            self._bus.publish(ev.ServerConnectionChanged(connected=False))

    # ---------------------------------------------------------------- 受信

    async def _reader(self, ep: Endpoint) -> None:
        while True:
            try:
                item = await ep.recv()
            except ValueError as e:  # 規約に合わないフレームは捨てて続ける（接続は切らない）
                log.warning("不正なフレームを無視: %s", e)
                continue
            if isinstance(item, BinaryFrame):
                self._on_audio(item)
            else:
                self._dispatch(item)

    def _on_audio(self, frame: BinaryFrame) -> None:
        h = frame.header
        if h.kind != "audio_out" or h.ref in self._cancelled:
            return  # 取り消したターンの音声は捨てる
        if self.audio.qsize() >= self._cfg.max_audio_queue:
            self.audio.get_nowait()  # 溜まりすぎたら古い順に捨てる
        self.audio.put_nowait(AudioClip(h.ref, h.seq, frame.payload))
        self._bus.publish(
            ev.AudioChunkReceived(turn_id=h.ref, seq=h.seq, nbytes=len(frame.payload))
        )

    def _dispatch(self, msg: Message) -> None:
        match msg:
            case Pong():
                if (waiter := self._pongs.get(msg.nonce)) is not None:
                    waiter.set()
                return
            case Ping():
                self._spawn(self._send(Pong(nonce=msg.nonce)))
                return
            case Transcript():
                self._bus.publish(ev.TranscriptReceived(utt_id=msg.utt_id, text=msg.text))
            case TurnStart():
                self._open_turns.add(msg.turn_id)
            case Sentence():
                self._bus.publish(
                    ev.SentenceReceived(
                        turn_id=msg.turn_id, idx=msg.idx, text=msg.text, emotion=msg.emotion
                    )
                )
            case TurnEnd():
                self._open_turns.discard(msg.turn_id)
                self._bus.publish(
                    ev.TurnEnded(turn_id=msg.turn_id, cancelled=msg.status == "cancelled")
                )
            case Error():
                log.warning("サーバーのエラー: %s %s", msg.code, msg.message)
                return
            case _:
                pass
        if (handler := self._handlers.get(msg.type)) is not None:
            self._spawn(self._guarded(handler, msg))
        else:
            self._reject_unhandled(msg)

    def _reject_unhandled(self, msg: Message) -> None:
        """返事を待っているサーバーを待たせない（処理が登録されていないときの既定の返事）。"""
        if isinstance(msg, MemoryOp):
            self._spawn(
                self._send(MemoryAck(op_id=msg.op_id, result="rejected", reason="unsupported"))
            )
        elif isinstance(msg, ToolCall):
            self._spawn(self._send(ToolResult(call_id=msg.call_id, ok=False, error="unsupported")))

    async def _guarded(self, handler: Handler, msg: Message) -> None:
        try:
            await handler(msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s の処理に失敗", msg.type)

    def _spawn(self, coro: Coroutine[Any, Any, object]) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    # ---------------------------------------------------------------- ハートビート

    async def _heartbeat(self, ep: Endpoint) -> None:
        while True:
            await asyncio.sleep(self._cfg.ping_interval_s)
            nonce = uuid.uuid4().hex[:12]
            waiter = self._pongs[nonce] = asyncio.Event()
            try:
                await self._send(Ping(nonce=nonce), ep=ep)
                await asyncio.wait_for(waiter.wait(), self._cfg.pong_timeout_s)
            except TimeoutError:
                log.warning("pong が返らない。接続を切って繋ぎ直す")
                return
            finally:
                self._pongs.pop(nonce, None)

    # ---------------------------------------------------------------- 送信

    async def _send(self, msg: Message, *, ep: Endpoint | None = None) -> bool:
        ep = ep or self._ep
        if ep is None:
            return False
        try:
            async with self._send_lock:
                await ep.send(msg)
        except TransportClosed:
            return False
        return True

    async def send(self, msg: Message) -> bool:
        """メッセージを送る。繋がっていなければ送らずに False。"""
        return await self._send(msg)

    async def begin_utterance(
        self, utt_id: str, *, doa_rad: float | None = None, person_hint: str | None = None
    ) -> bool:
        self._seq[utt_id] = 0
        return await self._send(
            UtteranceBegin(utt_id=utt_id, doa_rad=doa_rad, person_hint=person_hint)
        )

    async def send_audio(self, utt_id: str, pcm_f32le: bytes, sample_rate: int = 16000) -> bool:
        """発話中のマイク音声を 1 チャンク送る（チャンク番号は ``utt_id`` ごとに自動で振る）。"""
        ep = self._ep
        if ep is None or utt_id not in self._seq:
            return False
        seq = self._seq[utt_id]
        self._seq[utt_id] = seq + 1
        header = BinaryHeader(
            kind="audio_in", ref=utt_id, seq=seq, format="pcm_f32le", sr=sample_rate
        )
        try:
            async with self._send_lock:
                await ep.send_frame(header, pcm_f32le)
        except TransportClosed:
            return False
        return True

    async def end_utterance(self, msg: UtteranceEnd) -> bool:
        self._seq.pop(msg.utt_id, None)
        return await self._send(msg)

    async def decide_floor(self, msg: FloorDecision) -> bool:
        return await self._send(msg)

    async def request_proactive(self, msg: ProactiveRequest) -> bool:
        return await self._send(msg)

    async def cancel_turn(self, turn_id: str, reason: str = "other") -> bool:
        """応答の取り消し（割り込み・退出）。そのターンの音声は、届いていても捨てる。"""
        self._cancelled.append(turn_id)
        kept = [c for c in self._drain_audio() if c.turn_id != turn_id]
        for clip in kept:
            self.audio.put_nowait(clip)
        return await self._send(TurnCancel(turn_id=turn_id, reason=reason))  # pyright: ignore[reportArgumentType]

    def _drain_audio(self) -> list[AudioClip]:
        out: list[AudioClip] = []
        while not self.audio.empty():
            out.append(self.audio.get_nowait())
        return out

    async def reply_memory_ack(self, op_id: str, result: str, reason: str | None = None) -> bool:
        return await self._send(MemoryAck(op_id=op_id, result=result, reason=reason))  # pyright: ignore[reportArgumentType]

    async def reply_tool_result(
        self, call_id: str, *, ok: bool, result: JsonValue = None, error: str | None = None
    ) -> bool:
        return await self._send(ToolResult(call_id=call_id, ok=ok, result=result, error=error))

    async def forget_person(self, person_id: str) -> bool:
        return await self._send(PersonForget(person_id=person_id))

    async def close_session(self, person_id: str) -> bool:
        return await self._send(SessionClose(person_id=person_id))


__all__ = ["AudioClip", "Connect", "LinkConfig", "NotConnected", "ServerClient"]
