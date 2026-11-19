"""ハンドシェイク後の 1 接続ぶんの状態機械（発話 → 文字起こし → 判定待ち → ターン）。設計書 §2.2。

送信関数を受け取るだけなので、通信路なしで単体テストできる。
規約（:mod:`reachy_protocol.conformance`）を守るため、失敗は必ず ref 付きのエラーか
``turn.end(error)`` で相手に知らせる。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Literal

from reachy_protocol.channel import BinaryFrame, Incoming
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    Error,
    FloorDecision,
    Message,
    Motion,
    ProactiveRequest,
    Sentence,
    Transcript,
    TurnCancel,
    TurnEnd,
    TurnStart,
    UtteranceBegin,
    UtteranceEnd,
)

from .providers import STTProvider, TTSProvider
from .responder import ReplyMotion, ReplySentence, Responder, TurnRequest

log = logging.getLogger(__name__)

SendMessage = Callable[[Message], Awaitable[None]]
SendFrame = Callable[[BinaryHeader, bytes], Awaitable[None]]

UttState = Literal["receiving", "transcribing", "waiting_decision"]


@dataclass
class _Utterance:
    state: UttState = "receiving"
    sr: int | None = None
    next_seq: int = 0
    audio: bytearray = field(default_factory=bytearray)
    text: str = ""
    expiry: asyncio.TimerHandle | None = None


@dataclass
class _Turn:
    turn_id: str
    ref: str
    task: asyncio.Task[None]


@dataclass(frozen=True)
class ConversationLimits:
    decision_timeout_s: float = 15.0
    """文字起こしを送ってから floor.decision を待つ時間。過ぎたら発話を破棄する。"""
    max_utterance_s: float = 30.0
    """1 発話の音声の上限（超えたら受信をやめてエラー）。"""


class Conversation:
    def __init__(
        self,
        *,
        send: SendMessage,
        send_frame: SendFrame,
        stt: STTProvider,
        tts: TTSProvider,
        responder: Responder,
        limits: ConversationLimits | None = None,
    ) -> None:
        self._send = send
        self._send_frame = send_frame
        self.stt = stt
        self.tts = tts
        self.responder = responder
        self.limits = limits or ConversationLimits()
        self._utts: dict[str, _Utterance] = {}
        self._turn: _Turn | None = None
        self._turn_seq = 0
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------ 入口

    async def handle(self, item: Incoming) -> bool:
        """1 件処理する。この状態機械の担当でなければ False を返す。"""
        if isinstance(item, BinaryFrame):
            if item.header.kind == "audio_in":
                await self._on_audio(item.header, item.payload)
            # frame（VLM 用の画像）は #68 で扱う。今は受け取って捨てる
            return True
        if isinstance(item, UtteranceBegin):
            await self._on_begin(item)
        elif isinstance(item, UtteranceEnd):
            await self._on_end(item)
        elif isinstance(item, FloorDecision):
            await self._on_decision(item)
        elif isinstance(item, ProactiveRequest):
            await self._start_turn(
                TurnRequest(
                    ref=item.intent_id,
                    context=item.context,
                    proactive_kind=item.kind,
                    strategy=item.strategy,
                )
            )
        elif isinstance(item, TurnCancel):
            await self._on_cancel(item)
        else:
            return False
        return True

    async def close(self) -> None:
        """接続の終了。進行中の処理をすべて止める（相手はもういないので何も送らない）。"""
        for utt in self._utts.values():
            if utt.expiry is not None:
                utt.expiry.cancel()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # ------------------------------------------------------------ 発話

    async def _error(self, code: str, message: str, ref: str | None) -> None:
        await self._send(Error.model_validate({"code": code, "message": message, "ref": ref}))

    async def _on_begin(self, msg: UtteranceBegin) -> None:
        if msg.utt_id in self._utts:
            await self._error("bad_message", "utt_id が重複している", msg.utt_id)
            return
        self._utts[msg.utt_id] = _Utterance()

    async def _on_audio(self, header: BinaryHeader, payload: bytes) -> None:
        utt = self._utts.get(header.ref)
        if utt is None or utt.state != "receiving":
            await self._error("unknown_ref", "開いていない発話への音声", header.ref)
            return
        if header.seq != utt.next_seq:
            await self._error(
                "bad_message", f"音声の seq が {utt.next_seq} でなく {header.seq}", header.ref
            )
            self._drop(header.ref)
            return
        sr = header.sr or 16000
        if utt.sr not in (None, sr):
            await self._error("bad_message", "発話の途中で sr が変わった", header.ref)
            self._drop(header.ref)
            return
        utt.sr = sr
        utt.next_seq += 1
        utt.audio.extend(payload)
        if len(utt.audio) / 4 / sr > self.limits.max_utterance_s:
            await self._error(
                "bad_message", f"発話が {self.limits.max_utterance_s} 秒を超えた", header.ref
            )
            self._drop(header.ref)

    async def _on_end(self, msg: UtteranceEnd) -> None:
        utt = self._utts.get(msg.utt_id)
        if utt is None or utt.state != "receiving":
            await self._error("unknown_ref", "開いていない発話の終了", msg.utt_id)
            return
        utt.state = "transcribing"
        self._spawn(self._transcribe(msg.utt_id, utt))

    async def _transcribe(self, utt_id: str, utt: _Utterance) -> None:
        try:
            result = await self.stt.transcribe(bytes(utt.audio), utt.sr or 16000)
        except Exception:
            log.exception("STT failed utt=%s", utt_id)
            self._drop(utt_id)
            await self._error("internal", "文字起こしに失敗した", utt_id)
            return
        utt.audio = bytearray()  # 判定待ちの間は音声を持たない
        utt.text = result.text
        utt.state = "waiting_decision"
        loop = asyncio.get_running_loop()
        utt.expiry = loop.call_later(self.limits.decision_timeout_s, self._expire, utt_id)
        await self._send(
            Transcript(
                utt_id=utt_id, text=result.text, lang=result.lang, confidence=result.confidence
            )
        )

    def _expire(self, utt_id: str) -> None:
        log.info("decision timeout utt=%s", utt_id)
        self._utts.pop(utt_id, None)

    def _drop(self, utt_id: str) -> None:
        utt = self._utts.pop(utt_id, None)
        if utt is not None and utt.expiry is not None:
            utt.expiry.cancel()

    async def _on_decision(self, msg: FloorDecision) -> None:
        utt = self._utts.get(msg.utt_id)
        if utt is None or utt.state != "waiting_decision":
            await self._error(
                "unknown_ref", "判定待ちでない発話への判定（期限切れ・未知）", msg.utt_id
            )
            return
        self._drop(msg.utt_id)
        if msg.decision != "respond" or msg.context is None:
            # ignore / backchannel: 割り込まない。場の文脈として残すのは #21
            return
        await self._start_turn(TurnRequest(ref=msg.utt_id, context=msg.context, user_text=utt.text))

    # ------------------------------------------------------------ ターン

    async def _start_turn(self, request: TurnRequest) -> None:
        # 同時に 2 つは話さない。進行中のターンは取消してから始める
        if self._turn is not None and not self._turn.task.done():
            self._turn.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn.task
        self._turn_seq += 1
        turn_id = f"t{self._turn_seq}"
        task = self._spawn(self._run_turn(turn_id, request))
        self._turn = _Turn(turn_id=turn_id, ref=request.ref, task=task)

    async def _on_cancel(self, msg: TurnCancel) -> None:
        turn = self._turn
        if turn is not None and turn.turn_id == msg.turn_id and not turn.task.done():
            turn.task.cancel()
        # 終了済み・未知のターンへの取消は行き違いとして黙って受け流す

    async def _run_turn(self, turn_id: str, request: TurnRequest) -> None:
        """生成と合成をパイプラインで流す。取消されても必ず turn.end を送る。"""
        await self._send(TurnStart(turn_id=turn_id, ref=request.ref))
        status: Literal["ok", "cancelled", "error"] = "ok"
        error: str | None = None
        queue: asyncio.Queue[ReplySentence | None] = asyncio.Queue()
        producer = asyncio.create_task(self._produce(turn_id, request, queue))
        try:
            idx = 0
            while (item := await queue.get()) is not None:
                await self._send(
                    Sentence(turn_id=turn_id, idx=idx, text=item.text, emotion=item.emotion)
                )
                wav = await self.tts.synthesize(item.text, item.emotion)
                header = BinaryHeader(kind="audio_out", ref=turn_id, seq=idx, format="wav_pcm16")
                await self._send_frame(header, wav)
                idx += 1
            await producer  # 生成側の例外をここで受ける
        except asyncio.CancelledError:
            status = "cancelled"
        except Exception as e:
            log.exception("turn failed turn=%s", turn_id)
            status, error = "error", type(e).__name__
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer
        await self._send(TurnEnd(turn_id=turn_id, status=status, error=error))

    async def _produce(
        self, turn_id: str, request: TurnRequest, queue: asyncio.Queue[ReplySentence | None]
    ) -> None:
        try:
            async for event in self.responder.respond(request):
                if isinstance(event, ReplyMotion):
                    await self._send(Motion(turn_id=turn_id, name=event.name))
                else:
                    await queue.put(event)
        finally:
            await queue.put(None)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
