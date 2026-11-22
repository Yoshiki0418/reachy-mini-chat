"""ハンドシェイク後の 1 接続ぶんの状態機械（発話 → 文字起こし → 判定待ち → ターン）。設計書 §2.2。

送信関数を受け取るだけなので、通信路なしで単体テストできる。
規約（:mod:`reachy_protocol.conformance`）を守るため、失敗は必ず ref 付きのエラーか
``turn.end(error)`` で相手に知らせる。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import JsonValue

from reachy_protocol.channel import BinaryFrame, Incoming
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    Error,
    FloorDecision,
    MemoryAck,
    MemoryOp,
    Message,
    Motion,
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
)

from .history import person_thread
from .overheard import OverheardBuffer
from .providers import STTProvider, TTSProvider
from .responder import (
    MemoryOpName,
    MemoryResult,
    ReplyMotion,
    ReplySentence,
    Responder,
    ToolOutcome,
    TurnRequest,
)
from .trace import Clock, TurnTrace

log = logging.getLogger(__name__)

SendMessage = Callable[[Message], Awaitable[None]]
SendFrame = Callable[[BinaryHeader, bytes], Awaitable[None]]

UttState = Literal["receiving", "transcribing", "waiting_decision"]


ForgetHistory = Callable[[str, str], Awaitable[object]]
"""(robot_id, person_id) の会話履歴をすべて消す。"""

SummarizeThread = Callable[[str], Awaitable[str | None]]
"""会話スレッドを要約する。話題がなければ None。"""


@dataclass
class _Utterance:
    state: UttState = "receiving"
    sr: int | None = None
    next_seq: int = 0
    audio: bytearray = field(default_factory=bytearray)
    text: str = ""
    expiry: asyncio.TimerHandle | None = None
    t_end: float = 0.0
    """utterance.end を受けた時刻。"""
    stt_s: float | None = None
    t_transcript: float = 0.0
    """transcript を送り始めた時刻（判定待ちの起点）。"""


@dataclass
class _Turn:
    turn_id: str
    ref: str
    person_id: str | None = None
    """このターンの相手。忘れられたとき、ターンが終わってからもう一度履歴を消すため。"""
    task: asyncio.Task[None] | None = None
    trace: TurnTrace | None = None
    ended: bool = False
    """turn.end を送ったか。止めたタスクが送らずに終わったら、代わりに送る。"""


@dataclass(frozen=True)
class ConversationLimits:
    decision_timeout_s: float = 15.0
    """文字起こしを送ってから floor.decision を待つ時間。過ぎたら発話を破棄する。"""
    max_utterance_s: float = 30.0
    """1 発話の音声の上限（超えたら受信をやめてエラー）。"""
    memory_ack_timeout_s: float = 3.0
    """memory.op を送ってから memory.ack を待つ時間。"""
    tool_timeout_s: float = 1.0
    """tool.call を送ってから tool.result を待つ時間（設計書 §4.2）。"""
    summary_timeout_s: float = 30.0
    """退出した人の会話の要約を作る時間の上限。"""


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
        robot_id: str = "robot",
        overheard: OverheardBuffer | None = None,
        forget_history: ForgetHistory | None = None,
        summarize: SummarizeThread | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.robot_id = robot_id
        self._forget_history = forget_history
        self._summarize = summarize
        self._summaries: dict[str, asyncio.Task[None]] = {}
        """要約中のタスク（人ごと）。二重に走らせず、忘れられたら取り消す。"""
        self._clock = clock
        self.recent_traces: deque[dict[str, Any]] = deque(maxlen=100)
        """直近のターンのトレース（reachy_server.trace のログと同じ内容）。"""
        self._last_day: dict[str, date] = {}
        """この接続で最後に話した日（人ごと）。退出のとき、どの日のスレッドを要約するか。"""
        self.overheard = overheard or OverheardBuffer(ttl_s=0)
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
        self._forgotten_during_turn: set[str] = set()
        self._op_seq = 0
        self._pending_ops: dict[str, asyncio.Future[MemoryAck]] = {}
        self._call_seq = 0
        self._pending_calls: dict[str, asyncio.Future[ToolResult]] = {}

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
                    **self._common(),
                )
            )
        elif isinstance(item, TurnCancel):
            await self._on_cancel(item)
        elif isinstance(item, MemoryAck):
            self._on_memory_ack(item)
        elif isinstance(item, ToolResult):
            self._on_tool_result(item)
        elif isinstance(item, SessionClose):
            # 要約は LLM を呼ぶので時間がかかる。受信ループを止めない（memory.ack も同じ口で届く）
            self._start_summary(item.person_id)
        else:
            return False
        return True

    def _common(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "overheard": self.overheard.recent(),
            "memory": self._memory_op,
            "robot_tool": self._robot_tool,
        }

    # ------------------------------------------------------------ 記憶操作（設計書 §2.4）

    async def _memory_op(self, person_id: str, op: MemoryOpName, value: str | None) -> MemoryResult:
        """memory.op を送り、ロボットの検証結果（memory.ack）を待つ。

        人物の記憶の正はロボット。サーバーは操作を発行するだけで、結果が applied になるまで
        「覚えた」とは扱わない（時間内に返事がなければ timeout）。
        """
        self._op_seq += 1
        op_id = f"m{self._op_seq}"
        waiter: asyncio.Future[MemoryAck] = asyncio.get_running_loop().create_future()
        self._pending_ops[op_id] = waiter
        try:
            await self._send(MemoryOp(op_id=op_id, person_id=person_id, op=op, value=value))
            ack = await asyncio.wait_for(waiter, self.limits.memory_ack_timeout_s)
        except TimeoutError:
            log.warning("memory.ack timeout op=%s", op_id)
            return MemoryResult("timeout")
        finally:
            self._pending_ops.pop(op_id, None)
        if op == "forget" and ack.result == "applied":
            await self.forget_person(person_id)
        return MemoryResult(ack.result, ack.reason)

    async def forget_person(self, person_id: str) -> None:
        """その人の会話履歴をすべて消す（person.forget、または forget の memory.ack を受けたとき）。

        進行中のターンが同じ人のものなら、そのターンが終わってからもう一度消す。ターンの途中で
        消しても、実行中のグラフが後から履歴を書き戻してしまうため（ターンが終われば書き込みは
        止まっている）。
        """
        turn = self._turn
        if turn is not None and not turn.ended and turn.person_id == person_id:
            self._forgotten_during_turn.add(person_id)
        # 要約の途中なら止める。そのまま終わると、忘れた会話の要約を add_episode で送ってしまう
        summary = self._summaries.pop(person_id, None)
        if summary is not None and not summary.done():
            summary.cancel()
        await self._delete_history(person_id)

    async def _delete_history(self, person_id: str) -> None:
        if self._forget_history is not None:
            await self._forget_history(self.robot_id, person_id)

    async def _after_turn(self, turn: _Turn) -> None:
        """ターンの生成が完全に止まった後の後始末。ターン中に忘れられた人の履歴を消し直す。"""
        person_id = turn.person_id
        if person_id is not None and person_id in self._forgotten_during_turn:
            self._forgotten_during_turn.discard(person_id)
            await self._delete_history(person_id)

    # ------------------------------------------------------------ 退出と要約（設計書 §2.2）

    def _start_summary(self, person_id: str) -> None:
        running = self._summaries.get(person_id)
        if running is not None and not running.done():
            log.debug("summary already running person=%s", person_id)  # 退出の合図が重なった
            return
        task = self._spawn(self._summarize_session(person_id))
        self._summaries[person_id] = task

        def forget_task(t: asyncio.Task[None]) -> None:
            if self._summaries.get(person_id) is t:
                del self._summaries[person_id]

        task.add_done_callback(forget_task)

    async def _summarize_session(self, person_id: str) -> None:
        """相手の退出で、その日の会話を要約し、memory.op(add_episode) でロボットに保存させる。"""
        day = self._last_day.get(person_id)
        if self._summarize is None or day is None:
            return  # この接続では話していない（要約する会話がない）
        turn = self._turn
        if turn is not None and turn.person_id == person_id and turn.task is not None:
            await asyncio.wait([turn.task])  # 返答の途中なら、終わってから（履歴が出そろってから）
        try:
            summary = await asyncio.wait_for(
                self._summarize(person_thread(self.robot_id, person_id, day)),
                self.limits.summary_timeout_s,
            )
        except Exception:
            log.exception("summary failed person=%s", person_id)
            return
        if summary:
            result = await self._memory_op(person_id, "add_episode", summary)
            if result.result != "applied":
                log.info("episode not saved person=%s result=%s", person_id, result)

    # ------------------------------------------------------------ ロボット側ツール（設計書 §4.2）

    async def _robot_tool(self, name: str, args: dict[str, JsonValue]) -> ToolOutcome:
        """tool.call を送り、ロボットの tool.result を待つ。データはロボットにあるので、RPC で引く。"""
        self._call_seq += 1
        call_id = f"c{self._call_seq}"
        waiter: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = waiter
        try:
            await self._send(ToolCall(call_id=call_id, name=name, args=args))
            res = await asyncio.wait_for(waiter, self.limits.tool_timeout_s)
        except TimeoutError:
            log.warning("tool.result timeout call=%s name=%s", call_id, name)
            return ToolOutcome("timeout")
        finally:
            self._pending_calls.pop(call_id, None)
        if not res.ok:
            return ToolOutcome("error", error=res.error)
        return ToolOutcome("ok", res.result)

    def _on_tool_result(self, res: ToolResult) -> None:
        waiter = self._pending_calls.get(res.call_id)
        if waiter is None or waiter.done():
            log.debug("tool.result for unknown or finished call=%s", res.call_id)  # 行き違い
            return
        waiter.set_result(res)

    def _on_memory_ack(self, ack: MemoryAck) -> None:
        waiter = self._pending_ops.get(ack.op_id)
        if waiter is None or waiter.done():
            log.debug("memory.ack for unknown or finished op=%s", ack.op_id)  # 行き違い
            return
        waiter.set_result(ack)

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
        utt.t_end = self._clock()
        self._spawn(self._transcribe(msg.utt_id, utt))

    async def _transcribe(self, utt_id: str, utt: _Utterance) -> None:
        try:
            result = await self.stt.transcribe(bytes(utt.audio), utt.sr or 16000)
        except Exception:
            log.exception("STT failed utt=%s", utt_id)
            self._drop(utt_id)
            await self._error("internal", "文字起こしに失敗した", utt_id)
            return
        utt.stt_s = self._clock() - utt.t_end
        utt.audio = bytearray()  # 判定待ちの間は音声を持たない
        utt.text = result.text
        # 判定を受け付ける状態にする前に、起点の時刻を決める（送信が終わる前に判定が届いても、
        # 判定待ちの時間が正しく出るように）
        utt.t_transcript = self._clock()
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
            # ignore / backchannel: 割り込まない。場の文脈としてだけ短時間覚えておく
            self.overheard.add(utt.text)
            return
        await self._start_turn(
            TurnRequest(ref=msg.utt_id, context=msg.context, user_text=utt.text, **self._common()),
            stt_s=utt.stt_s,
            decision_wait_s=self._clock() - utt.t_transcript,
            t_speech_end=utt.t_end,
        )

    # ------------------------------------------------------------ ターン

    async def _start_turn(
        self,
        request: TurnRequest,
        *,
        stt_s: float | None = None,
        decision_wait_s: float | None = None,
        t_speech_end: float | None = None,
    ) -> None:
        # 同時に 2 つは話さない。進行中のターンは閉じてから始める
        if self._turn is not None:
            await self._stop_turn(self._turn)
        self._turn_seq += 1
        person = request.context.person
        if person is not None:
            self._last_day[person.id] = request.context.situation.local_time.date()
        turn_id = f"t{self._turn_seq}"
        trace = TurnTrace(
            turn_id,
            request.ref,
            "utterance" if request.user_text is not None else "proactive",
            clock=self._clock,
        )
        trace.stt_s, trace.decision_wait_s = stt_s, decision_wait_s
        trace.t_speech_end = t_speech_end
        turn = _Turn(
            turn_id=turn_id, ref=request.ref, person_id=person.id if person else None, trace=trace
        )
        self._turn = turn
        # turn.start はタスクを作る前に送る。タスクが動き出す前に取消されても、
        # 「始まったターンは必ず turn.end で閉じる」を _stop_turn が守れるようにするため
        await self._send(TurnStart(turn_id=turn.turn_id, ref=request.ref))
        turn.task = self._spawn(self._run_turn(turn, request))

    async def _stop_turn(self, turn: _Turn) -> None:
        """ターンを止め、まだなら turn.end(cancelled) を送る。"""
        if turn.task is not None and not turn.task.done():
            turn.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await turn.task
        if not turn.ended:
            await self._send(TurnEnd(turn_id=turn.turn_id, status="cancelled"))
            turn.ended = True
            # タスクが動き出す前に取消された（_run_turn が最後まで走らなかった）ターンも記録する
            self._finish_trace(turn, "cancelled")

    def _finish_trace(self, turn: _Turn, status: str) -> None:
        if turn.trace is not None and (summary := turn.trace.emit(status)) is not None:
            self.recent_traces.append(summary)

    async def _on_cancel(self, msg: TurnCancel) -> None:
        turn = self._turn
        if turn is not None and turn.turn_id == msg.turn_id and not turn.ended:
            await self._stop_turn(turn)
        # 終了済み・未知のターンへの取消は行き違いとして黙って受け流す

    async def _run_turn(self, turn: _Turn, request: TurnRequest) -> None:
        """生成と合成をパイプラインで流す。取消されても turn.end を送る。"""
        turn_id = turn.turn_id
        status: Literal["ok", "cancelled", "error"] = "ok"
        error: str | None = None
        queue: asyncio.Queue[ReplySentence | None] = asyncio.Queue()
        producer = asyncio.create_task(self._produce(turn, request, queue))
        try:
            idx = 0
            while (item := await queue.get()) is not None:
                await self._send(
                    Sentence(turn_id=turn_id, idx=idx, text=item.text, emotion=item.emotion)
                )
                t_tts = self._clock()
                wav = await self.tts.synthesize(item.text, item.emotion)
                if turn.trace is not None:
                    # 合成の時間は、送る前に確定する（通信が詰まっても TTS のせいにしない）
                    turn.trace.synthesized(self._clock() - t_tts)
                header = BinaryHeader(kind="audio_out", ref=turn_id, seq=idx, format="wav_pcm16")
                await self._send_frame(header, wav)
                if turn.trace is not None:
                    turn.trace.mark("first_audio")
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
            # 生成はここで完全に止まった。取消で割り込まれても後始末は最後までやる
            await asyncio.shield(self._after_turn(turn))
        try:
            await self._send(TurnEnd(turn_id=turn_id, status=status, error=error))
            turn.ended = True
        finally:
            # 接続が閉じて turn.end を送れなくても、トレースは残す
            self._finish_trace(turn, status)

    async def _produce(
        self, turn: _Turn, request: TurnRequest, queue: asyncio.Queue[ReplySentence | None]
    ) -> None:
        try:
            async for event in self.responder.respond(request):
                if isinstance(event, ReplyMotion):
                    await self._send(Motion(turn_id=turn.turn_id, name=event.name))
                else:
                    if turn.trace is not None:
                        turn.trace.mark("first_sentence")
                    await queue.put(event)
        finally:
            await queue.put(None)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
