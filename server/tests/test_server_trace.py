"""ターン単位のトレース: 区間時間の内訳が 1 行の JSON で読める（#27）。時計は手動で進める。"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from reachy_protocol.channel import BinaryFrame
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    FloorDecision,
    PersonProfile,
    ProactiveRequest,
    Situation,
    TurnCancel,
    TurnContext,
    UtteranceBegin,
    UtteranceEnd,
    UtteranceFeatures,
)
from reachy_server.conversation import Conversation
from reachy_server.providers.base import Transcription
from reachy_server.responder import ReplyEvent, ReplySentence, TurnRequest
from reachy_server.trace import TurnTrace

CTX = TurnContext(
    person=PersonProfile(id="p1", name="山田", familiarity="daily", visit_count=9),
    situation=Situation(people_count=1, local_time=datetime(2026, 11, 19, 9, tzinfo=UTC)),
)
FEAT = UtteranceFeatures(duration_s=1, eye_contact_ratio=1, people_count=1)
SECRET = "リーチ、山田です。住所は秘密"


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TickSTT:
    name = "tick"

    def __init__(self, clock: ManualClock, seconds: float) -> None:
        self.clock, self.seconds = clock, seconds

    async def transcribe(self, pcm_f32le: bytes, sr: int) -> Transcription:
        self.clock.advance(self.seconds)
        return Transcription(SECRET)


class TickTTS:
    name = "tick"

    def __init__(self, clock: ManualClock, seconds: float) -> None:
        self.clock, self.seconds = clock, seconds

    async def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        self.clock.advance(self.seconds)
        return b"RIFF"


class TickResponder:
    def __init__(
        self,
        clock: ManualClock,
        sentences: list[str],
        first_s: float = 0.7,
        gate: asyncio.Event | None = None,
        fail: bool = False,
    ) -> None:
        self.clock, self.sentences, self.first_s = clock, sentences, first_s
        self.gate, self.fail = gate, fail

    async def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        self.clock.advance(self.first_s)  # LLM が最初の文を出すまで
        if self.fail:
            raise RuntimeError("LLM down")
        for text in self.sentences:
            if self.gate is not None:
                await self.gate.wait()
            yield ReplySentence(text)
            await asyncio.sleep(0)  # 合成側に回す


class Rig:
    def __init__(self, *, stt_s: float = 0.2, tts_s: float = 0.15, **responder_kw: Any) -> None:
        self.clock = ManualClock()
        self.out: list[Any] = []
        sentences = responder_kw.pop("sentences", ["おはよう！"])
        self.conv = Conversation(
            send=self._send,
            send_frame=self._send_frame,
            stt=TickSTT(self.clock, stt_s),
            tts=TickTTS(self.clock, tts_s),
            responder=TickResponder(self.clock, sentences, **responder_kw),
            clock=self.clock,
        )

    async def _send(self, msg: Any) -> None:
        self.out.append(msg)

    async def _send_frame(self, header: BinaryHeader, payload: bytes) -> None:
        self.out.append(BinaryFrame(header, payload))

    async def utter_and_respond(self, decision_wait_s: float = 0.01) -> None:
        await self.conv.handle(UtteranceBegin(utt_id="u1"))
        header = BinaryHeader(kind="audio_in", ref="u1", seq=0, format="pcm_f32le", sr=16000)
        await self.conv.handle(BinaryFrame(header, struct.pack("<16f", *([0.0] * 16))))
        await self.conv.handle(UtteranceEnd(utt_id="u1", features=FEAT))
        await self.settle()  # 文字起こしまで
        self.clock.advance(decision_wait_s)  # ロボットが判定している間
        await self.conv.handle(FloorDecision(utt_id="u1", decision="respond", context=CTX))

    async def settle(self) -> None:
        async with asyncio.timeout(2):
            while self.conv._tasks:  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0.005)


# ---------------------------------------------------------------- 区間の内訳


async def test_utterance_turn_breakdown() -> None:
    rig = Rig(stt_s=0.2, tts_s=0.15, first_s=0.7)
    await rig.utter_and_respond(decision_wait_s=0.01)
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["kind"] == "utterance" and t["status"] == "ok"
    assert t["stt_ms"] == 200.0  # 発話の終わり → 文字起こし
    assert t["decision_wait_ms"] == 10.0  # ロボットの判定を待った時間
    assert t["first_sentence_ms"] == 700.0  # LLM が最初の文を出すまで
    assert t["tts_first_ms"] == 150.0  # 最初の文の合成
    assert t["first_audio_ms"] == 850.0  # ターン開始 → 最初の音声
    assert t["felt_ms"] == pytest.approx(200 + 10 + 850)  # 話し終えてから第一声まで
    assert t["tts_total_ms"] == 150.0 and t["sentences"] == 1
    assert t["turn_id"] == "t1" and t["ref"] == "u1"


async def test_each_sentence_adds_to_the_synthesis_total() -> None:
    rig = Rig(tts_s=0.1, sentences=["一つ目。", "二つ目。", "三つ目。"])
    await rig.utter_and_respond()
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["sentences"] == 3
    assert t["tts_total_ms"] == pytest.approx(300.0)
    assert t["tts_first_ms"] == pytest.approx(100.0)  # 合計ではなく最初の文の分


async def test_proactive_turn_has_no_stt_or_decision_time() -> None:
    rig = Rig(first_s=0.5, tts_s=0.1)
    await rig.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=CTX))
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["kind"] == "proactive"
    assert t["stt_ms"] is None and t["decision_wait_ms"] is None and t["felt_ms"] is None
    assert t["first_audio_ms"] == pytest.approx(600.0)


async def test_cancelled_turn_before_any_audio_is_traced() -> None:
    gate = asyncio.Event()
    rig = Rig(gate=gate)
    await rig.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=CTX))
    await asyncio.sleep(0.02)
    await rig.conv.handle(TurnCancel(turn_id="t1"))
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["status"] == "cancelled"
    assert t["first_sentence_ms"] is None and t["first_audio_ms"] is None
    assert t["sentences"] == 0


async def test_failed_turn_is_traced_with_its_status() -> None:
    rig = Rig(fail=True)
    await rig.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=CTX))
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["status"] == "error" and t["first_audio_ms"] is None


# ---------------------------------------------------------------- ログ


async def test_trace_is_one_json_line_without_personal_information(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = Rig()
    with caplog.at_level(logging.INFO, logger="reachy_server.trace"):
        await rig.utter_and_respond()
        await rig.settle()
    lines = [r.getMessage() for r in caplog.records if r.name == "reachy_server.trace"]
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "turn_trace" and data["first_audio_ms"] is not None
    # 名前・発言の内容は出さない（person_id 以外の人物情報をログに出さない: 設計書 §3.7）
    assert "山田" not in lines[0] and SECRET not in lines[0] and "住所" not in lines[0]


# ---------------------------------------------------------------- TurnTrace 単体


def test_marks_record_only_the_first_time() -> None:
    clock = ManualClock()
    trace = TurnTrace("t1", "u1", "utterance", clock=clock)
    clock.advance(0.3)
    trace.mark("first_audio")
    clock.advance(0.5)
    trace.mark("first_audio")  # 2 文目以降では更新しない
    assert trace.since_start("first_audio") == pytest.approx(0.3)
    assert trace.since_start("first_sentence") is None


# ---------------------------------------------------------------- レビュー指摘（Codex）


async def test_slow_sending_is_not_counted_as_synthesis() -> None:
    """通信が詰まって音声を送るのに時間がかかっても、TTS の時間には含めない。"""
    rig = Rig(tts_s=0.15, first_s=0.7)

    async def slow_send_frame(header: BinaryHeader, payload: bytes) -> None:
        rig.clock.advance(0.5)  # 送信待ち（バックプレッシャ）

    rig.conv._send_frame = slow_send_frame  # pyright: ignore[reportPrivateUsage]
    await rig.utter_and_respond()
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["tts_first_ms"] == 150.0  # 合成だけ
    assert t["tts_total_ms"] == 150.0
    assert t["first_audio_ms"] == pytest.approx(700 + 150 + 500)  # 第一声は送り終えるまで


async def test_turn_cancelled_before_its_task_starts_is_still_traced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """タスクが動き出す前に取消されたターンも、ログと recent_traces に 1 回だけ残る。"""
    rig = Rig()
    with caplog.at_level(logging.INFO, logger="reachy_server.trace"):
        await rig.conv.handle(
            ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=CTX)
        )
        await rig.conv.handle(TurnCancel(turn_id="t1"))  # 1 回も await で譲らずに取消
        await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["status"] == "cancelled" and t["turn_id"] == "t1"
    assert t["first_audio_ms"] is None and t["sentences"] == 0
    assert len([r for r in caplog.records if r.name == "reachy_server.trace"]) == 1


async def test_a_replaced_turn_is_traced_once_each() -> None:
    rig = Rig()
    for i in ("i1", "i2"):  # 続けて来た要求は、前のターンを閉じてから始める
        await rig.conv.handle(
            ProactiveRequest(intent_id=i, kind="greet", strategy="x", context=CTX)
        )
    await rig.settle()
    assert sorted(t["turn_id"] for t in rig.conv.recent_traces) == ["t1", "t2"]


def test_a_trace_is_emitted_only_once() -> None:
    trace = TurnTrace("t1", "u1", "utterance", clock=ManualClock())
    assert trace.emit("ok") is not None
    assert trace.emit("cancelled") is None


# ---------------------------------------------------------------- 起動の入口でログが出る


def test_the_entrypoint_configures_logging_so_traces_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``python -m reachy_server`` で、トレース（INFO）が捨てられない。"""
    from reachy_server import __main__ as entry

    root = logging.getLogger()
    saved = (root.level, list(root.handlers))
    monkeypatch.setenv("REACHY_SERVER_TOKEN", "x" * 16)
    monkeypatch.setattr(entry, "create_app", lambda _settings: object())
    monkeypatch.setattr(entry.uvicorn, "run", lambda *_a, **_k: None)
    try:
        root.setLevel(logging.WARNING)  # 何も設定していない状態（既定は WARNING）
        entry.main()
        assert logging.getLogger("reachy_server.trace").isEnabledFor(logging.INFO)
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)  # 外部はうるさいので静かに
    finally:
        root.handlers[:] = saved[1]
        root.setLevel(saved[0])


def test_log_level_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_server.config import ServerSettings

    monkeypatch.setenv("REACHY_SERVER_TOKEN", "x" * 16)
    monkeypatch.setenv("REACHY_SERVER_LOG_LEVEL", "debug")
    assert ServerSettings().log_level == "debug"  # pyright: ignore[reportCallIssue]
    monkeypatch.setenv("REACHY_SERVER_LOG_LEVEL", "loud")
    with pytest.raises(ValueError):
        ServerSettings()  # pyright: ignore[reportCallIssue]


async def test_felt_latency_includes_time_not_in_any_section() -> None:
    """文字起こしの送信待ちなど、どの区間にも入らない時間も、体感の遅れに入る。"""
    rig = Rig(stt_s=0.2, tts_s=0.15, first_s=0.7)
    send = rig.conv._send  # pyright: ignore[reportPrivateUsage]

    async def slow_send(msg: Any) -> None:
        if type(msg).__name__ == "Transcript":
            rig.clock.advance(0.5)  # 送信ロック・バックプレッシャで待たされた
        await send(msg)

    rig.conv._send = slow_send  # pyright: ignore[reportPrivateUsage]
    await rig.utter_and_respond(decision_wait_s=0.01)
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["stt_ms"] == 200.0
    # 判定待ちは「文字起こしを送り始めてから」なので、送信待ちの 0.5 s も含む（どこかの区間に入る）
    assert t["decision_wait_ms"] == pytest.approx(500 + 10)
    assert t["felt_ms"] == pytest.approx(200 + 500 + 10 + 850)  # 発話の終わり → 第一声の実際の差


async def test_a_turn_interrupted_by_a_closed_connection_is_still_traced() -> None:
    """ロボットが切断して turn.end を送れなくても、トレースが残る。"""
    from reachy_protocol.transport import TransportClosed

    gate = asyncio.Event()
    rig = Rig(gate=gate)
    send = rig.conv._send  # pyright: ignore[reportPrivateUsage]
    closed = False

    async def maybe_closed(msg: Any) -> None:
        if closed:
            raise TransportClosed("closed")
        await send(msg)

    rig.conv._send = maybe_closed  # pyright: ignore[reportPrivateUsage]
    await rig.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=CTX))
    await asyncio.sleep(0.02)
    closed = True  # 接続が閉じた
    await rig.conv.close()
    (t,) = rig.conv.recent_traces
    assert t["status"] == "cancelled" and t["turn_id"] == "t1"


async def test_a_decision_that_arrives_before_the_transcript_send_returns() -> None:
    """文字起こしの送信が終わる前にロボットの判定が届いても、判定待ちの時間が正しい。"""
    rig = Rig(stt_s=0.2, tts_s=0.15, first_s=0.7)
    send = rig.conv._send  # pyright: ignore[reportPrivateUsage]

    async def fast_robot(msg: Any) -> None:
        await send(msg)
        if type(msg).__name__ == "Transcript":
            rig.clock.advance(0.3)  # ロボットが 0.3 s で判定し、送信の完了より先に届く
            await rig.conv.handle(FloorDecision(utt_id="u1", decision="respond", context=CTX))

    rig.conv._send = fast_robot  # pyright: ignore[reportPrivateUsage]
    await rig.conv.handle(UtteranceBegin(utt_id="u1"))
    header = BinaryHeader(kind="audio_in", ref="u1", seq=0, format="pcm_f32le", sr=16000)
    await rig.conv.handle(BinaryFrame(header, struct.pack("<16f", *([0.0] * 16))))
    await rig.conv.handle(UtteranceEnd(utt_id="u1", features=FEAT))
    await rig.settle()
    (t,) = rig.conv.recent_traces
    assert t["decision_wait_ms"] == pytest.approx(300.0)  # 起点が 0 のままなら 10 万 ms になる
    assert t["felt_ms"] == pytest.approx(200 + 300 + 850)
