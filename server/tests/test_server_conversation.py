"""Conversation（発話 → 文字起こし → 判定待ち → ターン）の状態遷移。通信路なしで検証する。"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from reachy_protocol.channel import BinaryFrame
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    Error,
    FloorDecision,
    Motion,
    PersonProfile,
    ProactiveRequest,
    Sentence,
    Situation,
    Transcript,
    TurnCancel,
    TurnContext,
    TurnEnd,
    TurnStart,
    UtteranceBegin,
    UtteranceEnd,
    UtteranceFeatures,
)
from reachy_server.conversation import Conversation, ConversationLimits
from reachy_server.providers import FakeSTT, FakeTTS
from reachy_server.responder import ReplyEvent, ReplyMotion, ReplySentence, TurnRequest

CTX = TurnContext(
    person=PersonProfile(id="p1", name="山田", familiarity="daily", visit_count=9),
    situation=Situation(people_count=1, local_time=datetime(2026, 11, 19, 9, tzinfo=UTC)),
)
FEAT = UtteranceFeatures(duration_s=1, eye_contact_ratio=1, people_count=1)


class ScriptedResponder:
    """台本どおりの応答。各イベントの前に待ったり、途中で例外を出したりできる。"""

    def __init__(
        self,
        events: Sequence[ReplyEvent] = (
            ReplySentence("おはよう！", "happy"),
            ReplySentence("今日も早いね。"),
        ),
        delay_s: float = 0.0,
        fail_after: int | None = None,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.delay_s = delay_s
        self.fail_after = fail_after
        self.gate = gate
        self.requests: list[TurnRequest] = []
        self.closed = False

    async def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        self.requests.append(request)
        try:
            for i, event in enumerate(self.events):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("LLM down")
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                if i == 1 and self.gate is not None:
                    await self.gate.wait()
                yield event
        finally:
            self.closed = True


class Harness:
    def __init__(self, **kwargs: Any) -> None:
        self.out: list[Any] = []
        self.stt: FakeSTT = kwargs.pop("stt", FakeSTT(script=["リーチ、おはよう"]))
        self.tts: FakeTTS = kwargs.pop("tts", FakeTTS())
        self.responder: ScriptedResponder = kwargs.pop("responder", ScriptedResponder())
        self.conv = Conversation(
            send=self._send,
            send_frame=self._send_frame,
            stt=self.stt,
            tts=self.tts,
            responder=self.responder,
            limits=kwargs.pop("limits", ConversationLimits()),
        )

    async def _send(self, msg: Any) -> None:
        self.out.append(msg)

    async def _send_frame(self, header: BinaryHeader, payload: bytes) -> None:
        self.out.append(BinaryFrame(header, payload))

    async def utter(self, utt_id: str = "u1", chunks: int = 2, samples: int = 1600) -> None:
        await self.conv.handle(UtteranceBegin(utt_id=utt_id))
        for seq in range(chunks):
            await self.conv.handle(audio(utt_id, seq, samples))
        await self.conv.handle(UtteranceEnd(utt_id=utt_id, features=FEAT))

    async def settle(self, timeout: float = 1.0) -> None:
        """裏のタスク（文字起こし・ターン）が終わるのを待つ。"""
        async with asyncio.timeout(timeout):
            while self.conv._tasks:  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0.005)

    def kinds(self) -> list[str]:
        return [f"audio{o.header.seq}" if isinstance(o, BinaryFrame) else o.type for o in self.out]

    def of(self, cls: type) -> list[Any]:
        return [o for o in self.out if isinstance(o, cls)]


def audio(utt: str, seq: int, samples: int = 1600, sr: int = 16000) -> BinaryFrame:
    header = BinaryHeader(kind="audio_in", ref=utt, seq=seq, format="pcm_f32le", sr=sr)
    return BinaryFrame(header, struct.pack(f"<{samples}f", *([0.0] * samples)))


def respond(utt: str = "u1") -> FloorDecision:
    return FloorDecision(utt_id=utt, decision="respond", context=CTX)


# ---------------------------------------------------------------- 正常系


async def test_full_turn() -> None:
    h = Harness()
    await h.utter(chunks=3, samples=1600)
    await h.settle()
    [t] = h.of(Transcript)
    assert (t.utt_id, t.text) == ("u1", "リーチ、おはよう")
    assert h.stt.calls == [pytest.approx(0.3)]  # 3 チャンクがつながって STT に渡る

    await h.conv.handle(respond())
    await h.settle()
    assert h.kinds() == [
        "transcript",
        "turn.start",
        "sentence",
        "audio0",
        "sentence",
        "audio1",
        "turn.end",
    ]
    assert h.of(TurnEnd)[0].status == "ok"
    assert [s.emotion for s in h.of(Sentence)] == ["happy", "neutral"]
    assert h.tts.calls == [("おはよう！", "happy"), ("今日も早いね。", "neutral")]
    req = h.responder.requests[0]
    assert (req.ref, req.user_text, req.context) == ("u1", "リーチ、おはよう", CTX)


@pytest.mark.parametrize("decision", ["ignore", "backchannel"])
async def test_non_respond_decision_does_not_speak(decision: str) -> None:
    h = Harness()
    await h.utter()
    await h.settle()
    await h.conv.handle(FloorDecision.model_validate({"utt_id": "u1", "decision": decision}))
    await h.settle()
    assert h.kinds() == ["transcript"]
    assert h.responder.requests == []
    # 判定は 1 回きり。同じ発話への 2 回目は unknown_ref
    await h.conv.handle(respond())
    assert h.of(Error)[-1].code == "unknown_ref"


async def test_proactive_turn() -> None:
    h = Harness()
    await h.conv.handle(
        ProactiveRequest(intent_id="i1", kind="greet", strategy="daily", context=CTX)
    )
    await h.settle()
    assert h.of(TurnStart)[0].ref == "i1"
    req = h.responder.requests[0]
    assert (req.user_text, req.proactive_kind, req.strategy) == (None, "greet", "daily")


async def test_motion_is_forwarded_within_turn() -> None:
    h = Harness(responder=ScriptedResponder([ReplyMotion("nod"), ReplySentence("うん。")]))
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="engage", context=CTX))
    await h.settle()
    assert h.kinds() == ["turn.start", "motion", "sentence", "audio0", "turn.end"]
    assert h.of(Motion)[0].turn_id == "t1"


async def test_first_sentence_is_sent_before_generation_finishes() -> None:
    """パイプライン化: 2 文目の生成を待たずに 1 文目の音声を送る（最初の声までを短く）。"""
    gate = asyncio.Event()
    h = Harness(responder=ScriptedResponder(gate=gate))
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    async with asyncio.timeout(1.0):  # 1 文目の音声が出るまで待つ（固定時間で待たない）
        while "audio0" not in h.kinds():
            await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)
    assert h.kinds() == ["turn.start", "sentence", "audio0"]  # 2 文目はまだ生成中
    gate.set()
    await h.settle()
    assert h.kinds()[-1] == "turn.end"


# ---------------------------------------------------------------- 取消と割り込み


async def test_cancel_stops_generation_and_synthesis() -> None:
    events = [ReplySentence(f"{i}番目。") for i in range(10)]
    h = Harness(responder=ScriptedResponder(events, delay_s=0.02), tts=FakeTTS(delay_s=0.02))
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await asyncio.sleep(0.07)
    await h.conv.handle(TurnCancel(turn_id="t1", reason="barge_in"))
    await h.settle()
    [end] = h.of(TurnEnd)
    assert end.status == "cancelled"
    assert len(h.of(Sentence)) < 10
    assert h.responder.closed  # 生成も止まっている
    sent_after_end = h.out[h.out.index(end) + 1 :]
    assert sent_after_end == []


async def test_cancel_of_finished_or_unknown_turn_is_ignored() -> None:
    h = Harness()
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await h.settle()
    before = list(h.out)
    await h.conv.handle(TurnCancel(turn_id="t1"))
    await h.conv.handle(TurnCancel(turn_id="t99"))
    assert h.out == before  # 行き違いなので何も送らない


async def test_new_turn_cancels_running_turn_first() -> None:
    events = [ReplySentence(f"{i}番目。") for i in range(10)]
    h = Harness(responder=ScriptedResponder(events, delay_s=0.02))
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await asyncio.sleep(0.05)
    await h.conv.handle(ProactiveRequest(intent_id="i2", kind="engage", context=CTX))
    await h.settle(timeout=2.0)
    starts_and_ends = [(o.type, o.turn_id) for o in h.out if isinstance(o, TurnStart | TurnEnd)]
    assert starts_and_ends == [
        ("turn.start", "t1"),
        ("turn.end", "t1"),
        ("turn.start", "t2"),
        ("turn.end", "t2"),
    ]
    assert [e.status for e in h.of(TurnEnd)] == ["cancelled", "ok"]


# ---------------------------------------------------------------- 時間切れと失敗


async def test_decision_timeout_discards_utterance() -> None:
    h = Harness(limits=ConversationLimits(decision_timeout_s=0.05))
    await h.utter()
    await h.settle()
    await asyncio.sleep(0.1)
    await h.conv.handle(respond())
    assert h.of(Error)[-1].code == "unknown_ref"
    assert h.of(TurnStart) == []


async def test_stt_failure_reports_error_with_ref() -> None:
    def boom(_audio: bytes, _sr: int) -> str:
        raise RuntimeError("GPU OOM")

    h = Harness(stt=FakeSTT(script=boom))
    await h.utter()
    await h.settle()
    [err] = h.of(Error)
    assert (err.code, err.ref) == ("internal", "u1")
    assert h.of(Transcript) == []


async def test_responder_failure_ends_turn_with_error() -> None:
    h = Harness(responder=ScriptedResponder(fail_after=1))
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await h.settle()
    [end] = h.of(TurnEnd)
    assert (end.status, end.error) == ("error", "RuntimeError")


async def test_tts_failure_ends_turn_with_error() -> None:
    class BrokenTTS(FakeTTS):
        async def synthesize(self, text: str, emotion: Any = "neutral") -> bytes:
            raise RuntimeError("SBV2 crashed")

    h = Harness(tts=BrokenTTS())
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await h.settle()
    assert h.of(TurnEnd)[0].status == "error"


# ---------------------------------------------------------------- 不正な入力


async def test_duplicate_begin() -> None:
    h = Harness()
    await h.conv.handle(UtteranceBegin(utt_id="u1"))
    await h.conv.handle(UtteranceBegin(utt_id="u1"))
    assert h.of(Error)[0].code == "bad_message"


async def test_audio_for_unknown_utterance() -> None:
    h = Harness()
    await h.conv.handle(audio("u9", 0))
    assert (h.of(Error)[0].code, h.of(Error)[0].ref) == ("unknown_ref", "u9")


async def test_audio_seq_gap_drops_utterance() -> None:
    h = Harness()
    await h.conv.handle(UtteranceBegin(utt_id="u1"))
    await h.conv.handle(audio("u1", 0))
    await h.conv.handle(audio("u1", 2))
    await h.conv.handle(UtteranceEnd(utt_id="u1", features=FEAT))
    assert [e.code for e in h.of(Error)] == ["bad_message", "unknown_ref"]


async def test_sample_rate_change_drops_utterance() -> None:
    h = Harness()
    await h.conv.handle(UtteranceBegin(utt_id="u1"))
    await h.conv.handle(audio("u1", 0, sr=16000))
    await h.conv.handle(audio("u1", 1, sr=48000))
    assert "sr" in h.of(Error)[0].message


async def test_too_long_utterance_is_rejected() -> None:
    h = Harness(limits=ConversationLimits(max_utterance_s=0.15))
    await h.conv.handle(UtteranceBegin(utt_id="u1"))
    for seq in range(3):
        await h.conv.handle(audio("u1", seq, samples=1600))  # 0.1 秒ずつ
    assert "超えた" in h.of(Error)[0].message
    await h.conv.handle(UtteranceEnd(utt_id="u1", features=FEAT))
    assert h.of(Error)[-1].code == "unknown_ref"


async def test_end_of_unknown_utterance() -> None:
    h = Harness()
    await h.conv.handle(UtteranceEnd(utt_id="u9", features=FEAT))
    assert h.of(Error)[0].code == "unknown_ref"


async def test_handle_returns_false_for_other_messages() -> None:
    from reachy_protocol.messages import Ping

    assert await Harness().conv.handle(Ping(nonce="n")) is False


async def test_vlm_frames_are_accepted_and_ignored_for_now() -> None:
    h = Harness()
    frame = BinaryFrame(BinaryHeader(kind="frame", ref="u1", seq=0, format="jpeg"), b"\xff\xd8")
    assert await h.conv.handle(frame) is True
    assert h.out == []


async def test_close_stops_everything() -> None:
    h = Harness(responder=ScriptedResponder([ReplySentence("x。")] * 50, delay_s=0.05))
    await h.utter("u2")
    await h.conv.handle(ProactiveRequest(intent_id="i1", kind="greet", context=CTX))
    await asyncio.sleep(0.02)
    await h.conv.close()
    assert h.conv._tasks == set()  # pyright: ignore[reportPrivateUsage]
