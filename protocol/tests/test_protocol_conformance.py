"""順序規約のチェッカー: 正しいやり取りは通し、違反は理由つきで検出する。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import SecretStr

from reachy_protocol import messages as m
from reachy_protocol.channel import BinaryFrame
from reachy_protocol.conformance import ConversationChecker, Direction, ProtocolViolation
from reachy_protocol.framing import BinaryHeader

R: Direction = "robot->server"
S: Direction = "server->robot"
CTX = m.TurnContext(
    person=m.PersonProfile(id="p1", familiarity="new", visit_count=2),
    situation=m.Situation(people_count=1, local_time=datetime(2026, 11, 19, tzinfo=UTC)),
)
FEAT = m.UtteranceFeatures(duration_s=1, eye_contact_ratio=1, people_count=1)


def pcm(utt: str, seq: int) -> BinaryFrame:
    return BinaryFrame(
        BinaryHeader(kind="audio_in", ref=utt, seq=seq, format="pcm_f32le", sr=16000), b""
    )


def wav(turn: str, seq: int) -> BinaryFrame:
    return BinaryFrame(BinaryHeader(kind="audio_out", ref=turn, seq=seq, format="wav_pcm16"), b"")


HANDSHAKE: list[tuple[Direction, Any]] = [
    (R, m.Hello(robot_id="r", token=SecretStr("t"))),
    (S, m.Welcome(server_version="x")),
]


def utterance(utt: str = "u1", decision: str = "respond") -> list[tuple[Direction, Any]]:
    return [
        (R, m.UtteranceBegin(utt_id=utt)),
        (R, pcm(utt, 0)),
        (R, pcm(utt, 1)),
        (R, m.UtteranceEnd(utt_id=utt, features=FEAT)),
        (S, m.Transcript(utt_id=utt, text="リーチ")),
        (R, m.FloorDecision.model_validate({"utt_id": utt, "decision": decision, "context": CTX})),
    ]


def turn(turn_id: str = "t1", ref: str = "u1", n: int = 2) -> list[tuple[Direction, Any]]:
    steps: list[tuple[Direction, Any]] = [(S, m.TurnStart(turn_id=turn_id, ref=ref))]
    for i in range(n):
        steps += [(S, m.Sentence(turn_id=turn_id, idx=i, text="x")), (S, wav(turn_id, i))]
    steps.append((S, m.TurnEnd(turn_id=turn_id)))
    return steps


def check(steps: list[tuple[Direction, Any]]) -> list[str]:
    c = ConversationChecker()
    for direction, item in steps:
        c.observe(direction, item)
    return c.finish()


def assert_violation(steps: list[tuple[Direction, Any]], fragment: str) -> None:
    violations = check(steps)
    assert any(fragment in v for v in violations), violations


# ---------------------------------------------------------------- 正しいやり取り


def test_full_valid_conversation() -> None:
    steps = (
        HANDSHAKE
        + utterance()
        + turn()[:3]  # turn.start, sentence 0, audio 0
        + [
            (S, m.Motion(turn_id="t1", name="nod")),
            (S, m.MemoryOp(op_id="o1", person_id="p1", op="set_name", value="山田")),
            (R, m.MemoryAck(op_id="o1", result="applied")),
            (S, m.ToolCall(call_id="c1", name="lab_presence")),
            (R, m.ToolResult(call_id="c1", ok=True, result=[])),
        ]
        + turn()[3:]
        + utterance("u2", "ignore")
        + [
            (R, m.ProactiveRequest(intent_id="i1", kind="greet", context=CTX)),
            (R, BinaryFrame(BinaryHeader(kind="frame", ref="i1", seq=0, format="jpeg"), b"")),
        ]
        + turn("t2", "i1", n=1)
        + [(R, m.Ping(nonce="n1")), (S, m.Pong(nonce="n1")), (R, m.SessionClose(person_id="p1"))]
    )
    assert check(steps) == []


def test_cancel_after_end_is_a_tolerated_race() -> None:
    assert check(HANDSHAKE + utterance() + turn() + [(R, m.TurnCancel(turn_id="t1"))]) == []


def test_cancelled_turn_may_drop_audio() -> None:
    steps = (
        HANDSHAKE
        + utterance()
        + [
            (S, m.TurnStart(turn_id="t1", ref="u1")),
            (S, m.Sentence(turn_id="t1", idx=0, text="x")),
            (R, m.TurnCancel(turn_id="t1")),
            (S, m.TurnEnd(turn_id="t1", status="cancelled")),
        ]
    )
    assert check(steps) == []


def test_error_with_ref_releases_obligations() -> None:
    steps = HANDSHAKE + [
        (R, m.UtteranceBegin(utt_id="u1")),
        (R, m.UtteranceEnd(utt_id="u1", features=FEAT)),
        (S, m.Error(code="internal", ref="u1")),  # STT 失敗を通知すれば transcript は不要
    ]
    assert check(steps) == []


def test_error_before_handshake_is_allowed() -> None:
    assert (
        check(
            [(R, m.Hello(robot_id="r", token=SecretStr("bad"))), (S, m.Error(code="unauthorized"))]
        )
        == []
    )


# ---------------------------------------------------------------- 違反の検出


@pytest.mark.parametrize(
    ("steps", "fragment"),
    [
        # ハンドシェイク
        ([(R, m.Ping(nonce="n"))], "hello の前に"),
        (
            [(R, m.Hello(robot_id="r", token=SecretStr("t"))), (S, m.Pong(nonce="n"))],
            "welcome の前に",
        ),
        ([(S, m.Welcome(server_version="x"))], "hello の前に welcome"),
        (HANDSHAKE + [(R, m.Hello(robot_id="r", token=SecretStr("t")))], "hello が 2 回"),
        # 方向
        (HANDSHAKE + [(R, m.TurnEnd(turn_id="t1"))], "robot->server に turn.end は送れない"),
        (HANDSHAKE + [(R, wav("t1", 0))], "robot が binary audio_out"),
        # 発話
        (HANDSHAKE + [(R, pcm("u9", 0))], "開いていない発話"),
        (HANDSHAKE + [(R, m.UtteranceBegin(utt_id="u1")), (R, pcm("u1", 1))], "seq が 0 でなく 1"),
        (HANDSHAKE + [(R, m.UtteranceBegin(utt_id="u1"))] * 2, "utt_id u1 が重複"),
        (HANDSHAKE + [(S, m.Transcript(utt_id="u1", text="x"))], "終了していない発話"),
        (
            HANDSHAKE + utterance()[:5] + [(S, m.Transcript(utt_id="u1", text="x"))],
            "transcript(utt=u1) が 2 回",
        ),
        (
            HANDSHAKE + utterance()[:4] + [(R, m.FloorDecision(utt_id="u1", decision="ignore"))],
            "文字起こし前",
        ),
        (
            HANDSHAKE
            + utterance("u1", "ignore")
            + [(R, m.FloorDecision(utt_id="u1", decision="ignore"))],
            "2 回送られた",
        ),
        # ターン: ignore した発話に応答するのは最重要の違反（割り込み）
        (HANDSHAKE + utterance("u1", "ignore") + turn(), "respond と判定されていない"),
        (HANDSHAKE + utterance("u1", "backchannel") + turn(), "respond と判定されていない"),
        (HANDSHAKE + turn(ref="nope"), "未知の ref"),
        (HANDSHAKE + utterance() + turn() + turn("t2"), "2 回始まった"),
        (
            HANDSHAKE + utterance() + turn()[:1] + [(S, m.Sentence(turn_id="t1", idx=1, text="x"))],
            "idx が 0 でなく 1",
        ),
        (
            HANDSHAKE + utterance() + turn()[:1] + [(S, wav("t1", 0))],
            "対応する sentence がまだない",
        ),
        (HANDSHAKE + utterance() + turn()[:3] + [(S, wav("t1", 0))], "重複"),
        (
            HANDSHAKE + utterance() + turn() + [(S, m.Sentence(turn_id="t1", idx=2, text="x"))],
            "終了済みのターン",
        ),
        (HANDSHAKE + [(S, m.Motion(turn_id="t9", name="nod"))], "未知のターン"),
        (HANDSHAKE + [(R, m.TurnCancel(turn_id="t9"))], "未知のターン"),
        # 応答の義務
        (HANDSHAKE + utterance()[:4], "transcript が返らなかった"),
        (HANDSHAKE + utterance()[:5], "floor.decision が送られなかった"),
        (HANDSHAKE + utterance(), "ターンが始まらなかった"),
        (HANDSHAKE + utterance() + turn()[:-1], "turn.end で閉じられなかった"),
        (HANDSHAKE + utterance() + turn()[:2] + [(S, m.TurnEnd(turn_id="t1"))], "audio_out がない"),
        (
            HANDSHAKE + [(R, m.ProactiveRequest(intent_id="i1", kind="greet", context=CTX))],
            "自発発話の依頼 i1",
        ),
        (
            HANDSHAKE + [(S, m.MemoryOp(op_id="o1", person_id="p", op="forget"))],
            "memory.ack が返らなかった",
        ),
        (HANDSHAKE + [(R, m.MemoryAck(op_id="o9", result="applied"))], "未知の op_id"),
        (HANDSHAKE + [(S, m.ToolCall(call_id="c1", name="x"))], "tool.result が返らなかった"),
        (HANDSHAKE + [(R, m.ToolResult(call_id="c9", ok=True))], "未知の call_id"),
        # 重複・二重応答・その他
        ([(R, pcm("u1", 0))], "hello の前に binary audio_in"),
        (
            HANDSHAKE
            + [(R, BinaryFrame(BinaryHeader(kind="frame", ref="x", seq=0, format="jpeg"), b""))],
            "未知の発話・自発発話",
        ),
        (HANDSHAKE + [(S, m.Welcome(server_version="x"))], "welcome が 2 回"),
        (HANDSHAKE + [(R, m.UtteranceEnd(utt_id="u1", features=FEAT))], "開いた発話がない"),
        (
            HANDSHAKE + [(R, m.ProactiveRequest(intent_id="i1", kind="greet", context=CTX))] * 2,
            "intent_id i1 が重複",
        ),
        (
            HANDSHAKE + [(S, m.MemoryOp(op_id="o1", person_id="p", op="forget"))] * 2,
            "op_id o1 が重複",
        ),
        (
            HANDSHAKE
            + [(S, m.MemoryOp(op_id="o1", person_id="p", op="forget"))]
            + [(R, m.MemoryAck(op_id="o1", result="applied"))] * 2,
            "memory.ack(op=o1) が 2 回",
        ),
        (HANDSHAKE + [(S, m.ToolCall(call_id="c1", name="x"))] * 2, "call_id c1 が重複"),
        (
            HANDSHAKE
            + [(S, m.ToolCall(call_id="c1", name="x"))]
            + [(R, m.ToolResult(call_id="c1", ok=True))] * 2,
            "tool.result(call=c1) が 2 回",
        ),
        (
            HANDSHAKE
            + utterance()
            + turn()[:1]
            + utterance("u2")
            + [(S, m.TurnStart(turn_id="t1", ref="u2"))],
            "turn_id t1 が重複",
        ),
        (HANDSHAKE + [(S, wav("t9", 0))], "未知のターン"),
        (HANDSHAKE + [(R, m.Ping(nonce="n1"))], "pong が返らなかった"),
        (HANDSHAKE + [(S, m.Pong(nonce="n1"))], "対応する ping がない"),
    ],
)
def test_violations(steps: list[tuple[Direction, Any]], fragment: str) -> None:
    assert_violation(steps, fragment)


def test_assert_ok_raises_with_all_violations() -> None:
    c = ConversationChecker()
    c.observe(R, m.Ping(nonce="n"))
    c.observe(R, m.Ping(nonce="n2"))
    with pytest.raises(ProtocolViolation) as e:
        c.assert_ok()
    assert len(e.value.violations) == 2
    assert "hello の前に" in str(e.value)


def test_finish_is_idempotent() -> None:
    c = ConversationChecker()
    c.observe(R, m.Ping(nonce="n"))
    assert c.finish() == c.finish()
