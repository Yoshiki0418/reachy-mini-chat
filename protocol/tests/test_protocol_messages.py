"""制御メッセージ: 全種類の往復・判別・方向・不変条件・前方互換。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, get_args

import pytest
from pydantic import SecretStr

from reachy_protocol import messages as m

JST = timezone(timedelta(hours=9))

CONTEXT = m.TurnContext(
    person=m.PersonProfile(
        id="p-1",
        name="山田",
        familiarity="daily",
        visit_count=12,
        streak_days=5,
        days_absent=1,
        facts=["コーヒーが好き"],
        last_summary="昨日は締切の話をした",
    ),
    situation=m.Situation(
        people_count=2,
        local_time=datetime(2026, 11, 19, 9, 30, tzinfo=JST),
        others_present=["佐藤"],
        first_visitor_today=True,
    ),
)

# 全メッセージ種別のサンプル（下の test_samples_cover_every_message_type で網羅を保証する）
SAMPLES: list[Any] = [
    m.Hello(robot_id="reachy-1", token=SecretStr("s3cret"), capabilities=["tool.lab_presence"]),
    m.UtteranceBegin(utt_id="u1", doa_rad=0.3, person_hint="p-1"),
    m.UtteranceEnd(
        utt_id="u1",
        features=m.UtteranceFeatures(
            duration_s=1.8, eye_contact_ratio=0.75, people_count=2, speaker_person_id="p-1"
        ),
    ),
    m.FloorDecision(utt_id="u1", decision="respond", reasons=["name_call"], context=CONTEXT),
    m.ProactiveRequest(intent_id="i1", kind="greet", strategy="daily_streak", context=CONTEXT),
    m.MemoryAck(op_id="o1", result="rejected", reason="not_a_name"),
    m.ToolResult(call_id="c1", ok=True, result={"visitors": ["山田", "佐藤"]}),
    m.TurnCancel(turn_id="t1", reason="barge_in"),
    m.PersonForget(person_id="p-1"),
    m.SessionClose(person_id="p-1"),
    m.Welcome(server_version="0.1.0", models={"stt": "large-v3-turbo", "tts": "sbv2"}),
    m.Transcript(utt_id="u1", text="リーチ、おはよう", confidence=0.92),
    m.TurnStart(turn_id="t1", ref="u1"),
    m.Sentence(turn_id="t1", idx=0, text="おはよう！5日連続だね。", emotion="happy"),
    m.Motion(turn_id="t1", name="nod"),
    m.MemoryOp(op_id="o1", person_id="p-1", op="add_fact", value="猫を飼っている"),
    m.ToolCall(call_id="c1", name="lab_presence", args={"date": "2026-11-19"}),
    m.TurnEnd(turn_id="t1", status="ok"),
    m.Ping(nonce="n1"),
    m.Pong(nonce="n1"),
    m.Error(code="unknown_ref", message="no such turn", ref="t9"),
]


def _all_message_classes() -> set[type]:
    (union, _field) = get_args(m.Message)
    return set(get_args(union))


def test_samples_cover_every_message_type() -> None:
    assert {type(s) for s in SAMPLES} == _all_message_classes()


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda s: s.type)
def test_round_trip(msg: Any) -> None:
    wire = m.dump_message(msg)
    back = m.parse_message(wire)
    assert type(back) is type(msg)
    assert back == msg


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda s: s.type)
def test_round_trip_from_dict_and_bytes(msg: Any) -> None:
    wire = m.dump_message(msg)
    assert m.parse_message(json.loads(wire)) == msg
    assert m.parse_message(wire.encode()) == msg


def test_every_type_belongs_to_a_direction() -> None:
    types = {cls.model_fields["type"].default for cls in _all_message_classes()}
    assert types == m.ROBOT_TO_SERVER | m.SERVER_TO_ROBOT
    # 双方向なのは死活監視とエラーだけ
    assert {"ping", "pong", "error"} == m.ROBOT_TO_SERVER & m.SERVER_TO_ROBOT


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda s: s.type)
def test_direction_is_enforced(msg: Any) -> None:
    wire = m.dump_message(msg)
    if msg.type in m.ROBOT_TO_SERVER:
        assert m.parse_from_robot(wire) == msg
    else:
        with pytest.raises(m.MessageDirectionError):
            m.parse_from_robot(wire)
    if msg.type in m.SERVER_TO_ROBOT:
        assert m.parse_from_server(wire) == msg
    else:
        with pytest.raises(m.MessageDirectionError):
            m.parse_from_server(wire)


# ---------------------------------------------------------------- 前方互換・不正入力


def test_unknown_fields_are_ignored_for_forward_compat() -> None:
    wire = json.dumps({"type": "turn.end", "turn_id": "t1", "added_in_1_1": {"x": 1}})
    msg = m.parse_message(wire)
    assert isinstance(msg, m.TurnEnd)
    assert not hasattr(msg, "added_in_1_1")


def test_missing_ts_gets_default() -> None:
    msg = m.parse_message({"type": "ping", "nonce": "n"})
    assert msg.ts > 0


@pytest.mark.parametrize(
    "data",
    [
        '{"type": "no.such.type"}',
        '{"turn_id": "t1"}',  # type なし
        '{"type": "turn.end"}',  # 必須フィールドなし
        '{"type": "turn.end", "turn_id": ""}',  # 空の ID
        '{"type": "turn.end", "turn_id": "t1", "status": "weird"}',
        "not json",
    ],
)
def test_invalid_messages_raise_validation_error(data: str) -> None:
    with pytest.raises(m.ValidationError):
        m.parse_message(data)


def test_messages_are_immutable() -> None:
    msg = m.TurnEnd(turn_id="t1")
    with pytest.raises(m.ValidationError):
        msg.turn_id = "t2"  # type: ignore[misc]


# ---------------------------------------------------------------- 不変条件


def test_respond_requires_context() -> None:
    with pytest.raises(m.ValidationError, match="context"):
        m.FloorDecision(utt_id="u1", decision="respond")


@pytest.mark.parametrize("decision", ["ignore", "backchannel"])
def test_non_respond_does_not_need_context(decision: str) -> None:
    m.FloorDecision.model_validate({"utt_id": "u1", "decision": decision})


@pytest.mark.parametrize("op", ["set_name", "add_fact", "add_episode"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_memory_op_requires_value(op: str, value: str | None) -> None:
    with pytest.raises(m.ValidationError, match="value"):
        m.MemoryOp.model_validate({"op_id": "o", "person_id": "p", "op": op, "value": value})


def test_memory_op_forget_has_no_value() -> None:
    m.MemoryOp(op_id="o", person_id="p", op="forget")
    with pytest.raises(m.ValidationError):
        m.MemoryOp(op_id="o", person_id="p", op="forget", value="x")


def test_failed_tool_result_needs_error() -> None:
    with pytest.raises(m.ValidationError, match="error"):
        m.ToolResult(call_id="c", ok=False)
    m.ToolResult(call_id="c", ok=False, error="timeout")


@pytest.mark.parametrize("ratio", [-0.1, 1.01])
def test_eye_contact_ratio_bounds(ratio: float) -> None:
    with pytest.raises(m.ValidationError):
        m.UtteranceFeatures(duration_s=1, eye_contact_ratio=ratio, people_count=1)


def test_profile_limits_facts() -> None:
    with pytest.raises(m.ValidationError):
        m.PersonProfile(id="p", familiarity="new", visit_count=2, facts=["x"] * 21)


# ---------------------------------------------------------------- 認証トークン


def test_token_is_hidden_in_repr_but_sent_on_wire() -> None:
    hello = m.Hello(robot_id="r", token=SecretStr("s3cret"))
    assert "s3cret" not in repr(hello)
    assert "s3cret" not in str(hello.model_dump())
    assert json.loads(m.dump_message(hello))["token"] == "s3cret"
    back = m.parse_message(m.dump_message(hello))
    assert isinstance(back, m.Hello)
    assert back.token.get_secret_value() == "s3cret"


def test_hello_defaults_to_current_protocol_version() -> None:
    from reachy_protocol import PROTOCOL_VERSION

    assert m.Hello(robot_id="r", token=SecretStr("t")).protocol_version == PROTOCOL_VERSION
