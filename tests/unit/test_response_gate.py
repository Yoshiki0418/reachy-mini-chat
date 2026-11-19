"""応答ゲート: 「自分に向けられた発話か」の判定と、文単位ストリーミングの結線。"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fakes import FakeAgent, make_pipeline

AUDIO = np.zeros(16000, dtype=np.float32)


async def collect(gen) -> list[dict]:
    return [ev async for ev in gen]


def types_of(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


async def test_overheard_speech_is_ignored(tmp_path) -> None:
    """目線なし・名前呼びなし = 人同士の会話。割り込まない。"""
    agent = FakeAgent(["こんにちは。"])
    p, tts = make_pipeline("明日のゼミの資料できた？", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=False))
    assert types_of(events) == ["stt", "turn_end"]
    assert agent.calls == []
    assert tts.spoken == []


async def test_name_call_without_eye_contact_responds(tmp_path) -> None:
    agent = FakeAgent(["はーい。"])
    p, tts = make_pipeline("リーチ、今何時？", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=False))
    assert "audio" in types_of(events)
    assert tts.spoken == ["はーい。"]


async def test_eye_contact_responds(tmp_path) -> None:
    agent = FakeAgent(["おはよう！"])
    p, _ = make_pipeline("おはよう", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=True))
    assert types_of(events)[-1] == "turn_end"
    assert "audio" in types_of(events)


async def test_gaze_unavailable_falls_back_to_always_respond(tmp_path) -> None:
    """robot が eye_contact を送らない（gaze 無効）なら従来どおり応答する。"""
    agent = FakeAgent(["うん。"])
    p, _ = make_pipeline("ねえ", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=None))
    assert "audio" in types_of(events)


async def test_empty_transcript_ends_turn(tmp_path) -> None:
    agent = FakeAgent(["x。"])
    p, _ = make_pipeline("", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=True))
    assert types_of(events) == ["stt", "turn_end"]
    assert agent.calls == []


async def test_streams_sentence_by_sentence_with_motion_first(tmp_path) -> None:
    agent = FakeAgent(["おは", "よう！今日も", "来たね。えら", "い"], motions=["happy"])
    p, tts = make_pipeline("リーチおはよう", agent, tmp_path)
    events = await collect(p.run_turn(AUDIO, "s1", eye_contact=True))
    assert types_of(events) == ["stt", "motion", "audio", "audio", "audio", "turn_end"]
    assert tts.spoken == ["おはよう！", "今日も来たね。", "えらい"]
    assert events[1]["name"] == "happy"


async def test_session_id_is_thread_id(tmp_path) -> None:
    agent = FakeAgent(["うん。"])
    p, _ = make_pipeline("リーチ", agent, tmp_path)
    await collect(p.run_turn(AUDIO, "abc123", eye_contact=True))
    assert agent.calls[0]["config"]["configurable"]["thread_id"] == "abc123"


@pytest.mark.parametrize("reason", ["engage", "greet", "idle", "unknown-reason"])
async def test_proactive_turn_speaks_without_stt(tmp_path, reason: str) -> None:
    agent = FakeAgent(["こんにちは！"])
    p, _ = make_pipeline("(unused)", agent, tmp_path)
    events = await collect(p.run_proactive("s1", reason=reason))
    assert "stt" not in types_of(events)
    assert types_of(events)[-1] == "turn_end"
    sent = agent.calls[0]["inputs"]["messages"][0].content
    assert "合図" in sent
