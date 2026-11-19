"""文分割と、LLM をストリーミングして文に区切る最小の Responder。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reachy_protocol.messages import PersonProfile, Situation, TurnContext
from reachy_server.providers import FakeLLM
from reachy_server.responder import LLMResponder, ReplySentence, TurnRequest, build_messages
from reachy_server.text import split_completed

SIT = Situation(people_count=1, local_time=datetime(2026, 11, 19, 9, tzinfo=UTC))


@pytest.mark.parametrize(
    ("buffer", "sentences", "rest"),
    [
        ("", [], ""),
        ("こんにちは", [], "こんにちは"),
        ("おはよう。元気？今日は", ["おはよう。", "元気？"], "今日は"),
        ("えっ！？本当に", ["えっ！？"], "本当に"),  # 連続した区切りは 1 文
        ("一行目\n二行目\n", ["一行目", "二行目"], ""),
        ("。 。", ["。", "。"], ""),
        ("Hi! OK.", ["Hi!", "OK."], ""),
    ],
)
def test_split_completed(buffer: str, sentences: list[str], rest: str) -> None:
    assert split_completed(buffer) == (sentences, rest)


async def test_llm_responder_yields_sentences_including_tail() -> None:
    responder = LLMResponder(FakeLLM(replies=["おはよう！今日も来たね。えらい"]))
    req = TurnRequest(ref="u1", context=TurnContext(situation=SIT), user_text="おはよう")
    got = [e async for e in responder.respond(req)]
    assert got == [
        ReplySentence("おはよう！"),
        ReplySentence("今日も来たね。"),
        ReplySentence("えらい"),
    ]


def test_build_messages_for_utterance_and_proactive() -> None:
    person = PersonProfile(id="p1", name="山田", familiarity="daily", visit_count=9)
    msgs = build_messages(
        TurnRequest(ref="u1", context=TurnContext(person=person, situation=SIT), user_text="やあ")
    )
    assert "山田" in msgs[0].text and "daily" in msgs[0].text
    assert msgs[1].text == "やあ"
    anon = build_messages(
        TurnRequest(ref="i1", context=TurnContext(situation=SIT), proactive_kind="see_off")
    )
    assert "相手: 不明" in anon[0].text
    assert "見送" in anon[1].text
