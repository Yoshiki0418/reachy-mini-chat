"""LangGraph エージェント: 文脈の描画・ツール呼び出し・履歴・安全弁。外部 API は使わない。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from reachy_protocol.messages import Motion, PersonProfile, Situation, TurnContext
from reachy_protocol.testing import FakeRobotClient, default_context, run_conversation_scenario
from reachy_protocol.transport import Transport
from reachy_server.agent import AgentResponder
from reachy_server.agent.prompt import ASK_NAME, PERSONA, render_context, time_of_day, user_message
from reachy_server.agent.responder import thread_id
from reachy_server.config import ServerSettings
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers, build_providers
from reachy_server.providers.fakes import ScriptedChatModel, tool_call
from reachy_server.responder import ReplyEvent, ReplyMotion, ReplySentence, TurnRequest
from reachy_server.session import Session

JST = timezone(timedelta(hours=9))
YAMADA = PersonProfile(
    id="p1",
    name="山田",
    familiarity="daily",
    visit_count=30,
    streak_days=5,
    days_absent=1,
    facts=["コーヒーが好き"],
    last_summary="締切が近いと話していた",
)


def ctx(person: PersonProfile | None = YAMADA, hour: int = 9, **sit: object) -> TurnContext:
    return TurnContext(
        person=person,
        situation=Situation.model_validate(
            {"people_count": 1, "local_time": datetime(2026, 11, 19, hour, 30, tzinfo=JST)} | sit
        ),
    )


def req(
    text: str | None = "おはよう", context: TurnContext | None = None, **kw: object
) -> TurnRequest:
    return TurnRequest(ref="u1", context=context or ctx(), user_text=text, **kw)  # type: ignore[arg-type]


async def run(
    llm: FakeLLM, *requests: TurnRequest
) -> tuple[list[list[ReplyEvent]], ScriptedChatModel]:
    responder = AgentResponder(llm)
    out = [[e async for e in responder.respond(r)] for r in requests]
    return out, llm.models[0]


def sentences(events: list[ReplyEvent]) -> list[str]:
    return [e.text for e in events if isinstance(e, ReplySentence)]


# ---------------------------------------------------------------- 応答とツール


async def test_reply_is_split_into_sentences() -> None:
    [events], _ = await run(FakeLLM(["おはよう！今日も来たね。えらい"]), req())
    assert events == [
        ReplySentence("おはよう！"),
        ReplySentence("今日も来たね。"),
        ReplySentence("えらい"),
    ]


async def test_motion_tool_becomes_reply_motion_before_sentences() -> None:
    llm = FakeLLM([tool_call("express_motion", {"motion": "happy"}), "やったね！うれしい。"])
    [events], _ = await run(llm, req("合格したよ"))
    assert events == [
        ReplyMotion("happy"),
        ReplySentence("やったね！"),
        ReplySentence("うれしい。"),
    ]


async def test_tool_results_are_not_spoken() -> None:
    llm = FakeLLM([tool_call("get_current_time"), "いまは朝の九時半だよ。"])
    [events], model = await run(llm, req("今何時？"))
    assert sentences(events) == ["いまは朝の九時半だよ。"]
    tool_msg = model.inputs[1][-1]  # 2 回目の LLM 呼び出しにはツールの結果が入っている
    assert tool_msg.type == "tool" and "年" in str(tool_msg.content)


async def test_runaway_tool_loop_is_stopped() -> None:
    llm = FakeLLM([tool_call("get_current_time")])  # 永遠にツールを呼び続ける
    responder = AgentResponder(llm)
    with pytest.raises(GraphRecursionError):
        async with asyncio.timeout(5):
            _ = [e async for e in responder.respond(req())]


# ---------------------------------------------------------------- プロンプトと文脈


async def test_system_prompt_contains_persona_and_turn_context() -> None:
    _, model = await run(FakeLLM(["やあ。"]), req())
    system, human = model.inputs[0]
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert system.text.startswith(PERSONA)
    for fragment in [
        "山田さん",
        "ほぼ毎日来ている常連",
        "5 日連続",
        "1 日ぶり",
        "コーヒーが好き",
        "締切が近い",
        "（木）09:30・朝",
    ]:
        assert fragment in system.text, fragment
    assert human.text == "おはよう"


async def test_unknown_name_asks_for_it_and_anonymous_is_described() -> None:
    nameless = YAMADA.model_copy(
        update={"name": None, "familiarity": "first_time", "visit_count": 0}
    )
    _, m1 = await run(FakeLLM(["やあ。"]), req(context=ctx(nameless)))
    assert ASK_NAME in m1.inputs[0][0].text and "名前未登録の人" in m1.inputs[0][0].text
    _, m2 = await run(FakeLLM(["やあ。"]), req(context=ctx(None)))
    assert "誰か分からない" in m2.inputs[0][0].text and ASK_NAME not in m2.inputs[0][0].text


async def test_proactive_request_uses_directive() -> None:
    _, model = await run(
        FakeLLM(["おはよう！"]), req(text=None, proactive_kind="greet", strategy="daily_streak")
    )
    system, human = model.inputs[0]
    assert human.text.startswith("（合図：目の前に人が来ました")
    assert "話しかけ方の方針: daily_streak" in system.text


# ---------------------------------------------------------------- 履歴（スレッド）


async def test_history_is_kept_per_person_and_day_with_fresh_context() -> None:
    llm = FakeLLM(["一回目の返事。", "二回目の返事。", "別の人への返事。"])
    sato = YAMADA.model_copy(update={"id": "p2", "name": "佐藤"})
    await run(
        llm,
        req("最初の話", ctx(hour=9)),
        req("続きの話", ctx(hour=10)),
        req("こんにちは", ctx(sato, hour=10)),
    )
    model = llm.models[0]
    second, third = model.inputs[1], model.inputs[2]
    # 同じ人・同じ日: 前のやり取りが履歴にある。文脈（システム）は最新の時刻だけ
    assert [m.text for m in second[1:]] == ["最初の話", "一回目の返事。", "続きの話"]
    assert "10:30" in second[0].text and "09:30" not in second[0].text
    # 別の人: 履歴は混ざらない
    assert [m.text for m in third[1:]] == ["こんにちは"]


def test_thread_id() -> None:
    assert thread_id(req()) == "p1:2026-11-19"
    assert thread_id(req(context=ctx(None))) == "anon:2026-11-19"


# ---------------------------------------------------------------- 描画の細部


@pytest.mark.parametrize(
    ("hour", "label"),
    [
        (4, "深夜"),
        (5, "朝"),
        (10, "朝"),
        (11, "昼"),
        (13, "昼"),
        (14, "午後"),
        (17, "午後"),
        (18, "夜"),
        (22, "夜"),
        (23, "深夜"),
    ],
)
def test_time_of_day(hour: int, label: str) -> None:
    assert time_of_day(hour) == label


def test_render_context_situation_details() -> None:
    text = render_context(
        req(context=ctx(people_count=3, others_present=["佐藤", "鈴木"], first_visitor_today=True))
    )
    assert "3 人（ほかに: 佐藤、鈴木）" in text
    assert "今日いちばん最初に来た人" in text


def test_render_context_omits_trivial_details() -> None:
    newbie = PersonProfile(
        id="p3", name="田中", familiarity="new", visit_count=2, streak_days=1, days_absent=0
    )
    text = render_context(req(context=ctx(newbie)))
    assert "連続" not in text and "ぶり" not in text and "覚えていること" not in text


@pytest.mark.parametrize("kind", ["greet", "engage", "see_off"])
def test_user_message_for_each_proactive_kind(kind: str) -> None:
    assert user_message(req(text=None, proactive_kind=kind)).startswith("（合図：")


# ---------------------------------------------------------------- 結線


async def test_motion_reaches_robot_through_session(settings: ServerSettings, token: str) -> None:
    llm = FakeLLM([tool_call("express_motion", {"motion": "nod"}), "うん、そうだね。"])
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=llm)

    async def serve(transport: Transport) -> None:
        await Session(transport, settings, providers).run()

    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context())
        turn = await robot.next_turn()
        assert turn.motions == ["nod"]
        assert [s.text for s in turn.sentences] == ["うん、そうだね。"]
        assert any(isinstance(m, Motion) for m in robot.received)

    await run_conversation_scenario(serve, scenario, token=token)


def test_openai_provider_is_built_lazily_without_network(
    settings: ServerSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    providers = build_providers(
        settings.model_copy(
            update={"llm": "openai", "openai_model": "gpt-4o-mini", "llm_max_tokens": 120}
        )
    )
    model = providers.llm.chat_model()
    assert providers.describe()["llm"] == "openai"
    assert type(model).__name__ == "ChatOpenAI"
    assert (model.model_name, model.max_tokens) == ("gpt-4o-mini", 120)  # type: ignore[attr-defined]
