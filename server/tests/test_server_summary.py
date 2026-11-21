"""会話の要約: session.close → 履歴を要約 → memory.op(add_episode)。LLM は偽物を使う。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from reachy_protocol.messages import (
    MemoryAck,
    MemoryOp,
    PersonProfile,
    SessionClose,
    Situation,
    TurnContext,
)
from reachy_protocol.testing import FakeRobotClient, default_context, run_conversation_scenario
from reachy_protocol.transport import Transport
from reachy_server.agent import AgentResponder
from reachy_server.agent.prompt import PROACTIVE_DIRECTIVES
from reachy_server.config import ServerSettings
from reachy_server.conversation import Conversation, ConversationLimits
from reachy_server.history import HistoryStore, person_thread
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers
from reachy_server.responder import ReplySentence, TurnRequest
from reachy_server.session import Session
from reachy_server.summary import (
    MAX_SUMMARY_CHARS,
    SUMMARY_PROMPT,
    Summarizer,
    clean_summary,
    dialogue,
    render,
)

PERSON = PersonProfile(id="p1", name="山田", familiarity="daily", visit_count=9)
NOW = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)
THREAD = person_thread("robot", "p1", NOW.date())


def ctx() -> TurnContext:
    return TurnContext(person=PERSON, situation=Situation(people_count=1, local_time=NOW))


async def talk(store: HistoryStore, llm: FakeLLM, *texts: str | None) -> None:
    """履歴に会話を作る（None は自発発話の挨拶）。"""
    responder = AgentResponder(llm, checkpointer=await store.checkpointer())
    for text in texts:
        kind = "greet" if text is None else None
        req = TurnRequest(
            ref="r", context=ctx(), user_text=text, proactive_kind=kind, robot_id="robot"
        )
        _ = [e async for e in responder.respond(req) if isinstance(e, ReplySentence)]


# ---------------------------------------------------------------- 整形


def test_dialogue_keeps_only_what_was_said() -> None:
    messages = [
        HumanMessage(PROACTIVE_DIRECTIVES["greet"]),  # 自発発話の合図は発言ではない
        AIMessage("おはよう！"),
        HumanMessage("卒論で画像認識をやってるんだ"),
        AIMessage("", tool_calls=[{"name": "remember_fact", "args": {}, "id": "c1"}]),
        ToolMessage("記録した", tool_call_id="c1"),
        AIMessage("すごい、がんばってね。"),
        SystemMessage("system"),
    ]
    assert dialogue(messages) == [
        ("リーチ", "おはよう！"),
        ("相手", "卒論で画像認識をやってるんだ"),
        ("リーチ", "すごい、がんばってね。"),
    ]
    assert render(dialogue(messages)).splitlines()[1] == "相手: 卒論で画像認識をやってるんだ"


@pytest.mark.parametrize("text", ["なし", "  なし。 ", "特になし", "", "   "])
def test_nothing_to_remember_gives_none(text: str) -> None:
    assert clean_summary(text) is None


def test_clean_summary_strips_quotes_and_truncates() -> None:
    assert clean_summary("「卒論の話をした」") == "卒論の話をした"
    assert len(clean_summary("あ" * 1000) or "") == MAX_SUMMARY_CHARS


def test_prompt_forbids_sensitive_information_and_blaming() -> None:
    for word in ("住所", "健康", "家族", "責める"):
        assert word in SUMMARY_PROMPT


# ---------------------------------------------------------------- 要約の生成


async def test_summarizes_the_thread_from_the_history() -> None:
    store = HistoryStore()
    await talk(
        store,
        FakeLLM(["おはよう、山田さん。", "それはすごいね。"]),
        "おはよう",
        "卒論で画像認識を扱ってる",
    )
    summary_llm = FakeLLM(["卒論で画像認識を扱っていると話していた。"])
    summarizer = Summarizer(summary_llm, await store.checkpointer())

    assert await summarizer.summarize_thread(THREAD) == "卒論で画像認識を扱っていると話していた。"
    system, human = summary_llm.models[0].inputs[0]
    assert system.content == SUMMARY_PROMPT
    assert "相手: 卒論で画像認識を扱ってる" in str(human.content)
    assert "リーチ: それはすごいね。" in str(human.content)


async def test_the_same_conversation_is_not_summarized_twice() -> None:
    store = HistoryStore()
    await talk(store, FakeLLM(["やあ。", "そうなんだ。"]), "こんにちは", "猫を飼ってる")
    summarizer = Summarizer(FakeLLM(["猫を飼っていると話していた。"]), await store.checkpointer())
    assert await summarizer.summarize_thread(THREAD) is not None
    assert await summarizer.summarize_thread(THREAD) is None  # 増えていない


async def test_a_greeting_without_a_reply_has_nothing_to_summarize() -> None:
    store = HistoryStore()
    await talk(store, FakeLLM(["おはよう！"]), None)  # こちらから挨拶しただけ
    llm = FakeLLM(["要約"])
    assert await Summarizer(llm, await store.checkpointer()).summarize_thread(THREAD) is None
    assert llm.models == []  # LLM は呼ばない


async def test_unknown_thread_has_nothing_to_summarize() -> None:
    summarizer = Summarizer(FakeLLM(["要約"]), await HistoryStore().checkpointer())
    assert await summarizer.summarize_thread("robot:nobody:2026-11-21") is None


async def test_a_conversation_with_nothing_to_remember_gives_none() -> None:
    store = HistoryStore()
    await talk(store, FakeLLM(["やあ。"]), "うん")
    summarizer = Summarizer(FakeLLM(["なし"]), await store.checkpointer())
    assert await summarizer.summarize_thread(THREAD) is None


# ---------------------------------------------------------------- 契約: ロボット ⇄ サーバー


async def wait_for(predicate: object, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():  # type: ignore[operator]
            await asyncio.sleep(0.02)


async def test_session_close_sends_the_summary_as_add_episode(
    settings: ServerSettings, token: str
) -> None:
    chat_llm = FakeLLM(["山田さんだね。"])
    summary_llm = FakeLLM(["卒論で画像認識を扱っていると話していた。"])
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=summary_llm)  # 要約は providers.llm

    async def serve(transport: Transport) -> None:
        # 会話の応答だけ、別の LLM（台本）にする
        session = Session(
            transport,
            settings,
            providers,
            responder_factory=lambda _p, saver: AgentResponder(chat_llm, checkpointer=saver),
        )
        await session.run()

    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context())
        await robot.next_turn()
        person = default_context().person
        assert person is not None
        await robot.ep.send(SessionClose(person_id=person.id))
        await wait_for(lambda: robot.memory_ops)
        (op,) = robot.memory_ops
        assert (op.op, op.person_id) == ("add_episode", person.id)
        assert op.value is not None and "話していた" in op.value

    await run_conversation_scenario(serve, scenario, token=token)


async def test_session_close_for_a_stranger_sends_nothing(
    settings: ServerSettings, token: str
) -> None:
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=FakeLLM(["要約"]))

    async def serve(transport: Transport) -> None:
        await Session(transport, settings, providers).run()

    async def scenario(robot: FakeRobotClient) -> None:
        await robot.ep.send(SessionClose(person_id="nobody"))  # この接続では話していない
        await asyncio.sleep(0.3)
        assert robot.memory_ops == []

    await run_conversation_scenario(serve, scenario, token=token)


# ---------------------------------------------------------------- Conversation の動き


def conversation(sent: list[object], summarize: object, **limits: float) -> Conversation:
    async def send(msg: object) -> None:
        sent.append(msg)

    async def send_frame(*_a: object) -> None:
        pass

    class Quick:
        async def respond(self, request: TurnRequest):  # pyright: ignore[reportUnknownParameterType]
            yield ReplySentence("はい。")

    return Conversation(
        send=send,  # pyright: ignore[reportArgumentType]
        send_frame=send_frame,  # pyright: ignore[reportArgumentType]
        stt=FakeSTT(),
        tts=FakeTTS(),
        responder=Quick(),  # pyright: ignore[reportArgumentType]
        limits=ConversationLimits(memory_ack_timeout_s=1.0, **limits),
        robot_id="robot",
        summarize=summarize,  # pyright: ignore[reportArgumentType]
    )


async def greet(conv: Conversation) -> None:
    from reachy_protocol.messages import ProactiveRequest

    await conv.handle(ProactiveRequest(intent_id="i1", kind="greet", strategy="x", context=ctx()))
    while conv._tasks:  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*list(conv._tasks), return_exceptions=True)  # pyright: ignore[reportPrivateUsage]


async def test_summary_uses_the_thread_of_the_last_day_talked() -> None:
    asked: list[str] = []

    async def summarize(thread_id: str) -> str | None:
        asked.append(thread_id)
        return "話していた。"

    sent: list[object] = []
    conv = conversation(sent, summarize)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    await asyncio.sleep(0.1)
    await conv.handle(MemoryAck(op_id="m1", result="applied"))
    assert asked == [THREAD]
    ops = [m for m in sent if isinstance(m, MemoryOp)]
    assert [(o.op, o.value) for o in ops] == [("add_episode", "話していた。")]


async def test_a_failing_summary_does_not_break_the_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken(thread_id: str) -> str | None:
        raise RuntimeError("LLM が落ちた")

    sent: list[object] = []
    conv = conversation(sent, broken)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    while conv._tasks:  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*list(conv._tasks), return_exceptions=True)  # pyright: ignore[reportPrivateUsage]
    assert not [m for m in sent if isinstance(m, MemoryOp)]
    assert "summary failed" in caplog.text


async def test_a_slow_summary_is_given_up() -> None:
    async def slow(thread_id: str) -> str | None:
        await asyncio.sleep(5)
        return "遅い"

    sent: list[object] = []
    conv = conversation(sent, slow, summary_timeout_s=0.05)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    while conv._tasks:  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*list(conv._tasks), return_exceptions=True)  # pyright: ignore[reportPrivateUsage]
    assert not [m for m in sent if isinstance(m, MemoryOp)]


async def test_close_without_a_summarizer_is_ignored() -> None:
    sent: list[object] = []
    conv = conversation(sent, None)
    await greet(conv)
    assert await conv.handle(SessionClose(person_id="p1"))
    await asyncio.sleep(0.05)
    assert not [m for m in sent if isinstance(m, MemoryOp)]


# ---------------------------------------------------------------- 忘却・二重の退出（Codex の指摘）


class SlowSummaryLLM:
    """要約に時間がかかる LLM。同時に走ったかを数える。"""

    name = "slow"

    def __init__(self, delay_s: float = 0.2) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def chat_model(self) -> object:
        outer = self

        class Model:
            async def ainvoke(self, _messages: object) -> AIMessage:
                outer.calls += 1
                await asyncio.sleep(outer.delay_s)
                return AIMessage("卒論の話をしていた。")

        return Model()


async def test_concurrent_summaries_of_one_thread_call_the_llm_once() -> None:
    store = HistoryStore()
    await talk(store, FakeLLM(["やあ。", "そうなんだ。"]), "こんにちは", "卒論をやってる")
    llm = SlowSummaryLLM()
    summarizer = Summarizer(llm, await store.checkpointer())  # pyright: ignore[reportArgumentType]
    results = await asyncio.gather(
        summarizer.summarize_thread(THREAD), summarizer.summarize_thread(THREAD)
    )
    assert llm.calls == 1
    assert sorted(results, key=lambda r: r is None) == ["卒論の話をしていた。", None]


async def test_a_different_conversation_of_the_same_length_is_still_summarized() -> None:
    # 忘れた後に、同じ発言数の別の会話をしても、要約が飛ばされない
    store = HistoryStore()
    await talk(store, FakeLLM(["やあ。", "そうなんだ。"]), "こんにちは", "猫を飼ってる")
    summarizer = Summarizer(FakeLLM(["猫の話をしていた。"]), await store.checkpointer())
    assert await summarizer.summarize_thread(THREAD) is not None
    await store.delete_thread(THREAD)  # 忘れられた
    await talk(store, FakeLLM(["やあ。", "いいね。"]), "こんにちは", "犬を飼ってる")
    assert await summarizer.summarize_thread(THREAD) is not None


async def test_forgetting_during_a_pending_summary_cancels_it() -> None:
    cancelled: list[bool] = []
    release = asyncio.Event()

    async def summarize(thread_id: str) -> str | None:
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return "忘れたはずの会話の要約"

    sent: list[object] = []
    conv = conversation(sent, summarize)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    await asyncio.sleep(0.05)  # 要約中
    await conv.forget_person("p1")  # person.forget が届く
    release.set()
    await asyncio.sleep(0.1)
    assert cancelled == [True]
    assert not [m for m in sent if isinstance(m, MemoryOp)]  # 忘れた会話を add_episode で送らない
    assert not conv._summaries  # pyright: ignore[reportPrivateUsage]


async def test_forgetting_someone_else_keeps_the_summary_running() -> None:
    sent: list[object] = []

    async def summarize(thread_id: str) -> str | None:
        await asyncio.sleep(0.1)
        return "話していた。"

    conv = conversation(sent, summarize)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    await asyncio.sleep(0.02)
    await conv.forget_person("p9")
    await asyncio.sleep(0.2)
    await conv.handle(MemoryAck(op_id="m1", result="applied"))
    assert [m.op for m in sent if isinstance(m, MemoryOp)] == ["add_episode"]


async def test_overlapping_session_close_sends_one_episode() -> None:
    calls: list[str] = []

    async def summarize(thread_id: str) -> str | None:
        calls.append(thread_id)
        await asyncio.sleep(0.1)
        return "話していた。"

    sent: list[object] = []
    conv = conversation(sent, summarize)
    await greet(conv)
    await conv.handle(SessionClose(person_id="p1"))
    await conv.handle(SessionClose(person_id="p1"))  # 退出の合図が重なった
    await asyncio.sleep(0.2)
    await conv.handle(MemoryAck(op_id="m1", result="applied"))
    assert len(calls) == 1
    assert [m.op for m in sent if isinstance(m, MemoryOp)] == ["add_episode"]
    await asyncio.sleep(0.05)
    # 終わったあとの退出は、また要約できる
    await conv.handle(SessionClose(person_id="p1"))
    await asyncio.sleep(0.2)
    assert len(calls) == 2
