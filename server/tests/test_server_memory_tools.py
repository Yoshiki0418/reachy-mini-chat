"""記憶ツール: LLM のツール呼び出し → memory.op → ロボットの検証 → 結果が LLM に戻る。

人物の記憶の正はロボット側（設計書 §2.4）。拒否された・確認できなかったことを、覚えた体で
話さないことを、LLM に返る文面と、ロボット役を立てた契約テストで確かめる。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from reachy_protocol.messages import (
    MemoryAck,
    MemoryOp,
    PersonProfile,
    Situation,
    TurnContext,
)
from reachy_protocol.testing import FakeRobotClient, default_context, run_conversation_scenario
from reachy_protocol.transport import Transport
from reachy_server.agent import AgentResponder
from reachy_server.config import ServerSettings
from reachy_server.conversation import Conversation, ConversationLimits
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers
from reachy_server.providers.fakes import ScriptedChatModel, tool_call
from reachy_server.responder import (
    MemoryOpName,
    MemoryResult,
    ReplyEvent,
    ReplySentence,
    TurnRequest,
)
from reachy_server.session import Session

PERSON = PersonProfile(id="p1", name=None, familiarity="first_time", visit_count=1)
NOW = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)


def ctx(person: PersonProfile | None = PERSON) -> TurnContext:
    return TurnContext(person=person, situation=Situation(people_count=1, local_time=NOW))


class Recorder:
    """記憶操作の口の代わり。呼ばれた内容を覚え、決めた結果を返す。"""

    def __init__(self, result: MemoryResult | None = None) -> None:
        self.result = result or MemoryResult("applied")
        self.calls: list[tuple[str, MemoryOpName, str | None]] = []

    async def __call__(self, person_id: str, op: MemoryOpName, value: str | None) -> MemoryResult:
        self.calls.append((person_id, op, value))
        return self.result


async def run(
    llm: FakeLLM, port: Recorder | None, person: PersonProfile | None = PERSON
) -> tuple[list[ReplyEvent], ScriptedChatModel]:
    request = TurnRequest(ref="u1", context=ctx(person), user_text="こんにちは", memory=port)
    events = [e async for e in AgentResponder(llm).respond(request)]
    return events, llm.models[0]


def tool_results(model: ScriptedChatModel) -> list[str]:
    """LLM が（次の呼び出しで）見たツールの結果。"""
    return [str(m.content) for call in model.inputs for m in call if isinstance(m, ToolMessage)][
        -1:
    ]


# ---------------------------------------------------------------- 発行

TOOLS: list[tuple[str, dict[str, str], MemoryOpName, str | None]] = [
    ("set_person_name", {"name": " 山田 "}, "set_name", "山田"),
    ("remember_fact", {"fact": "コーヒーが好き"}, "add_fact", "コーヒーが好き"),
    ("forget_me", {}, "forget", None),
]


@pytest.mark.parametrize(("tool", "args", "op", "value"), TOOLS)
async def test_each_tool_issues_its_memory_op(
    tool: str, args: dict[str, str], op: MemoryOpName, value: str | None
) -> None:
    port = Recorder()
    events, _ = await run(FakeLLM([tool_call(tool, args), "うん。"]), port)
    assert port.calls == [("p1", op, value)]
    assert [e.text for e in events if isinstance(e, ReplySentence)] == ["うん。"]


async def test_applied_tells_the_llm_it_is_recorded() -> None:
    _, model = await run(
        FakeLLM([tool_call("set_person_name", {"name": "山田"}), "山田さん！"]), Recorder()
    )
    (result,) = tool_results(model)
    assert "「山田」と記録した" in result
    assert "記録できなかった" not in result


# ---------------------------------------------------------------- 拒否・未確認は覚えた体で話さない


async def test_rejected_is_reported_with_the_reason_and_not_treated_as_remembered() -> None:
    port = Recorder(MemoryResult("rejected", "名前らしくない"))
    _, model = await run(
        FakeLLM([tool_call("set_person_name", {"name": "あああ"}), "もう一度いい？"]), port
    )
    (result,) = tool_results(model)
    assert "記録できなかった" in result and "名前らしくない" in result
    assert "覚えたことにして話さない" in result


async def test_timeout_is_not_treated_as_applied() -> None:
    port = Recorder(MemoryResult("timeout"))
    _, model = await run(
        FakeLLM([tool_call("remember_fact", {"fact": "猫が好き"}), "そうなんだ。"]), port
    )
    (result,) = tool_results(model)
    assert "確認できなかった" in result and "覚えたことにして話さない" in result


async def test_rejected_result_stays_in_history_for_later_turns() -> None:
    # 拒否の結果は履歴（チェックポイント）に残るので、次のターンの LLM も覚えていない前提で話せる
    port = Recorder(MemoryResult("rejected", "機微情報"))
    llm = FakeLLM(
        [tool_call("remember_fact", {"fact": "住所は…"}), "それは覚えないでおくね。", "ところで。"]
    )
    responder = AgentResponder(llm)
    for text in ("住所は…", "ところで"):
        request = TurnRequest(ref=text, context=ctx(), user_text=text, memory=port)
        _ = [e async for e in responder.respond(request)]
    last_input = llm.models[0].inputs[-1]
    assert any(
        isinstance(m, ToolMessage) and "記録できなかった" in str(m.content) for m in last_input
    )


# ---------------------------------------------------------------- 発行しないとき


async def test_unknown_person_cannot_be_remembered() -> None:
    port = Recorder()
    _, model = await run(
        FakeLLM([tool_call("set_person_name", {"name": "山田"}), "はじめまして。"]),
        port,
        person=None,
    )
    assert port.calls == []
    assert "相手が誰か分からない" in tool_results(model)[0]


async def test_without_a_memory_port_tools_report_unavailable() -> None:
    _, model = await run(
        FakeLLM([tool_call("remember_fact", {"fact": "猫が好き"}), "うん。"]), None
    )
    assert "記憶の機能が使えない" in tool_results(model)[0]


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("set_person_name", {"name": "   "}),
        ("set_person_name", {"name": "あ" * 33}),
        ("remember_fact", {"fact": ""}),
        ("remember_fact", {"fact": "あ" * 201}),
    ],
)
async def test_invalid_values_are_not_sent_to_the_robot(tool: str, args: dict[str, str]) -> None:
    port = Recorder()
    _, model = await run(FakeLLM([tool_call(tool, args), "うん。"]), port)
    assert port.calls == []
    assert "覚えたことにして話さない" in tool_results(model)[0]


async def test_ops_per_turn_are_capped() -> None:
    port = Recorder()
    # LLM が 1 回の応答で 5 件を同時に頼んでも、発行されるのは 3 件まで
    five = AIMessage(
        content="",
        tool_calls=[
            {"name": "remember_fact", "args": {"fact": f"事実{i}"}, "id": f"c{i}"} for i in range(5)
        ],
    )
    llm = FakeLLM([five, "ふむ。"])
    request = TurnRequest(ref="u1", context=ctx(), user_text="色々話す", memory=port)
    _ = [e async for e in AgentResponder(llm).respond(request)]
    assert len(port.calls) == 3


# ---------------------------------------------------------------- 契約: ロボット ⇄ サーバー

Policy = Callable[[MemoryOp], tuple[Literal["applied", "rejected"], str | None]]


async def converse(
    settings: ServerSettings, token: str, llm: FakeLLM, policy: Policy
) -> tuple[FakeRobotClient, list[str]]:
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=llm)
    seen: list[FakeRobotClient] = []

    async def serve(transport: Transport) -> None:
        await Session(transport, settings, providers).run()

    async def scenario(robot: FakeRobotClient) -> None:
        seen.append(robot)
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context())
        await robot.next_turn()

    await run_conversation_scenario(serve, scenario, token=token, memory_policy=policy)
    robot = seen[0]
    return robot, [m.op_id for m in robot.memory_ops]


async def test_robot_receives_memory_op_and_answers_with_an_ack(
    settings: ServerSettings, token: str
) -> None:
    llm = FakeLLM([tool_call("set_person_name", {"name": "山田"}), "山田さん、よろしくね。"])
    robot, op_ids = await converse(settings, token, llm, lambda op: ("applied", None))
    (op,) = robot.memory_ops
    assert (op.op, op.value) == ("set_name", "山田")
    assert op.person_id == default_context().person.id  # pyright: ignore[reportOptionalMemberAccess]
    assert op_ids == [op.op_id]


async def test_robot_rejection_reaches_the_llm_as_not_remembered(
    settings: ServerSettings, token: str
) -> None:
    llm = FakeLLM([tool_call("set_person_name", {"name": "あ"}), "もう一度教えてくれる？"])
    _ = await converse(settings, token, llm, lambda op: ("rejected", "短すぎる"))
    (result,) = tool_results(llm.models[0])
    assert "記録できなかった" in result and "短すぎる" in result


# ---------------------------------------------------------------- memory.ack の待ち受け


def conversation(sent: list[object], timeout_s: float) -> Conversation:
    async def send(msg: object) -> None:
        sent.append(msg)

    async def send_frame(*_a: object) -> None:  # pragma: no cover - 音声は送らない
        raise AssertionError

    class NoResponder:
        def respond(self, request: TurnRequest):  # pyright: ignore[reportUnknownParameterType]
            raise AssertionError

    return Conversation(
        send=send,  # pyright: ignore[reportArgumentType]
        send_frame=send_frame,  # pyright: ignore[reportArgumentType]
        stt=FakeSTT(),
        tts=FakeTTS(),
        responder=NoResponder(),  # pyright: ignore[reportArgumentType]
        limits=ConversationLimits(memory_ack_timeout_s=timeout_s),
    )


async def test_conversation_waits_for_the_matching_ack() -> None:
    sent: list[object] = []
    conv = conversation(sent, timeout_s=2.0)
    port = conv._common()["memory"]  # pyright: ignore[reportPrivateUsage]
    first = asyncio.create_task(port("p1", "add_fact", "猫が好き"))
    second = asyncio.create_task(port("p1", "add_fact", "犬も好き"))
    await asyncio.sleep(0.05)
    ops = [m for m in sent if isinstance(m, MemoryOp)]
    assert [o.op_id for o in ops] == ["m1", "m2"]
    # 返事が逆順で来ても、op_id で正しい相手に届く
    await conv.handle(MemoryAck(op_id="m2", result="rejected", reason="重複"))
    await conv.handle(MemoryAck(op_id="m1", result="applied"))
    assert await first == MemoryResult("applied")
    assert await second == MemoryResult("rejected", "重複")


async def test_conversation_times_out_and_ignores_the_late_ack() -> None:
    sent: list[object] = []
    conv = conversation(sent, timeout_s=0.05)
    port = conv._common()["memory"]  # pyright: ignore[reportPrivateUsage]
    assert await port("p1", "forget", None) == MemoryResult("timeout")
    assert await conv.handle(MemoryAck(op_id="m1", result="applied"))  # 行き違いは黙って受け流す
    assert not conv._pending_ops  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------- 忘却と履歴（ターン中の書き戻し）


class GatedResponder:
    """1 文話し、ゲートが開くまで止まり、履歴を書き戻してから最後の 1 文を話す。

    実行中のグラフが、ターンの途中で消された履歴を後から書き戻す状況を作る。
    """

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def respond(self, request: TurnRequest):  # pyright: ignore[reportUnknownParameterType]
        yield ReplySentence("はい。")
        self.started.set()
        await self.gate.wait()
        self.events.append("write")  # グラフのチェックポイントの書き込み
        yield ReplySentence("忘れたよ。")


def forgetting_conversation(
    events: list[str], responder: GatedResponder, sent: list[object]
) -> Conversation:
    async def send(msg: object) -> None:
        sent.append(msg)

    async def send_frame(*_a: object) -> None:
        pass

    async def forget_history(robot_id: str, person_id: str) -> None:
        events.append(f"delete:{person_id}")

    return Conversation(
        send=send,  # pyright: ignore[reportArgumentType]
        send_frame=send_frame,  # pyright: ignore[reportArgumentType]
        stt=FakeSTT(),
        tts=FakeTTS(),
        responder=responder,  # pyright: ignore[reportArgumentType]
        limits=ConversationLimits(memory_ack_timeout_s=1.0),
        forget_history=forget_history,
    )


async def start_proactive_turn(conv: Conversation, person: PersonProfile | None = PERSON) -> None:
    from reachy_protocol.messages import ProactiveRequest

    await conv.handle(
        ProactiveRequest(intent_id="i1", kind="greet", strategy="first", context=ctx(person))
    )


async def test_forget_during_a_turn_deletes_again_after_the_graph_stops_writing() -> None:
    events: list[str] = []
    responder = GatedResponder(events)
    conv = forgetting_conversation(events, responder, [])
    await start_proactive_turn(conv)
    await responder.started.wait()

    await conv.forget_person("p1")  # ターンの途中で person.forget が届く
    responder.gate.set()  # そのあと、実行中のグラフが履歴を書き戻す
    await _drain(conv)

    # 書き戻しの後にもう一度消している（最後が delete）
    assert events == ["delete:p1", "write", "delete:p1"]


async def _drain(conv: Conversation) -> None:
    while conv._tasks:  # pyright: ignore[reportPrivateUsage]
        await asyncio.gather(*list(conv._tasks), return_exceptions=True)  # pyright: ignore[reportPrivateUsage]


async def test_forget_for_someone_else_or_with_no_turn_deletes_once() -> None:
    events: list[str] = []
    responder = GatedResponder(events)
    conv = forgetting_conversation(events, responder, [])
    await conv.forget_person("p9")  # ターンがない
    await start_proactive_turn(conv)
    await responder.started.wait()
    await conv.forget_person("p9")  # 別の人
    responder.gate.set()
    await _drain(conv)
    assert events == ["delete:p9", "delete:p9", "write"]  # p1 のターンの後に余計な削除はない


async def test_a_cancelled_turn_still_cleans_up_after_a_forget() -> None:
    from reachy_protocol.messages import TurnCancel

    events: list[str] = []
    responder = GatedResponder(events)
    sent: list[object] = []
    conv = forgetting_conversation(events, responder, sent)
    await start_proactive_turn(conv)
    await responder.started.wait()
    await conv.forget_person("p1")
    await conv.handle(TurnCancel(turn_id="t1"))
    await _drain(conv)
    assert events == ["delete:p1", "delete:p1"]  # 取消でも、ターンの後にもう一度消す


async def test_forget_op_acknowledged_as_applied_deletes_the_history() -> None:
    events: list[str] = []
    conv = forgetting_conversation(events, GatedResponder(events), [])
    port = conv._common()["memory"]  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(port("p1", "forget", None))
    await asyncio.sleep(0.05)
    await conv.handle(MemoryAck(op_id="m1", result="applied"))
    assert await task == MemoryResult("applied")
    assert events == ["delete:p1"]  # ロボットが person.forget を送らなくても消える


async def test_forget_op_rejected_keeps_the_history() -> None:
    events: list[str] = []
    conv = forgetting_conversation(events, GatedResponder(events), [])
    port = conv._common()["memory"]  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(port("p1", "forget", None))
    await asyncio.sleep(0.05)
    await conv.handle(MemoryAck(op_id="m1", result="rejected", reason="同意が必要"))
    assert await task == MemoryResult("rejected", "同意が必要")
    assert events == []


async def test_forget_me_leaves_no_conversation_on_the_server(
    settings: ServerSettings, token: str
) -> None:
    """通し: 実際の履歴ストアで、forget_me の会話がターン後に残らない。"""
    from reachy_protocol.messages import PersonForget
    from reachy_server.history import HistoryStore

    store = HistoryStore()
    person = default_context().person
    assert person is not None
    llm = FakeLLM([tool_call("forget_me"), "わかった、忘れるね。"])
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=llm)
    robot_box: list[FakeRobotClient] = []

    def policy(op: MemoryOp) -> tuple[Literal["applied", "rejected"], str | None]:
        # ロボットは forget を適用し、ack より先に person.forget を送る（ターンの途中）
        asyncio.get_running_loop().create_task(
            robot_box[0].ep.send(PersonForget(person_id=op.person_id))
        )
        return "applied", None

    async def serve(transport: Transport) -> None:
        await Session(transport, settings, providers, history=store).run()

    async def scenario(robot: FakeRobotClient) -> None:
        robot_box.append(robot)
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context())
        turn = await robot.next_turn()
        assert turn.status == "ok"

    await run_conversation_scenario(serve, scenario, token=token, memory_policy=policy)
    leftovers = [t for t in await store.thread_ids() if person.id in t]
    assert leftovers == []
