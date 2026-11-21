"""ロボット側ツール（RPC）: LLM のツール呼び出し → tool.call → ロボットの tool.result → LLM。

データはロボットにある（設計書 §1.1）。答えが得られなかったとき、推測で答えさせないことも確かめる。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import JsonValue, SecretStr

from reachy_protocol.messages import (
    PersonProfile,
    Situation,
    ToolCall,
    ToolResult,
    TurnContext,
)
from reachy_protocol.testing import FakeRobotClient, default_context, run_conversation_scenario
from reachy_protocol.transport import Transport
from reachy_server.agent import AgentResponder
from reachy_server.config import ServerSettings
from reachy_server.conversation import Conversation, ConversationLimits
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers
from reachy_server.providers.fakes import ScriptedChatModel, tool_call
from reachy_server.responder import ReplySentence, ToolOutcome, TurnRequest
from reachy_server.session import Session

PERSON = PersonProfile(id="p1", name="山田", familiarity="daily", visit_count=9)
NOW = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)


def ctx() -> TurnContext:
    return TurnContext(person=PERSON, situation=Situation(people_count=1, local_time=NOW))


class Port:
    """ロボット側ツールの口の代わり。呼ばれた内容を覚え、決めた結果を返す。"""

    def __init__(self, outcome: ToolOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, dict[str, JsonValue]]] = []

    async def __call__(self, name: str, args: dict[str, JsonValue]) -> ToolOutcome:
        self.calls.append((name, args))
        return self.outcome


async def run(llm: FakeLLM, port: Port | None) -> tuple[list[str], ScriptedChatModel]:
    request = TurnRequest(ref="u1", context=ctx(), user_text="今日誰が来てる？", robot_tool=port)
    events = [e async for e in AgentResponder(llm).respond(request)]
    return [e.text for e in events if isinstance(e, ReplySentence)], llm.models[0]


def last_tool_result(model: ScriptedChatModel) -> str:
    return [str(m.content) for call in model.inputs for m in call if isinstance(m, ToolMessage)][-1]


# ---------------------------------------------------------------- 呼び出しと結果


async def test_lab_presence_calls_the_robot_and_returns_its_data_to_the_llm() -> None:
    data: JsonValue = {"today": ["山田", "佐藤"], "now": 2}
    port = Port(ToolOutcome("ok", data))
    spoken, model = await run(FakeLLM([tool_call("lab_presence"), "二人来てるよ。"]), port)
    assert port.calls == [("lab_presence", {})]
    assert spoken == ["二人来てるよ。"]
    result = last_tool_result(model)
    assert "山田" in result and "佐藤" in result  # ロボットのデータがそのまま LLM に届く
    assert "そのまま読み上げず" in result


async def test_recall_person_passes_the_name() -> None:
    port = Port(ToolOutcome("ok", {"last_visit": "2026-11-20"}))
    await run(FakeLLM([tool_call("recall_person", {"name": " 佐藤 "}), "昨日来てたよ。"]), port)
    assert port.calls == [("recall_person", {"name": "佐藤"})]


async def test_blank_name_is_not_sent() -> None:
    port = Port(ToolOutcome("ok", {}))
    _, model = await run(FakeLLM([tool_call("recall_person", {"name": "  "}), "誰のこと？"]), port)
    assert port.calls == []
    assert "推測で答えず" in last_tool_result(model)


async def test_long_results_are_truncated() -> None:
    big: JsonValue = {"names": [*(["あ" * 100] * 50)]}
    port = Port(ToolOutcome("ok", big))
    _, model = await run(FakeLLM([tool_call("lab_presence"), "たくさんいるね。"]), port)
    assert "以下省略" in last_tool_result(model)
    assert len(last_tool_result(model)) < 2300


# ---------------------------------------------------------------- 答えが得られないとき


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (ToolOutcome("error", error="unknown tool lab_presence"), "取得できなかった"),
        (ToolOutcome("timeout"), "時間内に答えが得られなかった"),
    ],
)
async def test_failure_tells_the_llm_not_to_guess(outcome: ToolOutcome, expected: str) -> None:
    _, model = await run(
        FakeLLM([tool_call("lab_presence"), "ごめん、分からないや。"]), Port(outcome)
    )
    result = last_tool_result(model)
    assert expected in result
    assert "推測で答えず、分からないと正直に伝える" in result


async def test_without_a_port_the_tool_is_unavailable() -> None:
    _, model = await run(FakeLLM([tool_call("lab_presence"), "分からないや。"]), None)
    assert "この場では使えない" in last_tool_result(model)


async def test_calls_per_turn_are_capped() -> None:
    port = Port(ToolOutcome("ok", {}))
    five = AIMessage(
        content="",
        tool_calls=[{"name": "lab_presence", "args": {}, "id": f"c{i}"} for i in range(5)],
    )
    await run(FakeLLM([five, "ふむ。"]), port)
    assert len(port.calls) == 3


# ---------------------------------------------------------------- 契約: ロボット ⇄ サーバー


async def converse(
    settings: ServerSettings, token: str, llm: FakeLLM, tools: dict[str, object]
) -> FakeRobotClient:
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=llm)
    seen: list[FakeRobotClient] = []

    async def serve(transport: Transport) -> None:
        await Session(transport, settings, providers).run()

    async def scenario(robot: FakeRobotClient) -> None:
        seen.append(robot)
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context())
        await robot.next_turn()

    await run_conversation_scenario(serve, scenario, token=token, tools=tools)
    return seen[0]


async def test_robot_receives_tool_call_and_its_result_reaches_the_llm(
    settings: ServerSettings, token: str
) -> None:
    llm = FakeLLM([tool_call("lab_presence"), "今日は三人来てるよ。"])
    robot = await converse(settings, token, llm, {"lab_presence": lambda _args: {"today": 3}})
    (call,) = robot.tool_calls
    assert (call.name, call.args) == ("lab_presence", {})
    assert '"today": 3' in last_tool_result(llm.models[0])


async def test_a_failing_robot_tool_reaches_the_llm_as_an_error(
    settings: ServerSettings, token: str
) -> None:
    def broken(_args: dict[str, JsonValue]) -> JsonValue:
        raise RuntimeError("DB が開けない")

    llm = FakeLLM([tool_call("lab_presence"), "ごめん、調べられなかった。"])
    await converse(settings, token, llm, {"lab_presence": broken})
    assert "DB が開けない" in last_tool_result(llm.models[0])


# ---------------------------------------------------------------- tool.result の待ち受け


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
        limits=ConversationLimits(tool_timeout_s=timeout_s),
    )


async def test_conversation_matches_results_by_call_id() -> None:
    sent: list[object] = []
    conv = conversation(sent, timeout_s=2.0)
    port = conv._common()["robot_tool"]  # pyright: ignore[reportPrivateUsage]
    first = asyncio.create_task(port("lab_presence", {}))
    second = asyncio.create_task(port("recall_person", {"name": "佐藤"}))
    await asyncio.sleep(0.05)
    calls = [m for m in sent if isinstance(m, ToolCall)]
    assert [(c.call_id, c.name) for c in calls] == [("c1", "lab_presence"), ("c2", "recall_person")]
    # 返事が逆順で来ても、call_id で正しい相手に届く
    await conv.handle(ToolResult(call_id="c2", ok=True, result={"last": "昨日"}))
    await conv.handle(ToolResult(call_id="c1", ok=False, error="boom"))
    assert await second == ToolOutcome("ok", {"last": "昨日"})
    assert await first == ToolOutcome("error", error="boom")


async def test_conversation_times_out_and_ignores_the_late_result() -> None:
    sent: list[object] = []
    conv = conversation(sent, timeout_s=0.05)
    port = conv._common()["robot_tool"]  # pyright: ignore[reportPrivateUsage]
    assert await port("lab_presence", {}) == ToolOutcome("timeout")
    assert await conv.handle(ToolResult(call_id="c1", ok=True, result=1))  # 行き違いは受け流す
    assert not conv._pending_calls  # pyright: ignore[reportPrivateUsage]


async def test_cancelled_turn_leaves_no_pending_call() -> None:
    sent: list[object] = []
    conv = conversation(sent, timeout_s=5.0)
    port = conv._common()["robot_tool"]  # pyright: ignore[reportPrivateUsage]
    task = asyncio.create_task(port("lab_presence", {}))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not conv._pending_calls  # pyright: ignore[reportPrivateUsage]


def test_settings_expose_the_timeout() -> None:
    s = ServerSettings(token=SecretStr("x" * 16))  # pyright: ignore[reportCallIssue]
    assert s.tool_timeout_s == 1.0  # 設計書 §4.2: タイムアウト 1 s
