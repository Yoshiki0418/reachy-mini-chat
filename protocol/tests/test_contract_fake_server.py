"""契約テスト: 汎用シナリオを FakeServer に流す。

M2 以降、本物のサーバーも同じ CONVERSATION_SCENARIOS / HANDSHAKE_SCENARIOS を流して検査する。
ここでは FakeServer 自体が契約を守っていること（＝ロボット側テストの相手役として正しいこと）と、
契約テストが規約違反を確実に検出できることを確かめる。
"""

from __future__ import annotations

import asyncio

import pytest

from reachy_protocol.conformance import ProtocolViolation
from reachy_protocol.messages import MemoryOp
from reachy_protocol.testing import (
    CONVERSATION_SCENARIOS,
    HANDSHAKE_SCENARIOS,
    FakeRobotClient,
    FakeServer,
    FakeServerConfig,
    default_context,
    run_conversation_scenario,
)
from reachy_protocol.testing.fake_robot import silence_pcm
from reachy_protocol.testing.scenarios import run_handshake_scenario
from reachy_protocol.transport import Transport, memory_pair


@pytest.mark.parametrize("name", list(CONVERSATION_SCENARIOS))
async def test_fake_server_passes_conversation_scenarios(name: str) -> None:
    server = FakeServer(FakeServerConfig(memory_ops=[("add_fact", "猫が好き")]))
    await run_conversation_scenario(server.serve, CONVERSATION_SCENARIOS[name])


@pytest.mark.parametrize("name", list(HANDSHAKE_SCENARIOS))
async def test_fake_server_passes_handshake_scenarios(name: str) -> None:
    await run_handshake_scenario(FakeServer().serve, HANDSHAKE_SCENARIOS[name])


# ---------------------------------------------------------------- FakeServer の振る舞い


async def test_transcribe_receives_streamed_audio_and_context_reaches_server() -> None:
    seen: dict[str, bytes] = {}

    def transcribe(utt_id: str, pcm: bytes) -> str:
        seen[utt_id] = pcm
        return "リーチ、おはよう"

    server = FakeServer(FakeServerConfig(transcribe=transcribe))
    audio = silence_pcm(0.3)

    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say(audio, chunk_bytes=1000)  # 複数チャンクに分けて送る
        assert t.text == "リーチ、おはよう"
        await robot.decide(t.utt_id, "respond", default_context(name="山田"))
        await robot.next_turn()

    await run_conversation_scenario(server.serve, scenario)
    assert seen == {"u1": audio}
    assert server.log.contexts[0].person is not None
    assert server.log.contexts[0].person.name == "山田"


async def test_memory_ops_are_acked_according_to_robot_policy() -> None:
    server = FakeServer(
        FakeServerConfig(memory_ops=[("set_name", "山田です"), ("add_fact", "猫好き")])
    )

    def policy(op: MemoryOp):  # type: ignore[no-untyped-def]
        return ("rejected", "not_a_name") if op.op == "set_name" else ("applied", None)

    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say()
        await robot.decide(t.utt_id, "respond")
        turn = await robot.next_turn()
        assert [op.op for op in turn.memory_ops] == ["set_name", "add_fact"]

    await run_conversation_scenario(server.serve, scenario, memory_policy=policy)
    assert [(a.result, a.reason) for a in server.log.acks] == [
        ("rejected", "not_a_name"),
        ("applied", None),
    ]


async def test_tool_calls_round_trip() -> None:
    server = FakeServer(
        FakeServerConfig(tool_calls=[("lab_presence", {"date": "today"}), ("missing", {})])
    )

    async def scenario(robot: FakeRobotClient) -> None:
        intent = await robot.request_proactive("greet")
        turn = await robot.next_turn()
        assert turn.ref == intent

    tools = {"lab_presence": lambda args: ["山田", "佐藤"]}
    await run_conversation_scenario(server.serve, scenario, tools=tools)
    ok, missing = server.log.tool_results
    assert ok.ok and ok.result == ["山田", "佐藤"]
    assert not missing.ok and missing.error == "unknown tool missing"


async def test_slow_turn_is_actually_cancelled() -> None:
    server = FakeServer(
        FakeServerConfig(reply=["一つ目。", "二つ目。", "三つ目。"], sentence_interval=1.0)
    )

    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say()
        await robot.decide(t.utt_id, "respond")
        start = await robot.wait_turn_start()
        await robot.cancel(start.turn_id)
        turn = await robot.next_turn(start=start)
        assert turn.status == "cancelled"
        assert len(turn.sentences) < 3

    await run_conversation_scenario(server.serve, scenario)


# ---------------------------------------------------------------- 違反を検出できること


class InterruptingServer(FakeServer):
    """ignore と判定された発話にも応答してしまう（人同士の会話に割り込む）。"""

    async def _handle(self, ep, msg):  # type: ignore[no-untyped-def]
        from reachy_protocol.messages import FloorDecision

        if isinstance(msg, FloorDecision) and msg.decision == "ignore":
            self._spawn(ep, msg.utt_id, default_context())
            return True
        return await super()._handle(ep, msg)


class NeverEndingServer(FakeServer):
    """turn.end を送らない。"""

    async def _turn(self, ep, turn_id, ref, context):  # type: ignore[no-untyped-def]
        from reachy_protocol.messages import TurnStart

        await ep.send(TurnStart(turn_id=turn_id, ref=ref))


async def test_detects_server_that_interrupts_human_conversation() -> None:
    with pytest.raises((ProtocolViolation, AssertionError)) as e:
        await run_conversation_scenario(
            InterruptingServer().serve, CONVERSATION_SCENARIOS["ignore_is_silent"]
        )
    assert "沈黙を期待" in str(e.value) or "respond と判定されていない" in str(e.value)


async def test_detects_server_that_never_ends_turn() -> None:
    async def scenario(robot: FakeRobotClient) -> None:
        t = await robot.say()
        await robot.decide(t.utt_id, "respond")
        await robot.wait_turn_start()

    with pytest.raises(ProtocolViolation, match=r"turn\.end で閉じられなかった"):
        await run_conversation_scenario(NeverEndingServer().serve, scenario)


async def test_detects_silent_server_with_timeout() -> None:
    async def mute(transport: Transport) -> None:
        try:
            while True:
                await transport.recv()
        except Exception:
            return

    with pytest.raises((TimeoutError, AssertionError)):
        await run_conversation_scenario(mute, CONVERSATION_SCENARIOS["respond_turn"], timeout=0.5)


async def test_server_closes_after_failed_handshake() -> None:
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(FakeServer().serve(server_end))
    async with FakeRobotClient(robot_end, token="wrong") as robot:
        await robot.connect()
    await asyncio.wait_for(task, 1.0)  # サーバー側の処理が終わっている
