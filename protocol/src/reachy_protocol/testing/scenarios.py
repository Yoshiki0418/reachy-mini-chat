"""どのサーバー実装にも流せる汎用の契約シナリオ。

サーバーは「Transport を受け取って 1 接続ぶん処理する async 関数」（:data:`ServerHandler`）
として渡す。今は :class:`FakeServer`、M2 以降は本物のサーバーを Fake プロバイダで動かして渡す。

シナリオは、内容（何を話すか）には依存せず、プロトコル上の振る舞いだけを確かめる。
すべての通信は :class:`ConversationChecker` で順序規約も検査する。
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any

from ..conformance import ConversationChecker
from ..messages import Error, Welcome
from ..transport import Transport, memory_pair
from .fake_robot import FakeRobotClient, default_context
from .tap import MonitoredTransport

__all__ = [
    "CONVERSATION_SCENARIOS",
    "HANDSHAKE_SCENARIOS",
    "ConversationScenario",
    "HandshakeScenario",
    "ServerHandler",
    "run_conversation_scenario",
    "run_handshake_scenario",
]

ServerHandler = Callable[[Transport], Coroutine[Any, Any, None]]
ConversationScenario = Callable[[FakeRobotClient], Awaitable[None]]
HandshakeScenario = Callable[[FakeRobotClient], Awaitable[None]]


# ---------------------------------------------------------------- 会話シナリオ（接続済みで開始）


async def respond_turn(robot: FakeRobotClient) -> None:
    """発話 → 文字起こし → respond → 1 文以上の応答と、各文の音声。"""
    transcript = await robot.say()
    await robot.decide(transcript.utt_id, "respond", default_context())
    turn = await robot.next_turn()
    assert turn.ref == transcript.utt_id
    assert turn.status == "ok"
    assert turn.sentences, "応答の文が 1 つもない"
    assert set(turn.audio) == set(range(len(turn.sentences))), "文と音声が対応していない"


async def ignore_is_silent(robot: FakeRobotClient) -> None:
    """ignore と判定した発話には応答しない（人同士の会話に割り込まない）。"""
    transcript = await robot.say()
    await robot.decide(transcript.utt_id, "ignore")
    await robot.expect_silence()


async def backchannel_is_silent(robot: FakeRobotClient) -> None:
    """backchannel はロボット側の非言語反応だけ。サーバーは生成しない。"""
    transcript = await robot.say()
    await robot.decide(transcript.utt_id, "backchannel")
    await robot.expect_silence()


async def proactive_turn(robot: FakeRobotClient) -> None:
    """自発発話の依頼には、その intent_id を ref にしたターンで応える。"""
    intent_id = await robot.request_proactive("greet")
    turn = await robot.next_turn()
    assert turn.ref == intent_id
    assert turn.status == "ok"
    assert turn.sentences


async def cancel_turn(robot: FakeRobotClient) -> None:
    """ターンの途中で取消すと、ターンは必ず turn.end で閉じる（行き違いなら ok もありうる）。"""
    transcript = await robot.say()
    await robot.decide(transcript.utt_id, "respond", default_context())
    start = await robot.wait_turn_start()
    await robot.cancel(start.turn_id)
    turn = await robot.next_turn(start=start)
    assert turn.status in {"cancelled", "ok"}


async def consecutive_turns(robot: FakeRobotClient) -> None:
    """ignore を挟んでも、次の発話には普通に応答できる。"""
    first = await robot.say()
    await robot.decide(first.utt_id, "ignore")
    second = await robot.say()
    await robot.decide(second.utt_id, "respond", default_context())
    turn = await robot.next_turn()
    assert turn.ref == second.utt_id


async def anonymous_person(robot: FakeRobotClient) -> None:
    """相手を同定できない（person=None）ときも応答でき、記憶操作は送られない。"""
    transcript = await robot.say()
    await robot.decide(transcript.utt_id, "respond", default_context(person_id=None))
    turn = await robot.next_turn()
    assert turn.status == "ok"
    assert turn.memory_ops == []


async def ping_pong(robot: FakeRobotClient) -> None:
    await robot.ping()


CONVERSATION_SCENARIOS: dict[str, ConversationScenario] = {
    "respond_turn": respond_turn,
    "ignore_is_silent": ignore_is_silent,
    "backchannel_is_silent": backchannel_is_silent,
    "proactive_turn": proactive_turn,
    "cancel_turn": cancel_turn,
    "consecutive_turns": consecutive_turns,
    "anonymous_person": anonymous_person,
    "ping_pong": ping_pong,
}


# ---------------------------------------------------------------- ハンドシェイク（未接続で開始）


async def handshake_ok(robot: FakeRobotClient) -> None:
    reply = await robot.connect()
    assert isinstance(reply, Welcome), f"welcome でなく {reply}"


async def incompatible_version_rejected(robot: FakeRobotClient) -> None:
    reply = await robot.connect(protocol_version="999.0.0")
    assert isinstance(reply, Error), f"error でなく {reply}"
    assert reply.code == "incompatible_version"


async def bad_token_rejected(robot: FakeRobotClient) -> None:
    robot.token = "wrong-token"
    reply = await robot.connect()
    assert isinstance(reply, Error), f"error でなく {reply}"
    assert reply.code == "unauthorized"


HANDSHAKE_SCENARIOS: dict[str, HandshakeScenario] = {
    "handshake_ok": handshake_ok,
    "incompatible_version_rejected": incompatible_version_rejected,
    "bad_token_rejected": bad_token_rejected,
}


# ---------------------------------------------------------------- 実行器


async def _run(
    server: ServerHandler,
    body: Callable[[FakeRobotClient], Awaitable[None]],
    *,
    connect: bool,
    timeout: float,
    robot_kwargs: dict[str, Any],
) -> ConversationChecker:
    checker = ConversationChecker()
    robot_end, server_end = memory_pair()
    server_task: asyncio.Task[None] = asyncio.create_task(server(server_end))
    try:
        async with asyncio.timeout(timeout):
            async with FakeRobotClient(
                MonitoredTransport(robot_end, checker, side="robot"), **robot_kwargs
            ) as robot:
                if connect:
                    reply = await robot.connect()
                    assert isinstance(reply, Welcome), f"接続できない: {reply}"
                await body(robot)
                # 後から届くはずの応答（memory.ack への反応など）を流し切る
                await asyncio.sleep(0.05)
            await asyncio.wait_for(server_task, 2.0)
    finally:
        if not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
    checker.assert_ok()
    return checker


async def run_conversation_scenario(
    server: ServerHandler,
    scenario: ConversationScenario,
    *,
    timeout: float = 10.0,
    **robot_kwargs: Any,
) -> ConversationChecker:
    """接続してからシナリオを流し、順序規約を検査する。違反があれば ProtocolViolation。"""
    return await _run(server, scenario, connect=True, timeout=timeout, robot_kwargs=robot_kwargs)


async def run_handshake_scenario(
    server: ServerHandler,
    scenario: HandshakeScenario,
    *,
    timeout: float = 10.0,
    **robot_kwargs: Any,
) -> ConversationChecker:
    """未接続の状態からシナリオを流す（ハンドシェイクの検査用）。"""
    return await _run(server, scenario, connect=False, timeout=timeout, robot_kwargs=robot_kwargs)
