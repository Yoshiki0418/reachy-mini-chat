"""Session: #11 の汎用契約シナリオを本物のサーバーに流す。"""

from __future__ import annotations

import asyncio
import json

import pytest

from reachy_protocol.messages import Error, Ping, TurnEnd, dump_message
from reachy_protocol.testing import (
    CONVERSATION_SCENARIOS,
    HANDSHAKE_SCENARIOS,
    FakeRobotClient,
    run_conversation_scenario,
    run_handshake_scenario,
)
from reachy_protocol.transport import Transport, TransportClosed, memory_pair
from reachy_server.config import ServerSettings
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers
from reachy_server.session import Session


def fake_providers() -> Providers:
    return Providers(stt=FakeSTT(), tts=FakeTTS(), llm=FakeLLM())


def handler(settings: ServerSettings):  # type: ignore[no-untyped-def]
    async def serve(transport: Transport) -> None:
        await Session(transport, settings, fake_providers()).run()

    return serve


@pytest.mark.parametrize("name", list(HANDSHAKE_SCENARIOS))
async def test_handshake_scenarios(settings: ServerSettings, token: str, name: str) -> None:
    await run_handshake_scenario(handler(settings), HANDSHAKE_SCENARIOS[name], token=token)


@pytest.mark.parametrize("name", list(CONVERSATION_SCENARIOS))
async def test_conversation_scenarios(settings: ServerSettings, token: str, name: str) -> None:
    await run_conversation_scenario(handler(settings), CONVERSATION_SCENARIOS[name], token=token)


# ---------------------------------------------------------------- 低レベルの振る舞い


async def _start(settings: ServerSettings) -> tuple[Transport, asyncio.Task[None]]:
    robot, server = memory_pair()
    task = asyncio.create_task(Session(server, settings, fake_providers()).run())
    return robot, task


async def _recv_json(t: Transport) -> dict[str, object]:
    frame = await asyncio.wait_for(t.recv(), 1.0)
    assert isinstance(frame, str)
    return json.loads(frame)


async def _assert_closed(t: Transport, task: asyncio.Task[None]) -> None:
    await asyncio.wait_for(task, 1.0)
    with pytest.raises(TransportClosed):
        await asyncio.wait_for(t.recv(), 1.0)


async def test_first_message_must_be_hello(settings: ServerSettings) -> None:
    robot, task = await _start(settings)
    await robot.send(dump_message(Ping(nonce="n")))
    reply = await _recv_json(robot)
    assert reply["type"] == "error" and reply["code"] == "bad_message"
    await _assert_closed(robot, task)


async def test_handshake_timeout_closes(settings: ServerSettings) -> None:
    robot, task = await _start(settings)  # 何も送らない
    reply = await _recv_json(robot)
    assert reply["code"] == "timeout"
    await _assert_closed(robot, task)


async def test_garbage_before_handshake_closes(settings: ServerSettings) -> None:
    robot, task = await _start(settings)
    await robot.send("not json")
    assert (await _recv_json(robot))["code"] == "bad_message"
    await _assert_closed(robot, task)


async def test_bad_frames_after_handshake_keep_connection(
    settings: ServerSettings, token: str
) -> None:
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, fake_providers()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        await robot_end.send("not json")
        await robot_end.send(b"\x00\x00")  # 壊れたバイナリ
        await robot_end.send(dump_message(TurnEnd(turn_id="t1")))  # 逆方向
        await robot.ping()  # 接続は生きている
        errors = [m for m in robot.received if isinstance(m, Error)]
        assert [e.code for e in errors] == ["bad_message"] * 3
    await asyncio.wait_for(task, 1.0)


async def test_second_hello_is_rejected(settings: ServerSettings, token: str) -> None:
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, fake_providers()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        reply = await robot.connect()
        assert isinstance(reply, Error) and reply.code == "bad_message"
    await asyncio.wait_for(task, 1.0)


async def test_unexpected_crash_reports_internal_error_and_closes(
    settings: ServerSettings, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1 接続の想定外の例外で、サーバー全体を巻き込まない。"""

    async def boom(self: Session, item: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(Session, "_dispatch", boom)
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, fake_providers()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        await robot_end.send(dump_message(Ping(nonce="n")))
        await asyncio.wait_for(task, 1.0)  # 例外を外へ漏らさずに終わる
    assert any(isinstance(m, Error) and m.code == "internal" for m in robot.received)


async def test_error_from_robot_is_accepted(settings: ServerSettings, token: str) -> None:
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, fake_providers()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        await robot_end.send(dump_message(Error(code="internal", message="camera down")))
        await robot.ping()  # 接続は生きている
    await asyncio.wait_for(task, 1.0)


async def test_robot_disconnecting_during_handshake(settings: ServerSettings) -> None:
    robot, task = await _start(settings)
    await robot.close()  # hello を送らずに切る
    await asyncio.wait_for(task, 1.0)


async def test_messages_handled_by_later_issues_are_accepted_quietly(
    settings: ServerSettings, token: str
) -> None:
    """記憶操作の ack・ツール結果・忘却・退出は、担当の Issue（#21〜#24）まで受け取るだけ。"""
    from reachy_protocol.messages import MemoryAck, PersonForget, SessionClose, ToolResult

    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, fake_providers()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        for msg in [
            MemoryAck(op_id="o1", result="applied"),
            ToolResult(call_id="c1", ok=True),
            PersonForget(person_id="p1"),
            SessionClose(person_id="p1"),
        ]:
            await robot_end.send(dump_message(msg))
        await robot.ping()  # 接続は生きていて、エラーも返っていない
        assert [m for m in robot.received if isinstance(m, Error)] == []
    await asyncio.wait_for(task, 1.0)
