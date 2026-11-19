"""メモリ上の通信路と、型付きの送受信口（方向の検査）。"""

from __future__ import annotations

import pytest

from reachy_protocol.channel import BinaryFrame, Endpoint
from reachy_protocol.framing import BinaryHeader, encode_frame
from reachy_protocol.messages import MessageDirectionError, Ping, TurnEnd, dump_message
from reachy_protocol.transport import TransportClosed, memory_pair


async def test_frames_flow_both_ways_in_order() -> None:
    a, b = memory_pair()
    await a.send("one")
    await a.send(b"two")
    await b.send("back")
    assert await b.recv() == "one"
    assert await b.recv() == b"two"
    assert await a.recv() == "back"


async def test_close_is_seen_by_peer_and_is_sticky() -> None:
    a, b = memory_pair()
    await a.send("last")
    await a.close()
    assert await b.recv() == "last"  # 閉じる前に送ったものは届く
    for _ in range(2):
        with pytest.raises(TransportClosed):
            await b.recv()
    with pytest.raises(TransportClosed):
        await b.send("x")
    with pytest.raises(TransportClosed):
        await a.send("x")
    await a.close()  # 2 回閉じてもよい


async def test_endpoint_round_trip() -> None:
    a, b = memory_pair()
    robot, server = Endpoint(a, "robot"), Endpoint(b, "server")
    ping = Ping(nonce="n")
    await robot.send(ping)
    assert await server.recv() == ping
    header = BinaryHeader(kind="audio_out", ref="t1", seq=0, format="wav_pcm16")
    await server.send_frame(header, b"RIFF")
    got = await robot.recv()
    assert got == BinaryFrame(header, b"RIFF")


async def test_endpoint_rejects_wrong_direction_message() -> None:
    a, b = memory_pair()
    server = Endpoint(b, "server")
    await a.send(dump_message(TurnEnd(turn_id="t1")))  # turn.end はサーバーしか送れない
    with pytest.raises(MessageDirectionError):
        await server.recv()


@pytest.mark.parametrize(
    ("receiver", "kind", "fmt"),
    [
        ("robot", "audio_in", "pcm_f32le"),
        ("robot", "frame", "jpeg"),
        ("server", "audio_out", "wav_pcm16"),
    ],
)
async def test_endpoint_rejects_wrong_direction_binary(receiver: str, kind: str, fmt: str) -> None:
    a, b = memory_pair()
    ep = Endpoint(b, "robot" if receiver == "robot" else "server")
    header = BinaryHeader.model_validate(
        {"kind": kind, "ref": "r", "seq": 0, "format": fmt, "sr": 16000}
    )
    await a.send(encode_frame(header, b""))
    with pytest.raises(MessageDirectionError):
        await ep.recv()


async def test_endpoint_refuses_to_send_wrong_direction_message() -> None:
    a, b = memory_pair()
    robot = Endpoint(a, "robot")
    with pytest.raises(MessageDirectionError):
        await robot.send(TurnEnd(turn_id="t1"))  # turn.end はサーバー専用
    await a.close()
    with pytest.raises(TransportClosed):
        await b.recv()  # 何も送られていない（閉じた通知だけが届く）


@pytest.mark.parametrize(
    ("sender", "kind", "fmt"),
    [
        ("robot", "audio_out", "wav_pcm16"),
        ("server", "audio_in", "pcm_f32le"),
        ("server", "frame", "jpeg"),
    ],
)
async def test_endpoint_refuses_to_send_wrong_direction_binary(
    sender: str, kind: str, fmt: str
) -> None:
    a, b = memory_pair()
    ep = Endpoint(a, "robot" if sender == "robot" else "server")
    header = BinaryHeader.model_validate(
        {"kind": kind, "ref": "r", "seq": 0, "format": fmt, "sr": 16000}
    )
    with pytest.raises(MessageDirectionError):
        await ep.send_frame(header, b"")
    await a.close()
    with pytest.raises(TransportClosed):
        await b.recv()
