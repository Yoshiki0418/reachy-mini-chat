"""WebSocketTransport: Starlette の WebSocket を Transport に合わせるアダプタの境界条件。"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from reachy_protocol.transport import TransportClosed
from reachy_server.ws import WebSocketTransport


class StubWebSocket:
    def __init__(
        self, incoming: list[dict[str, Any]] | None = None, fail_send: bool = False
    ) -> None:
        self.incoming = incoming or []
        self.sent: list[tuple[str, Any]] = []
        self.fail_send = fail_send
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.fail_send:
            raise WebSocketDisconnect()
        self.sent.append(("text", data))

    async def send_bytes(self, data: bytes) -> None:
        if self.fail_send:
            raise RuntimeError("closed")
        self.sent.append(("bytes", data))

    async def receive(self) -> dict[str, Any]:
        if not self.incoming:
            raise RuntimeError("no more")
        return self.incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


def make(ws: StubWebSocket) -> WebSocketTransport:
    return WebSocketTransport(ws)  # type: ignore[arg-type]


async def test_send_text_and_bytes() -> None:
    ws = StubWebSocket()
    t = make(ws)
    await t.send("hi")
    await t.send(b"\x01")
    assert ws.sent == [("text", "hi"), ("bytes", b"\x01")]


async def test_recv_text_bytes_and_disconnect() -> None:
    ws = StubWebSocket(
        [
            {"type": "websocket.receive", "text": "hi"},
            {"type": "websocket.receive", "bytes": b"\x02"},
            {"type": "websocket.disconnect", "code": 1000},
        ]
    )
    t = make(ws)
    assert await t.recv() == "hi"
    assert await t.recv() == b"\x02"
    with pytest.raises(TransportClosed):
        await t.recv()
    with pytest.raises(TransportClosed):  # 閉じたら以降も閉じたまま
        await t.recv()
    with pytest.raises(TransportClosed):
        await t.send("x")


@pytest.mark.parametrize("frame", ["text", b"bytes"])
async def test_send_failure_becomes_transport_closed(frame: str | bytes) -> None:
    t = make(StubWebSocket(fail_send=True))
    with pytest.raises(TransportClosed):
        await t.send(frame)


async def test_receive_failure_becomes_transport_closed() -> None:
    t = make(StubWebSocket([]))
    with pytest.raises(TransportClosed):
        await t.recv()


async def test_close_is_idempotent() -> None:
    ws = StubWebSocket()
    t = make(ws)
    await t.close()
    await t.close()
    assert ws.closed
    with pytest.raises(TransportClosed):
        await t.recv()
