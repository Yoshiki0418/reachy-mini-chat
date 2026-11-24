"""WebSocket の Transport を、実際の WebSocket（ローカルの echo サーバー）で確かめる。websockets が無ければ飛ばす。"""

from __future__ import annotations

import pytest

from reachy_front.client import connect_websocket
from reachy_protocol.transport import TransportClosed

websockets_server = pytest.importorskip("websockets.asyncio.server")


async def test_text_and_binary_frames_roundtrip_and_close_is_reported() -> None:
    async def echo(ws: object) -> None:
        async for message in ws:  # type: ignore[attr-defined]
            await ws.send(message)  # type: ignore[attr-defined]

    async with websockets_server.serve(echo, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        t = await connect_websocket(f"ws://127.0.0.1:{port}/v1/session")
        await t.send('{"type": "ping"}')
        assert await t.recv() == '{"type": "ping"}'
        await t.send(b"\x00\x01binary")
        assert await t.recv() == b"\x00\x01binary"
        await t.close()
        await t.close()  # 何度でも
        with pytest.raises(TransportClosed):
            await t.recv()
        with pytest.raises(TransportClosed):
            await t.send("x")


async def test_connection_refused_is_an_oserror() -> None:
    with pytest.raises(OSError):
        await connect_websocket("ws://127.0.0.1:1/v1/session", open_timeout_s=1.0)
