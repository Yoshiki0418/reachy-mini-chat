"""FastAPI（Starlette）の WebSocket を reachy_protocol の Transport に合わせるアダプタ。"""

from __future__ import annotations

import contextlib

from fastapi import WebSocket, WebSocketDisconnect

from reachy_protocol.transport import Frame, TransportClosed


class WebSocketTransport:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._closed = False

    async def send(self, frame: Frame) -> None:
        if self._closed:
            raise TransportClosed
        try:
            if isinstance(frame, bytes):
                await self.ws.send_bytes(frame)
            else:
                await self.ws.send_text(frame)
        except (WebSocketDisconnect, RuntimeError) as e:
            self._closed = True
            raise TransportClosed from e

    async def recv(self) -> Frame:
        if self._closed:
            raise TransportClosed
        try:
            msg = await self.ws.receive()
        except (WebSocketDisconnect, RuntimeError) as e:
            self._closed = True
            raise TransportClosed from e
        if msg["type"] == "websocket.disconnect":
            self._closed = True
            raise TransportClosed
        data = msg.get("bytes")
        if data is not None:
            return bytes(data)
        return str(msg.get("text") or "")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self.ws.close()
