"""WebSocket の ``Transport``（本番用）。``websockets`` は実機の環境にあるものを使い、import は遅延。

実機でまだ確認していない点: apps_venv の ``websockets`` の版（``websockets.asyncio`` は 13 以上）。
"""

from __future__ import annotations

from typing import Any

from reachy_protocol.transport import Frame, TransportClosed


class WebSocketClientTransport:
    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send(self, frame: Frame) -> None:
        try:
            await self._ws.send(frame)
        except Exception as e:
            raise TransportClosed from e

    async def recv(self) -> Frame:
        try:
            data = await self._ws.recv()
        except Exception as e:
            raise TransportClosed from e
        return data

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:  # すでに閉じている
            return


async def connect_websocket(url: str, *, open_timeout_s: float = 5.0) -> WebSocketClientTransport:
    """``ws://host:port/v1/session`` に繋ぐ。繋がらなければ OSError など。"""
    from websockets.asyncio.client import connect  # pyright: ignore[reportMissingImports]

    ws = await connect(url, max_size=None, open_timeout=open_timeout_s, ping_interval=None)
    return WebSocketClientTransport(ws)


__all__ = ["WebSocketClientTransport", "connect_websocket"]
