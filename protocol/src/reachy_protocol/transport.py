"""通信路の抽象。WebSocket の text / binary フレームをそのまま運ぶ。

本番ではロボット側・サーバー側がそれぞれ WebSocket ライブラリをこの口に合わせる（M2 / M3）。
テストではメモリ上の :func:`memory_pair` を使う。
"""

from __future__ import annotations

import asyncio
from typing import Protocol, TypeAlias

__all__ = ["Frame", "MemoryTransport", "Transport", "TransportClosed", "memory_pair"]

Frame: TypeAlias = str | bytes
"""str = text フレーム（JSON の制御メッセージ）、bytes = binary フレーム。"""


class TransportClosed(Exception):
    """相手か自分が通信路を閉じた。"""


class Transport(Protocol):
    async def send(self, frame: Frame) -> None:
        """1 フレーム送る。閉じていれば TransportClosed。"""
        ...

    async def recv(self) -> Frame:
        """1 フレーム受け取る。閉じていれば TransportClosed。"""
        ...

    async def close(self) -> None:
        """閉じる（何度呼んでもよい）。相手の recv も TransportClosed になる。"""
        ...


class _Closed:
    pass


_CLOSED = _Closed()


class MemoryTransport:
    """同一プロセス内でつながった通信路の片端。"""

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[Frame | _Closed] = asyncio.Queue()
        self._peer: MemoryTransport | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def send(self, frame: Frame) -> None:
        if self._closed or self._peer is None or self._peer._closed:
            raise TransportClosed
        self._peer._inbox.put_nowait(frame)

    async def recv(self) -> Frame:
        if self._closed and self._inbox.empty():
            raise TransportClosed
        item = await self._inbox.get()
        if isinstance(item, _Closed):
            self._closed = True
            self._inbox.put_nowait(item)  # 以降の recv も閉じたまま
            raise TransportClosed
        return item

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inbox.put_nowait(_CLOSED)
        if self._peer is not None and not self._peer._closed:
            self._peer._inbox.put_nowait(_CLOSED)


def memory_pair() -> tuple[MemoryTransport, MemoryTransport]:
    """つながった 2 つの端（ロボット側, サーバー側）を返す。"""
    a, b = MemoryTransport(), MemoryTransport()
    a._peer, b._peer = b, a  # pyright: ignore[reportPrivateUsage]
    return a, b
