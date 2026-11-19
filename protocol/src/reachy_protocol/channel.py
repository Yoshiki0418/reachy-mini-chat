"""型付きの送受信口。Transport の上で、メッセージとバイナリフレームを方向つきで読み書きする。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from .framing import BinaryHeader, decode_frame, encode_frame
from .messages import (
    Message,
    MessageDirectionError,
    dump_message,
    parse_from_robot,
    parse_from_server,
)
from .transport import Transport

__all__ = ["FRAME_KINDS_FROM", "BinaryFrame", "Endpoint", "Incoming", "Role"]

Role: TypeAlias = Literal["robot", "server"]

FRAME_KINDS_FROM: dict[Role, frozenset[str]] = {
    "robot": frozenset({"audio_in", "frame"}),
    "server": frozenset({"audio_out"}),
}
"""各役割が送ってよいバイナリフレームの kind。"""


@dataclass(frozen=True)
class BinaryFrame:
    header: BinaryHeader
    payload: bytes


Incoming: TypeAlias = Message | BinaryFrame


class Endpoint:
    """自分の役割（robot / server）を知っている送受信口。

    受信時は相手の方向で検証する（robot なら parse_from_server）。
    不正なフレームは ValidationError / MessageDirectionError / FramingError を送出する
    （どれも ValueError のサブクラス）。
    """

    def __init__(self, transport: Transport, role: Role) -> None:
        self.transport = transport
        self.role: Role = role
        self._parse = parse_from_server if role == "robot" else parse_from_robot
        self._peer: Role = "server" if role == "robot" else "robot"

    async def send(self, msg: Message) -> None:
        await self.transport.send(dump_message(msg))

    async def send_frame(self, header: BinaryHeader, payload: bytes) -> None:
        await self.transport.send(encode_frame(header, payload))

    async def recv(self) -> Incoming:
        frame = await self.transport.recv()
        if isinstance(frame, bytes):
            header, payload = decode_frame(frame)
            if header.kind not in FRAME_KINDS_FROM[self._peer]:
                raise MessageDirectionError(f"{self._peer} は binary {header.kind!r} を送れない")
            return BinaryFrame(header, payload)
        return self._parse(frame)

    async def close(self) -> None:
        await self.transport.close()
