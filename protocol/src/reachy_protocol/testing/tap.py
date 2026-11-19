"""通信を横から観測して ConversationChecker に渡す Transport ラッパー。"""

from __future__ import annotations

from ..channel import BinaryFrame, Incoming, Role
from ..conformance import ConversationChecker, Direction
from ..framing import decode_frame
from ..messages import parse_message
from ..transport import Frame, Transport


class MonitoredTransport:
    """片端（``side``）に挟み、送受信のすべてを checker に観測させる。

    フレームは方向を問わずに解釈するので、相手が逆方向のメッセージを送った場合も
    checker が違反として記録できる。解釈できないフレームも違反として記録する。
    """

    def __init__(self, inner: Transport, checker: ConversationChecker, side: Role) -> None:
        self.inner = inner
        self.checker = checker
        self._out: Direction = "robot->server" if side == "robot" else "server->robot"
        self._in: Direction = "server->robot" if side == "robot" else "robot->server"

    def _observe(self, direction: Direction, frame: Frame) -> None:
        try:
            item: Incoming
            if isinstance(frame, bytes):
                header, payload = decode_frame(frame)
                item = BinaryFrame(header, payload)
            else:
                item = parse_message(frame)
        except ValueError as e:
            self.checker.violations.append(f"{direction} に解釈できないフレーム: {e}")
            return
        self.checker.observe(direction, item)

    async def send(self, frame: Frame) -> None:
        self._observe(self._out, frame)
        await self.inner.send(frame)

    async def recv(self) -> Frame:
        frame = await self.inner.recv()
        self._observe(self._in, frame)
        return frame

    async def close(self) -> None:
        await self.inner.close()
