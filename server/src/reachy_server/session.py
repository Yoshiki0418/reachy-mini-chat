"""1 接続（= 1 台のロボットとの会話セッション）の処理。設計書 §4.1。

Transport の上で動くので、テストではメモリ上の通信路、本番では WebSocket でつなぐ。
ハンドシェイクと死活監視はここで、発話とターンは :class:`Conversation` が担う。
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
from collections.abc import Callable

from reachy_protocol.channel import Endpoint, Incoming
from reachy_protocol.framing import BinaryHeader
from reachy_protocol.messages import (
    Error,
    Hello,
    MemoryAck,
    Message,
    PersonForget,
    Ping,
    Pong,
    SessionClose,
    ToolResult,
    Welcome,
)
from reachy_protocol.transport import Transport, TransportClosed
from reachy_protocol.version import PROTOCOL_VERSION, is_compatible

from .config import ServerSettings
from .conversation import Conversation, ConversationLimits
from .providers import Providers
from .responder import LLMResponder, Responder

log = logging.getLogger(__name__)

ResponderFactory = Callable[[Providers], Responder]


class Session:
    def __init__(
        self,
        transport: Transport,
        settings: ServerSettings,
        providers: Providers,
        responder_factory: ResponderFactory = lambda p: LLMResponder(p.llm),
    ) -> None:
        self.ep = Endpoint(transport, "server")
        self.settings = settings
        self.providers = providers
        self.robot_id: str | None = None
        # 文字起こし・ターン・受信ループが同じ接続に送るので、送信は 1 本ずつに直列化する
        self._send_lock = asyncio.Lock()
        self.conversation = Conversation(
            send=self.send,
            send_frame=self.send_frame,
            stt=providers.stt,
            tts=providers.tts,
            responder=responder_factory(providers),
            limits=ConversationLimits(
                decision_timeout_s=settings.decision_timeout_s,
                max_utterance_s=settings.max_utterance_s,
            ),
        )

    async def send(self, msg: Message) -> None:
        async with self._send_lock:
            await self.ep.send(msg)

    async def send_frame(self, header: BinaryHeader, payload: bytes) -> None:
        async with self._send_lock:
            await self.ep.send_frame(header, payload)

    async def run(self) -> None:
        """接続が閉じるまで処理する。どんな失敗でも例外を外へ漏らさず接続を閉じる。"""
        try:
            if await self._handshake():
                await self._loop()
        except TransportClosed:
            pass
        except Exception:  # 想定外の失敗で他のセッションを巻き込まない
            log.exception("session crashed robot_id=%s", self.robot_id)
            await self._send_quietly(Error(code="internal"))
        finally:
            await self.conversation.close()
            await self.ep.close()
            log.info("session closed robot_id=%s", self.robot_id)

    async def _handshake(self) -> bool:
        try:
            first = await asyncio.wait_for(self.ep.recv(), self.settings.handshake_timeout_s)
        except TimeoutError:
            await self._send_quietly(Error(code="timeout", message="hello が届かない"))
            return False
        except ValueError as e:  # 不正なフレーム
            await self._send_quietly(Error(code="bad_message", message=str(e)))
            return False
        if not isinstance(first, Hello):
            await self._send_quietly(Error(code="bad_message", message="最初のメッセージは hello"))
            return False
        if not is_compatible(first.protocol_version):
            await self._send_quietly(
                Error(
                    code="incompatible_version",
                    message=f"robot={first.protocol_version} server={PROTOCOL_VERSION}",
                )
            )
            return False
        expected = self.settings.token.get_secret_value().encode()
        if not hmac.compare_digest(first.token.get_secret_value().encode(), expected):
            log.warning("unauthorized robot_id=%s", first.robot_id)
            await self._send_quietly(Error(code="unauthorized"))
            return False
        self.robot_id = first.robot_id
        await self.send(
            Welcome(server_version=self.settings.server_version, models=self.providers.describe())
        )
        log.info("session open robot_id=%s protocol=%s", first.robot_id, first.protocol_version)
        return True

    async def _loop(self) -> None:
        while True:
            try:
                item = await self.ep.recv()
            except ValueError as e:  # 不正なフレームは知らせて接続は維持する
                await self.send(Error(code="bad_message", message=str(e)))
                continue
            await self._dispatch(item)

    async def _dispatch(self, item: Incoming) -> None:
        if isinstance(item, Ping):
            await self.send(Pong(nonce=item.nonce))
        elif isinstance(item, Error):
            log.warning("robot reported error: %s", item)
        elif isinstance(item, Hello):
            await self.send(Error(code="bad_message", message="hello は接続時に 1 回だけ"))
        elif await self.conversation.handle(item):
            pass
        elif isinstance(item, MemoryAck | ToolResult | PersonForget | SessionClose):
            # 記憶操作（#22）・ツール RPC（#23）・忘却（#21）・要約（#24）で扱う。今は受け取るだけ
            log.debug("accepted (not yet handled): %s", item)
        else:  # pragma: no cover - 型の上では到達しない
            await self.send(Error(code="bad_message", message=f"未対応のメッセージ: {item}"))

    async def _send_quietly(self, msg: Error) -> None:
        with contextlib.suppress(TransportClosed):
            await self.send(msg)
