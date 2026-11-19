"""1 接続（= 1 台のロボットとの会話セッション）の処理。設計書 §4.1。

Transport の上で動くので、テストではメモリ上の通信路、本番では WebSocket でつなぐ。
この段階ではハンドシェイクと死活監視まで。発話とターンは #16 で追加する。
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging

from reachy_protocol.channel import BinaryFrame, Endpoint, Incoming
from reachy_protocol.messages import Error, Hello, Ping, Pong, Welcome
from reachy_protocol.transport import Transport, TransportClosed
from reachy_protocol.version import PROTOCOL_VERSION, is_compatible

from .config import ServerSettings
from .providers import Providers

log = logging.getLogger(__name__)


class Session:
    def __init__(
        self, transport: Transport, settings: ServerSettings, providers: Providers
    ) -> None:
        self.ep = Endpoint(transport, "server")
        self.settings = settings
        self.providers = providers
        self.robot_id: str | None = None

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
        await self.ep.send(
            Welcome(server_version=self.settings.server_version, models=self.providers.describe())
        )
        log.info("session open robot_id=%s protocol=%s", first.robot_id, first.protocol_version)
        return True

    async def _loop(self) -> None:
        while True:
            try:
                item = await self.ep.recv()
            except ValueError as e:  # 不正なフレームは知らせて接続は維持する
                await self.ep.send(Error(code="bad_message", message=str(e)))
                continue
            await self._dispatch(item)

    async def _dispatch(self, item: Incoming) -> None:
        if isinstance(item, Ping):
            await self.ep.send(Pong(nonce=item.nonce))
            return
        if isinstance(item, Error):
            log.warning("robot reported error: %s", item)
            return
        if isinstance(item, Hello):
            await self.ep.send(Error(code="bad_message", message="hello は接続時に 1 回だけ"))
            return
        # 発話・ターンなどは #16 以降で実装する
        what = f"binary {item.header.kind}" if isinstance(item, BinaryFrame) else item.type
        ref = item.header.ref if isinstance(item, BinaryFrame) else None
        await self.ep.send(Error(code="internal", message=f"{what} は未実装", ref=ref))

    async def _send_quietly(self, msg: Error) -> None:
        with contextlib.suppress(TransportClosed):
            await self.ep.send(msg)
