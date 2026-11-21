"""FastAPI アプリ。WS /v1/session と /healthz。

起動は ``python -m reachy_server``（設定の host / port で待ち受ける。:mod:`reachy_server.__main__`）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from reachy_protocol.version import PROTOCOL_VERSION

from .config import ServerSettings
from .history import HistoryStore
from .providers import Providers, build_providers
from .session import Session
from .ws import WebSocketTransport

log = logging.getLogger(__name__)


def create_app(
    settings: ServerSettings | None = None,
    providers: Providers | None = None,
    history: HistoryStore | None = None,
) -> FastAPI:
    settings = settings or ServerSettings()  # pyright: ignore[reportCallIssue]  # 環境変数から読む
    log.info("server settings: %s", settings.masked())
    # モデルは起動時に 1 回だけ用意し、全セッションで共有する（未知の名前ならここで起動を止める）
    providers = providers or build_providers(settings)
    store = history or HistoryStore(settings.history_db or None)
    retention = settings.history_retention_days

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await store.checkpointer()
        # 重いモデルは起動時に読み込む（最初の発話で待たせない）。必要なプロバイダだけが持つ口
        warm = getattr(providers.tts, "warm", None)
        if warm is not None:
            await warm()
        pruner = asyncio.create_task(store.prune_forever(retention))
        try:
            yield
        finally:
            pruner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pruner
            await store.close()
            close = getattr(providers.tts, "close", None)
            if close is not None:
                await close()

    app = FastAPI(title="reachy-server", version=settings.server_version, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {
            "status": "ok",
            "server_version": settings.server_version,
            "protocol_version": PROTOCOL_VERSION,
        }

    @app.websocket("/v1/session")
    async def session(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        await ws.accept()
        await Session(WebSocketTransport(ws), settings, providers, store).run()

    return app
