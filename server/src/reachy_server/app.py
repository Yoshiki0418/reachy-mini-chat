"""FastAPI アプリ。WS /v1/session と /healthz。

起動は ``python -m reachy_server``（設定の host / port で待ち受ける。:mod:`reachy_server.__main__`）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket

from reachy_protocol.version import PROTOCOL_VERSION

from .config import ServerSettings
from .session import Session
from .ws import WebSocketTransport

log = logging.getLogger(__name__)


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings()  # pyright: ignore[reportCallIssue]  # 環境変数から読む
    log.info("server settings: %s", settings.masked())
    app = FastAPI(title="reachy-server", version=settings.server_version)

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
        await Session(WebSocketTransport(ws), settings).run()

    return app
