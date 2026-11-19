"""起動の入口。設定（REACHY_SERVER_HOST / PORT など）どおりに uvicorn を起動する。

uv run python -m reachy_server
"""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import ServerSettings


def main() -> None:
    settings = ServerSettings()  # pyright: ignore[reportCallIssue]  # 環境変数から読む
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
