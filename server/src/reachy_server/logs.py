"""ログの設定。起動の入口（``python -m reachy_server``）で 1 回呼ぶ。

uvicorn は自分のロガーしか設定しないので、何もしないと ``reachy_server.*`` の INFO
（ターンのトレース ``reachy_server.trace`` など）は出ない。
"""

from __future__ import annotations

import logging

# 外部ライブラリは、INFO だと 1 回ごとのリクエストをすべて出してうるさい
_QUIET = ("httpx", "httpcore", "openai", "urllib3", "asyncio")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,  # 先に誰かが設定していても、このアプリの設定にする
    )
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
