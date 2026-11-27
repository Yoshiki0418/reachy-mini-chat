"""起動の入口。今は設定を読んで起動時のダンプを出すところまで（本体は #34・#35 以降）。

/venvs/apps_venv/bin/python -m reachy_front
"""

from __future__ import annotations

import json
import logging

from .core import FrontSettings

log = logging.getLogger("reachy_front")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = FrontSettings()
    log.info("settings: %s", json.dumps(settings.dump(), ensure_ascii=False))


if __name__ == "__main__":
    main()


async def _check_link() -> int:
    """サーバーとの接続チェック（会話はしない）。NG があれば 1 を返す。"""
    from .tools.check_link import run_link_checks, targets_from_settings

    settings = FrontSettings()
    results = await run_link_checks(settings, targets_from_settings(settings))
    return 1 if any(r.status == "NG" for r in results) else 0
