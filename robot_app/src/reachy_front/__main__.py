"""起動の入口。

/venvs/apps_venv/bin/python -m reachy_front             # 設定のダンプ（本体の会話は #35）
/venvs/apps_venv/bin/python -m reachy_front check-io    # 入出力の実機チェック（tools/check_io.py）
/venvs/apps_venv/bin/python -m reachy_front check-link  # サーバーとの接続チェック（tools/check_link.py）
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from .core import FrontSettings

log = logging.getLogger("reachy_front")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args[:1] == ["check-io"]:
        return asyncio.run(_check_io())
    if args[:1] == ["check-link"]:
        return asyncio.run(_check_link())
    if args:
        print(f"知らないコマンド: {args[0]}（check-io / check-link）", file=sys.stderr)
        return 2
    settings = FrontSettings()
    log.info("settings: %s", json.dumps(settings.dump(), ensure_ascii=False))
    return 0


async def _check_io() -> int:
    """実機の入出力チェック（サーバーなし）。NG があれば 1 を返す。"""
    from .io import ReachyRobotIO
    from .tools.check_io import run_checks

    results = await run_checks(ReachyRobotIO())
    return 1 if any(r.status == "NG" for r in results) else 0


async def _check_link() -> int:
    """サーバーとの接続チェック（会話はしない）。NG があれば 1 を返す。"""
    from .tools.check_link import run_link_checks, targets_from_settings

    settings = FrontSettings()
    results = await run_link_checks(settings, targets_from_settings(settings))
    return 1 if any(r.status == "NG" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
