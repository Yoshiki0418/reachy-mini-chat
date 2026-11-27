"""サーバーとの接続チェック。ロボット上で 1 コマンドで流し、どの段で止まるかを表で出す。

    PC_HOST=<サーバーのホスト名か IP> REACHY_SERVER_TOKEN=<トークン> \\
        /venvs/apps_venv/bin/python -m reachy_front check-link

1. 設定（ホスト・ポート・トークンの有無）  2. HTTP で届くか（/healthz）  3. WebSocket のハンドシェイク
（hello → welcome。トークンとプロトコル版）  4. 事前合成クリップの取得とキャッシュ（#26）
音声の会話はしない（会話は #35）。トークンの値は表示しない。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import SecretStr

from reachy_protocol.transport import Transport

from ..client import (
    ClipCache,
    ClipFetcher,
    HttpClipFetcher,
    LinkConfig,
    ServerClient,
    connect_websocket,
)
from ..core import EventBus, FrontSettings
from .check_io import CheckResult, Out


@dataclass
class LinkTargets:
    """チェックの対象。テストでは差し替える。"""

    healthz: Callable[[], Awaitable[str]]
    connect: Callable[[], Awaitable[Transport]]
    fetcher: ClipFetcher | None
    clips_dir: str


def _http_healthz(base_url: str) -> Callable[[], Awaitable[str]]:
    def get() -> str:
        with urllib.request.urlopen(base_url + "/healthz", timeout=5) as res:
            return json.loads(res.read()).get("server_version", "?")

    async def run() -> str:
        return await asyncio.to_thread(get)

    return run


def targets_from_settings(s: FrontSettings) -> LinkTargets:
    base = s.server.http_url
    token = s.server.token.get_secret_value() if s.server.token else ""
    return LinkTargets(
        healthz=_http_healthz(base),
        connect=lambda: connect_websocket(s.server.session_url),
        fetcher=HttpClipFetcher(base, token) if token else None,
        clips_dir=s.clips.dir,
    )


async def _ready_or_fatal(client: ServerClient) -> None:
    while not client.ready.is_set() and client.fatal is None:
        await asyncio.sleep(0.02)


async def run_link_checks(
    settings: FrontSettings, targets: LinkTargets, *, out: Out = print, timeout_s: float = 5.0
) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(name: str, status: str, detail: str) -> bool:
        results.append(CheckResult(name, status, detail))
        out(f"[{status}] {name}: {detail}")
        return status != "NG"

    srv = settings.server
    token_ok = srv.token is not None
    add(
        "設定",
        "OK" if token_ok else "NG",
        f"接続先 {srv.host}:{srv.port}、ロボット ID {srv.robot_id}、"
        + (
            "トークンあり"
            if token_ok
            else "REACHY_SERVER_TOKEN が未設定（16 文字以上、サーバーと同じ値）"
        ),
    )
    if not token_ok:
        return results

    try:
        version = await asyncio.wait_for(targets.healthz(), timeout_s + 1)
        add("HTTP で届く（/healthz）", "OK", f"サーバー {version}")
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as e:
        add(
            "HTTP で届く（/healthz）",
            "NG",
            f"{type(e).__name__}: {e}（ホスト名・ポート・ファイアウォールを確認）",
        )
        return results

    bus = EventBus()
    assert srv.token is not None
    client = ServerClient(
        targets.connect,
        bus,
        LinkConfig(
            robot_id=srv.robot_id,
            token=SecretStr(srv.token.get_secret_value()),
            connect_timeout_s=timeout_s,
            handshake_timeout_s=timeout_s,
        ),
    )
    client.start()
    try:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(_ready_or_fatal(client), timeout_s + 2)
        if client.ready.is_set():
            add(
                "WebSocket のハンドシェイク",
                "OK",
                f"welcome を受信（サーバー {client.server_version}）",
            )
        else:
            why = (
                client.fatal
                or "welcome が返らない（websockets の版・プロトコル版・サーバーのログを確認）"
            )
            add("WebSocket のハンドシェイク", "NG", why)
            return results
    finally:
        await client.close()
        await bus.close()

    if targets.fetcher is None:
        return results
    cache = ClipCache(targets.clips_dir, targets.fetcher)
    updated = await cache.refresh()
    n = cache.load()
    add(
        "事前合成クリップの取得",
        "OK" if n > 0 else "NG",
        f"{n} 件を {targets.clips_dir} に保存（版 {cache.version}、{'新しく取得' if updated else '変更なし'}）",
    )
    out(f"\n結果: NG {sum(r.status == 'NG' for r in results)} 件")
    return results


__all__ = ["LinkTargets", "run_link_checks", "targets_from_settings"]
