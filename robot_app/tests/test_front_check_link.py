"""サーバーとの接続チェック。FakeServer（メモリ上）で、成功と、止まる段ごとの失敗を確かめる。"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from reachy_front.client import ClipFetcher
from reachy_front.core import FrontSettings
from reachy_front.tools.check_link import LinkTargets, run_link_checks
from reachy_protocol.clips import ClipInfo, ClipManifest
from reachy_protocol.testing.fake_server import FakeServer, FakeServerConfig
from reachy_protocol.transport import Transport, memory_pair

TOKEN = "check-link-token-0123456789"


class _Fetcher:
    async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
        wav = b"RIFFx"
        info = ClipInfo.model_validate(
            {
                "id": "greeting-hello",
                "kind": "greeting",
                "text": "こんにちは",
                "sha256": hashlib.sha256(wav).hexdigest(),
                "nbytes": len(wav),
            }
        )
        return ClipManifest(version="v1", clips=[info])

    async def clip(self, clip_id: str) -> bytes:
        return b"RIFFx"


def settings(monkeypatch: pytest.MonkeyPatch, token: str | None = TOKEN) -> FrontSettings:
    for k in ("REACHY_SERVER_TOKEN", "PC_HOST", "PC_PORT"):
        monkeypatch.delenv(k, raising=False)
    if token:
        monkeypatch.setenv("REACHY_SERVER_TOKEN", token)
    return FrontSettings()


_TASKS: set[asyncio.Task[None]] = set()


def targets(tmp_path: Path, server: FakeServer, *, healthz_ok: bool = True) -> LinkTargets:
    async def connect() -> Transport:
        robot_side, server_side = memory_pair()
        task = asyncio.create_task(server.serve(server_side))
        _TASKS.add(task)
        task.add_done_callback(_TASKS.discard)
        return robot_side

    async def healthz() -> str:
        if not healthz_ok:
            raise ConnectionRefusedError("refused")
        return "fake-0.1"

    fetcher: ClipFetcher = _Fetcher()
    return LinkTargets(healthz, connect, fetcher, str(tmp_path / "clips"))


async def test_all_stages_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []
    results = await run_link_checks(
        settings(monkeypatch),
        targets(tmp_path, FakeServer(FakeServerConfig(token=TOKEN))),
        out=lines.append,
        timeout_s=1.0,
    )
    assert [r.status for r in results] == ["OK", "OK", "OK", "OK"]
    assert (tmp_path / "clips" / "manifest.json").exists()  # クリップをキャッシュした
    assert TOKEN not in "\n".join(lines)  # トークンは表示しない


async def test_missing_token_stops_at_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = await run_link_checks(
        settings(monkeypatch, None),
        targets(tmp_path, FakeServer()),
        out=lambda _s: None,
        timeout_s=1.0,
    )
    assert [(r.name, r.status) for r in results] == [("設定", "NG")]
    assert "REACHY_SERVER_TOKEN" in results[0].detail


async def test_unreachable_server_stops_at_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = await run_link_checks(
        settings(monkeypatch),
        targets(tmp_path, FakeServer(FakeServerConfig(token=TOKEN)), healthz_ok=False),
        out=lambda _s: None,
        timeout_s=1.0,
    )
    assert [r.status for r in results] == ["OK", "NG"]
    assert "ファイアウォール" in results[1].detail


async def test_wrong_token_is_reported_at_the_handshake_quickly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    results = await run_link_checks(
        settings(monkeypatch),
        targets(tmp_path, FakeServer(FakeServerConfig(token="another-token-0123456789"))),
        out=lambda _s: None,
        timeout_s=5.0,
    )
    assert [r.status for r in results] == ["OK", "OK", "NG"]
    assert results[2].detail == "unauthorized"
    assert loop.time() - t0 < 2.0  # タイムアウトまで待たず、拒否された時点で止まる


# ---------------------------------------------------------------- 入口（python -m reachy_front ...）


def test_entrypoint_dispatches_the_check_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_front import __main__ as entry

    called: list[str] = []

    async def fake_io() -> int:
        called.append("io")
        return 0

    async def fake_link() -> int:
        called.append("link")
        return 1

    monkeypatch.setattr(entry, "_check_io", fake_io)
    monkeypatch.setattr(entry, "_check_link", fake_link)
    assert entry.main(["check-io"]) == 0
    assert entry.main(["check-link"]) == 1  # NG があれば終了コード 1
    assert called == ["io", "link"]
    assert entry.main(["nope"]) == 2


def test_entrypoint_without_arguments_dumps_the_settings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from reachy_front import __main__ as entry

    monkeypatch.setenv("REACHY_SERVER_TOKEN", TOKEN)
    with caplog.at_level("INFO", logger="reachy_front"):
        assert entry.main([]) == 0
    assert "settings:" in caplog.text and TOKEN not in caplog.text


# ---------------------------------------------------------------- クリップ取得の失敗は OK にしない（Codex の指摘 #133）


async def test_a_failed_clip_fetch_is_ng_even_when_a_local_cache_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reachy_front.client import ClipFetchError

    server = FakeServer(FakeServerConfig(token=TOKEN))
    ok = targets(tmp_path, server)
    await run_link_checks(
        settings(monkeypatch), ok, out=lambda _s: None, timeout_s=1.0
    )  # 先にキャッシュを作る

    class Failing:
        async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
            raise ClipFetchError("HTTP 503")

        async def clip(self, clip_id: str) -> bytes:
            raise AssertionError

    broken = LinkTargets(ok.healthz, ok.connect, Failing(), ok.clips_dir)
    results = await run_link_checks(
        settings(monkeypatch), broken, out=lambda _s: None, timeout_s=1.0
    )
    clips = results[-1]
    assert clips.name == "事前合成クリップの取得"
    assert clips.status == "NG"  # ローカルに前のキャッシュがあっても、取得に失敗したことは隠さない
    assert "503" in clips.detail and "ローカル" in clips.detail
