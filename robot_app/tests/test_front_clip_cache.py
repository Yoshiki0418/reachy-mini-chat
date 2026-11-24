"""サーバーの事前合成クリップのキャッシュ。ディスク・版・検証・不正な ID・HTTP 取得。"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from reachy_front.client import ClipCache, ClipFetchError, HttpClipFetcher
from reachy_front.client.clips import MAX_CLIP_BYTES
from reachy_front.core import EventBus
from reachy_front.core import events as ev
from reachy_protocol.clips import ClipInfo, ClipManifest


def info(clip_id: str, wav: bytes, kind: str = "filler") -> ClipInfo:
    return ClipInfo.model_validate(
        {
            "id": clip_id,
            "kind": kind,
            "text": clip_id,
            "emotion": "neutral",
            "sha256": hashlib.sha256(wav).hexdigest(),
            "nbytes": len(wav),
        }
    )


class FakeFetcher:
    def __init__(self, version: str, clips: dict[str, bytes]) -> None:
        self.version = version
        self.clips = clips
        self.kinds: dict[str, str] = {}
        self.manifest_calls: list[str | None] = []
        self.clip_calls: list[str] = []
        self.fail_on: str | None = None
        self.tamper: set[str] = set()

    def set(self, version: str, clips: dict[str, bytes]) -> None:
        self.version, self.clips = version, clips

    async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
        self.manifest_calls.append(if_none_match)
        if if_none_match == self.version:
            return None
        return ClipManifest(
            version=self.version,
            clips=[info(i, w, self.kinds.get(i, "filler")) for i, w in self.clips.items()],
        )

    async def clip(self, clip_id: str) -> bytes:
        self.clip_calls.append(clip_id)
        if clip_id == self.fail_on:
            raise ClipFetchError("boom")
        wav = self.clips[clip_id]
        return wav + b"x" if clip_id in self.tamper else wav


# ---------------------------------------------------------------- 取得と保存


async def test_refresh_downloads_persists_and_loads_offline(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa", "greeting-hello": b"RIFFh"})
    fetcher.kinds["greeting-hello"] = "greeting"
    cache = ClipCache(tmp_path, fetcher)
    assert await cache.refresh() is True
    assert cache.version == "v1"
    assert [c.info.id for c in cache.of_kind("greeting")] == ["greeting-hello"]
    # サーバーなしで起動し直しても読める（フォールバックの要）
    offline = ClipCache(tmp_path)
    assert offline.load() == 2
    clip = offline.get("greeting-hello")
    assert clip is not None and clip.wav == b"RIFFh"
    assert offline.version == "v1"


async def test_unchanged_version_downloads_nothing(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa"})
    cache = ClipCache(tmp_path, fetcher)
    await cache.refresh()
    fetcher.clip_calls.clear()
    assert await cache.refresh() is False
    assert fetcher.manifest_calls[-1] == "v1"  # If-None-Match に版を付ける
    assert fetcher.clip_calls == []


async def test_new_version_fetches_only_changed_clips_and_removes_stale_files(
    tmp_path: Path,
) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa", "filler-b": b"RIFFb"})
    cache = ClipCache(tmp_path, fetcher)
    await cache.refresh()
    fetcher.clip_calls.clear()
    fetcher.set("v2", {"filler-a": b"RIFFa", "filler-c": b"RIFFc"})  # a は同じ、b は消え、c は新規
    assert await cache.refresh() is True
    assert fetcher.clip_calls == ["filler-c"]
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == ["filler-a.wav", "filler-c.wav"]
    assert {c.info.id for c in cache.of_kind("filler")} == {"filler-a", "filler-c"}
    assert ClipCache(tmp_path).load() == 2


# ---------------------------------------------------------------- 失敗しても前の版が残る


async def test_a_failure_midway_keeps_the_previous_version(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa"})
    cache = ClipCache(tmp_path, fetcher)
    await cache.refresh()
    fetcher.set("v2", {"filler-a": b"RIFFA2", "filler-b": b"RIFFb"})
    fetcher.fail_on = "filler-b"
    assert await cache.refresh() is False
    assert cache.version == "v1"
    clip = cache.get("filler-a")
    assert clip is not None and clip.wav == b"RIFFa"
    again = ClipCache(tmp_path)
    assert again.load() == 1 and again.version == "v1"  # ディスクも前の版のまま


async def test_audio_that_does_not_match_the_manifest_is_rejected(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa"})
    fetcher.tamper = {"filler-a"}  # 一覧のハッシュと違う中身
    cache = ClipCache(tmp_path, fetcher)
    assert await cache.refresh() is False
    assert cache.version is None and not list(tmp_path.glob("*"))


async def test_unreachable_server_is_not_an_exception(tmp_path: Path) -> None:
    class Down:
        async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
            raise ClipFetchError("ConnectionRefusedError")

        async def clip(self, clip_id: str) -> bytes:
            raise AssertionError

    assert await ClipCache(tmp_path, Down()).refresh() is False
    assert await ClipCache(tmp_path).refresh() is False  # 取得先なし


# ---------------------------------------------------------------- 不正な入力


async def test_unsafe_ids_and_oversized_clips_are_ignored(tmp_path: Path) -> None:
    cache_dir = tmp_path / "clips"
    evil = {"../evil": b"RIFFx", "UPPER": b"RIFFx", "a/b": b"RIFFx", "ok-1": b"RIFFok"}
    fetcher = FakeFetcher("v1", evil)
    cache = ClipCache(cache_dir, fetcher)
    assert await cache.refresh() is True
    assert [c.info.id for c in cache.of_kind("filler")] == ["ok-1"]
    assert fetcher.clip_calls == ["ok-1"]  # 不正な ID は取りにも行かない
    assert not (tmp_path / "evil.wav").exists()  # 外には書かない
    assert sorted(p.name for p in cache_dir.iterdir()) == ["manifest.json", "ok-1.wav"]


async def test_too_many_clips_are_rejected(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {f"c{i}": b"RIFF" for i in range(65)})
    assert await ClipCache(tmp_path, fetcher).refresh() is False


async def test_oversized_clip_is_skipped(tmp_path: Path) -> None:
    class Big(FakeFetcher):
        async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
            return ClipManifest(
                version="v1",
                clips=[
                    ClipInfo(
                        id="big",
                        kind="filler",
                        text="x",
                        sha256="0" * 64,
                        nbytes=MAX_CLIP_BYTES + 1,
                    )
                ],
            )

    fetcher = Big("v1", {})
    cache = ClipCache(tmp_path, fetcher)
    assert await cache.refresh() is True
    assert cache.get("big") is None and fetcher.clip_calls == []


def test_load_skips_corrupted_files_and_a_missing_manifest(tmp_path: Path) -> None:
    assert ClipCache(tmp_path).load() == 0  # 何も無い
    good, bad = b"RIFFgood", b"RIFFbad"
    manifest = ClipManifest(version="v1", clips=[info("good", good), info("bad", bad)])
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    (tmp_path / "good.wav").write_bytes(good)
    (tmp_path / "bad.wav").write_bytes(b"RIFFtampered")  # ハッシュが合わない
    cache = ClipCache(tmp_path)
    assert cache.load() == 1
    assert cache.get("good") is not None and cache.get("bad") is None
    (tmp_path / "manifest.json").write_text("{not json")
    assert ClipCache(tmp_path).load() == 0


async def test_concurrent_refreshes_are_serialized(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa"})
    cache = ClipCache(tmp_path, fetcher)
    results = await asyncio.gather(cache.refresh(), cache.refresh(), cache.refresh())
    assert sorted(results) == [False, False, True]
    assert fetcher.clip_calls == ["filler-a"]  # 1 回しか取らない


# ---------------------------------------------------------------- バス


async def test_reconnecting_triggers_a_refresh(tmp_path: Path) -> None:
    fetcher = FakeFetcher("v1", {"filler-a": b"RIFFa"})
    cache = ClipCache(tmp_path, fetcher)
    bus = EventBus()
    cache.attach(bus)
    bus.publish(ev.ServerConnectionChanged(connected=False))
    await bus.drain(1)
    assert fetcher.manifest_calls == []  # 切れたときは何もしない
    bus.publish(ev.ServerConnectionChanged(connected=True))
    await bus.drain(1)
    for _ in range(100):
        if cache.version == "v1":
            break
        await asyncio.sleep(0.01)
    assert cache.version == "v1"
    await bus.close()


# ---------------------------------------------------------------- HTTP（標準ライブラリのサーバーで）


class _Server:
    def __init__(self, token: str, version: str, clips: dict[str, bytes]) -> None:
        outer = self
        self.requests: list[tuple[str, str | None, str | None]] = []
        manifest = ClipManifest(version=version, clips=[info(i, w) for i, w in clips.items()])

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: object) -> None:
                return

            def do_GET(self) -> None:
                outer.requests.append(
                    (
                        self.path,
                        self.headers.get("Authorization"),
                        self.headers.get("If-None-Match"),
                    )
                )
                if self.headers.get("Authorization") != f"Bearer {token}":
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path == "/v1/clips":
                    if self.headers.get("If-None-Match") == f'"{version}"':
                        self.send_response(304)
                        self.end_headers()
                        return
                    body = manifest.model_dump_json().encode()
                elif self.path.startswith("/v1/clips/") and self.path[10:] in clips:
                    body = clips[self.path[10:]]
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def http_server() -> Iterator[_Server]:
    server = _Server("secret-token-0123456789", "v7", {"filler-a": b"RIFFa", "notice-x": b"RIFFn"})
    yield server
    server.stop()


async def test_http_fetcher_end_to_end(tmp_path: Path, http_server: _Server) -> None:
    cache = ClipCache(tmp_path, HttpClipFetcher(http_server.url, "secret-token-0123456789"))
    assert await cache.refresh() is True and cache.version == "v7"
    assert await cache.refresh() is False  # 304
    assert http_server.requests[-1][2] == '"v7"'
    assert all(a == "Bearer secret-token-0123456789" for _, a, _ in http_server.requests)
    assert ClipCache(tmp_path).load() == 2


async def test_http_fetcher_errors_become_clip_fetch_errors(
    tmp_path: Path, http_server: _Server
) -> None:
    with pytest.raises(ClipFetchError, match="401"):
        await HttpClipFetcher(http_server.url, "wrong-token-0123456789").manifest(None)
    with pytest.raises(ClipFetchError, match="404"):
        await HttpClipFetcher(http_server.url, "secret-token-0123456789").clip("nope")
    with pytest.raises(ClipFetchError):
        await HttpClipFetcher("http://127.0.0.1:1", "x" * 16, timeout_s=1.0).manifest(None)
