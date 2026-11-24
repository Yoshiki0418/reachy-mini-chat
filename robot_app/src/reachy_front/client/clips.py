"""サーバーが事前合成した音声（フィラー・定型挨拶）のキャッシュ（設計書 §4.3）。

- ディスクに置き（音声のファイル名はハッシュ入りで、一覧を最後に切り替える）、**サーバーが落ちていても起動時に読める**（サーバー断のフォールバックの材料。#36）
- 版は一覧の ``version``。前回と同じなら取り直さない（``If-None-Match`` → 304）。音声は SHA-256 で検証
- 取得は「全部そろってから」切り替える。途中で失敗しても、前の版がそのまま使える
- ID はサーバーから来るので、ファイル名に使う前に検査する（``../`` などを通さない）。件数・大きさにも上限
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from reachy_protocol.clips import ClipInfo, ClipKind, ClipManifest

from ..core import events as ev
from ..core.bus import EventBus
from ..core.events import Event

log = logging.getLogger(__name__)

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_CLIPS = 64
MAX_CLIP_BYTES = 4 * 1024 * 1024
_MANIFEST = "manifest.json"


def clip_filename(info: ClipInfo) -> str:
    """ディスク上のファイル名。**中身（ハッシュ）で決める**ので、版が変わっても古い音声を上書きしない
    （一覧を置き換える前に落ちても、前の版がそのまま読める）。"""
    return f"{info.id}-{info.sha256[:16]}.wav"


class ClipFetchError(Exception):
    pass


class ClipFetcher(Protocol):
    async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
        """一覧。``if_none_match`` の版と同じなら None（304）。"""
        ...

    async def clip(self, clip_id: str) -> bytes: ...


class HttpClipFetcher:
    """``GET /v1/clips`` と ``GET /v1/clips/{id}``（標準ライブラリだけ。同期呼び出しはスレッドへ）。"""

    def __init__(self, base_url: str, token: str, *, timeout_s: float = 5.0) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    def _get(self, path: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        req = urllib.request.Request(
            self._base + path,
            headers={"Authorization": f"Bearer {self._token}", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as res:
                return res.status, res.read(MAX_CLIP_BYTES + 1)
        except urllib.error.HTTPError as e:
            if e.code == 304:
                return 304, b""
            raise ClipFetchError(f"HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ClipFetchError(type(e).__name__) from e

    async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
        headers = {"If-None-Match": f'"{if_none_match}"'} if if_none_match else None
        status, body = await asyncio.to_thread(self._get, "/v1/clips", headers)
        if status == 304:
            return None
        try:
            return ClipManifest.model_validate_json(body)
        except ValidationError as e:
            raise ClipFetchError("不正な一覧") from e

    async def clip(self, clip_id: str) -> bytes:
        _, body = await asyncio.to_thread(self._get, f"/v1/clips/{clip_id}")
        return body


@dataclass(frozen=True)
class CachedClip:
    info: ClipInfo
    wav: bytes


class ClipCache:
    def __init__(self, directory: str | Path, fetcher: ClipFetcher | None = None) -> None:
        self._dir = Path(directory)
        self._fetcher = fetcher
        self._clips: dict[str, CachedClip] = {}
        self._version: str | None = None
        self._lock = asyncio.Lock()

    @property
    def version(self) -> str | None:
        return self._version

    def get(self, clip_id: str) -> CachedClip | None:
        return self._clips.get(clip_id)

    def of_kind(self, kind: ClipKind) -> list[CachedClip]:
        return [c for c in self._clips.values() if c.info.kind == kind]

    # ---------------------------------------------------------------- ディスクから

    def load(self) -> int:
        """ディスクのキャッシュを読む（ネットワーク不要）。読めた件数を返す。壊れた分は飛ばす。"""
        self._clips, self._version = {}, None
        try:
            manifest = ClipManifest.model_validate_json((self._dir / _MANIFEST).read_bytes())
        except (OSError, ValidationError):
            return 0
        clips: dict[str, CachedClip] = {}
        for info in manifest.clips[:MAX_CLIPS]:
            wav = self._read_valid(info)
            if wav is not None:
                clips[info.id] = CachedClip(info, wav)
        # 一部でも欠けて（壊れて）いたら、版を「最新」と見なさない。次の refresh で一覧を取り直し、
        # 無事な分は使い回して、欠けた分だけ取る
        complete = len(clips) == len(manifest.clips)
        self._clips, self._version = clips, manifest.version if complete else None
        return len(clips)

    def _read_valid(self, info: ClipInfo) -> bytes | None:
        if not _ID.match(info.id):
            return None
        try:
            wav = (self._dir / clip_filename(info)).read_bytes()
        except OSError:
            return None
        if hashlib.sha256(wav).hexdigest() != info.sha256 or len(wav) != info.nbytes:
            log.warning("キャッシュの音声が壊れている: %s", info.id)
            return None
        return wav

    # ---------------------------------------------------------------- サーバーから

    async def refresh(self) -> bool:
        """サーバーの版を確かめ、変わっていれば取り直す。新しい版に切り替えたら True。

        失敗しても例外にしない（False。前の版は残る）。同時に 2 回走らない。
        """
        if self._fetcher is None:
            return False
        async with self._lock:
            try:
                return await self._refresh(self._fetcher)
            except ClipFetchError as e:
                log.warning("クリップの取得に失敗（前の版を使い続ける）: %s", e)
                return False
            except OSError:
                log.exception("クリップのキャッシュを書けない")
                return False

    async def _refresh(self, fetcher: ClipFetcher) -> bool:
        manifest = await fetcher.manifest(self._version)
        if manifest is None or manifest.version == self._version:
            return False
        if len(manifest.clips) > MAX_CLIPS:
            raise ClipFetchError("クリップが多すぎる")
        new: dict[str, CachedClip] = {}
        for info in manifest.clips:
            if not _ID.match(info.id) or info.nbytes > MAX_CLIP_BYTES:
                log.warning("不正なクリップを無視: %r", info.id)
                continue
            old = self._clips.get(info.id)
            if old is not None and old.info.sha256 == info.sha256:
                new[info.id] = CachedClip(info, old.wav)  # 変わっていない分は取り直さない
                continue
            wav = await fetcher.clip(info.id)
            if len(wav) != info.nbytes or hashlib.sha256(wav).hexdigest() != info.sha256:
                raise ClipFetchError(f"音声が一覧と合わない: {info.id}")
            new[info.id] = CachedClip(info, wav)
        await asyncio.to_thread(self._write, manifest, new)
        self._clips, self._version = new, manifest.version
        log.info("クリップを更新: 版 %s、%d 件", manifest.version, len(new))
        return True

    def _write(self, manifest: ClipManifest, clips: dict[str, CachedClip]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        wanted = {clip_filename(c.info) for c in clips.values()}
        for clip in clips.values():
            self._atomic(self._dir / clip_filename(clip.info), clip.wav)
        kept = ClipManifest(version=manifest.version, clips=[c.info for c in clips.values()])
        # 一覧は最後に書く。途中で止まっても、前の一覧と音声がそろったまま
        self._atomic(self._dir / _MANIFEST, kept.model_dump_json().encode())
        for p in self._dir.glob("*.wav"):  # 一覧を切り替えた後で、使わなくなった音声を消す
            if p.name not in wanted:
                with contextlib.suppress(OSError):
                    p.unlink()

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- バスにつなぐ

    def attach(self, bus: EventBus) -> None:
        """サーバーにつながるたびに、版を確かめて取り直す（動いている event loop の中で呼ぶ）。"""
        tasks: set[asyncio.Task[bool]] = set()

        def on_event(event: Event) -> None:
            if isinstance(event, ev.ServerConnectionChanged) and event.connected:
                t = asyncio.get_running_loop().create_task(self.refresh())
                tasks.add(t)
                t.add_done_callback(tasks.discard)

        bus.subscribe(ev.ServerConnectionChanged, on_event, name="clip-cache")


__all__ = [
    "MAX_CLIPS",
    "MAX_CLIP_BYTES",
    "CachedClip",
    "ClipCache",
    "ClipFetchError",
    "ClipFetcher",
    "HttpClipFetcher",
    "clip_filename",
]
