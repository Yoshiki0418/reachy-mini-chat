"""フィラー・定型挨拶の事前合成と配布（HTTP）。"""

from __future__ import annotations

import hashlib
import io
import wave
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from reachy_protocol.clips import ClipManifest
from reachy_protocol.messages import Emotion
from reachy_server.app import create_app
from reachy_server.clips import DEFAULT_CLIPS, ClipSpec, ClipStore
from reachy_server.config import ServerSettings
from reachy_server.providers import FakeTTS, Providers
from reachy_server.providers.base import TTSProvider


@pytest.fixture
def client(settings: ServerSettings, providers: Providers) -> Iterator[TestClient]:
    with TestClient(create_app(settings, providers)) as c:  # lifespan（事前合成）を走らせる
        yield c


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_default_clips_have_unique_ids_and_cover_each_kind() -> None:
    assert len({s.id for s in DEFAULT_CLIPS}) == len(DEFAULT_CLIPS)
    assert {s.kind for s in DEFAULT_CLIPS} == {"filler", "greeting", "notice"}


def test_manifest_lists_every_clip_with_matching_hash(client: TestClient, token: str) -> None:
    res = client.get("/v1/clips", headers=auth(token))
    assert res.status_code == 200
    manifest = ClipManifest.model_validate(res.json())
    assert {c.id for c in manifest.clips} == {s.id for s in DEFAULT_CLIPS}
    assert res.headers["etag"] == f'"{manifest.version}"'
    for info in manifest.clips:
        audio = client.get(f"/v1/clips/{info.id}", headers=auth(token))
        assert audio.status_code == 200
        assert audio.headers["content-type"] == "audio/wav"
        assert hashlib.sha256(audio.content).hexdigest() == info.sha256
        assert len(audio.content) == info.nbytes
        with wave.open(io.BytesIO(audio.content)) as w:  # 本物の WAV
            assert w.getnchannels() == 1


def test_unchanged_version_returns_304(client: TestClient, token: str) -> None:
    etag = client.get("/v1/clips", headers=auth(token)).headers["etag"]
    res = client.get("/v1/clips", headers={**auth(token), "If-None-Match": etag})
    assert res.status_code == 304
    assert res.content == b""
    stale = client.get("/v1/clips", headers={**auth(token), "If-None-Match": '"old"'})
    assert stale.status_code == 200


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token-0123456789"}])
def test_clips_require_the_token(client: TestClient, headers: dict[str, str]) -> None:
    assert client.get("/v1/clips", headers=headers).status_code == 401
    assert client.get("/v1/clips/filler-etto", headers=headers).status_code == 401


def test_unknown_clip_is_404(client: TestClient, token: str) -> None:
    assert client.get("/v1/clips/nope", headers=auth(token)).status_code == 404


async def test_version_changes_when_the_voice_changes() -> None:
    a, b = ClipStore(), ClipStore()
    await a.build(FakeTTS(seconds_per_char=0.02))
    await b.build(FakeTTS(seconds_per_char=0.03))  # 声（音声の中身）が変わった
    assert a.manifest and b.manifest
    assert a.manifest.version != b.manifest.version
    again = ClipStore()
    await again.build(FakeTTS(seconds_per_char=0.02))
    assert again.manifest and again.manifest.version == a.manifest.version  # 同じなら同じ


async def test_one_failed_synthesis_does_not_block_the_rest() -> None:
    class Flaky:
        name = "flaky"

        async def synthesize(self, text: str, emotion: Emotion = "neutral") -> bytes:
            if text == "こんにちは！":
                raise RuntimeError("boom")
            return await FakeTTS().synthesize(text, emotion)

    store = ClipStore()
    tts: TTSProvider = Flaky()
    await store.build(tts)
    assert store.manifest is not None
    ids = {c.id for c in store.manifest.clips}
    assert "greeting-hello" not in ids
    assert len(ids) == len(DEFAULT_CLIPS) - 1


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        ClipStore((ClipSpec("a", "filler", "あ"), ClipSpec("a", "filler", "い")))


def test_manifest_is_not_ready_before_build_returns_503(
    settings: ServerSettings, providers: Providers, token: str
) -> None:
    # lifespan を走らせない（with を使わない）クライアント = 事前合成の前
    res = TestClient(create_app(settings, providers)).get("/v1/clips", headers=auth(token))
    assert res.status_code == 503
