"""プロバイダ: 登録簿による切替・Fake の振る舞い・音声変換。"""

from __future__ import annotations

import io
import logging
import struct
import wave

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from reachy_protocol.messages import Hello, dump_message
from reachy_server.app import create_app
from reachy_server.config import ServerSettings
from reachy_server.providers import (
    FakeLLM,
    FakeSTT,
    FakeTTS,
    LLMProvider,
    STTProvider,
    TTSProvider,
    UnknownProviderError,
    available,
    build_providers,
    register_stt,
)
from reachy_server.providers import registry as reg
from reachy_server.providers.audio import f32le_to_pcm16le, pcm16_to_wav, silent_wav


def pcm(*samples: float) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


# ---------------------------------------------------------------- 登録簿と切替


def test_fake_is_registered_for_every_kind() -> None:
    assert {k: "fake" in v for k, v in available().items()} == {
        "stt": True,
        "tts": True,
        "llm": True,
    }


def test_build_providers_defaults_to_fakes_and_warns(
    settings: ServerSettings, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        providers = build_providers(settings)
    assert providers.describe() == {"stt": "fake", "tts": "fake", "llm": "fake"}
    assert isinstance(providers.stt, STTProvider)
    assert isinstance(providers.tts, TTSProvider)
    assert isinstance(providers.llm, LLMProvider)
    assert "偽のプロバイダ" in caplog.text


def test_unknown_provider_lists_choices(settings: ServerSettings) -> None:
    with pytest.raises(UnknownProviderError, match=r"'whisper-xl'.*選択肢: .*fake"):
        build_providers(settings.model_copy(update={"stt": "whisper-xl"}))


def test_registered_provider_is_selected_by_settings(
    settings: ServerSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reg, "_STT", dict(reg._STT))  # 登録簿を汚さない

    @register_stt("custom")
    def _custom(s: ServerSettings) -> STTProvider:
        return FakeSTT(script=["custom"], name="custom")

    providers = build_providers(settings.model_copy(update={"stt": "custom"}))
    assert providers.describe()["stt"] == "custom"


def test_provider_names_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path, token: str) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REACHY_SERVER_TOKEN", token)
    monkeypatch.setenv("REACHY_SERVER_STT", "faster-whisper")
    s = ServerSettings()  # pyright: ignore[reportCallIssue]
    assert (s.stt, s.tts, s.llm) == ("faster-whisper", "fake", "fake")


def test_app_refuses_to_start_with_unknown_provider(settings: ServerSettings) -> None:
    with pytest.raises(UnknownProviderError):
        create_app(settings.model_copy(update={"tts": "nope"}))


def test_welcome_reports_provider_names(settings: ServerSettings, token: str) -> None:
    client = TestClient(create_app(settings))
    with client.websocket_connect("/v1/session") as ws:
        ws.send_text(dump_message(Hello(robot_id="r", token=SecretStr(token))))
        welcome = ws.receive_json()
    assert welcome["models"] == {"stt": "fake", "tts": "fake", "llm": "fake"}


# ---------------------------------------------------------------- Fake の振る舞い


async def test_fake_stt_follows_script_then_repeats_last() -> None:
    stt = FakeSTT(script=["一つ目", "二つ目"])
    texts = [(await stt.transcribe(pcm(0.0) * 16000, 16000)).text for _ in range(3)]
    assert texts == ["一つ目", "二つ目", "二つ目"]
    assert stt.calls == [pytest.approx(1.0)] * 3  # 受け取った音声の秒数


async def test_fake_stt_with_function() -> None:
    stt = FakeSTT(script=lambda audio, sr: f"{len(audio) // 4} samples @ {sr}")
    assert (await stt.transcribe(pcm(0.1, 0.2), 16000)).text == "2 samples @ 16000"


async def test_fake_tts_returns_playable_wav() -> None:
    tts = FakeTTS(sr=22050, seconds_per_char=0.1)
    data = await tts.synthesize("こんにちは", "happy")
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 22050)
        assert w.getnframes() == int(5 * 0.1 * 22050)
    assert tts.calls == [("こんにちは", "happy")]


async def test_fake_llm_streams_scripted_replies() -> None:
    model = FakeLLM(replies=["一回目の返事。", "二回目。"]).chat_model()
    first = "".join(str(c.content) for c in model.stream("やあ"))
    second = (await model.ainvoke("やあ")).content
    third = (await model.ainvoke("やあ")).content
    assert (first, second, third) == ("一回目の返事。", "二回目。", "二回目。")


async def test_fake_llm_each_model_starts_from_the_top() -> None:
    llm = FakeLLM(replies=["A", "B"])
    assert (await llm.chat_model().ainvoke("x")).content == "A"
    assert (await llm.chat_model().ainvoke("x")).content == "A"


# ---------------------------------------------------------------- 音声変換


def test_f32le_to_pcm16le_clips_and_scales() -> None:
    out = struct.unpack("<5h", f32le_to_pcm16le(pcm(0.0, 0.5, -0.5, 1.5, -2.0)))
    assert out == (0, 16383, -16383, 32767, -32767)


def test_wav_helpers() -> None:
    with wave.open(io.BytesIO(silent_wav(0.5, 16000))) as w:
        assert w.getnframes() == 8000
    with wave.open(io.BytesIO(pcm16_to_wav(b"\x01\x00\x02\x00", 8000))) as w:
        assert w.readframes(2) == b"\x01\x00\x02\x00"
