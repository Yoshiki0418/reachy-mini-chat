"""faster-whisper の STT プロバイダ。本物のモデルは使わず、差し替えた偽のモデルで検証する。"""

from __future__ import annotations

import asyncio
import math
import struct
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from reachy_server.providers import STTProvider
from reachy_server.providers.faster_whisper_stt import (
    FasterWhisperSTT,
    confidence_from_segments,
)


@dataclass
class Seg:
    text: str
    start: float = 0.0
    end: float = 1.0
    avg_logprob: float = -0.1


class FakeWhisperModel:
    """``WhisperModel.transcribe`` と同じ形（遅延評価のジェネレータ + info）を返す。"""

    def __init__(self, segments: list[Seg], language: str = "ja") -> None:
        self.segments = segments
        self.language = language
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": audio, **kwargs})

        def gen() -> Any:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                yield from self.segments
            finally:
                self.active -= 1

        return gen(), SimpleNamespace(language=self.language)


def pcm(*samples: float) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


def make(model: FakeWhisperModel, **kw: Any) -> FasterWhisperSTT:
    # 音声の変換は numpy を使うので、CI（ML 依存なし）では差し替える
    return FasterWhisperSTT(model_factory=lambda: model, to_audio=lambda b, sr: (len(b), sr), **kw)


def test_is_an_stt_provider_and_names_the_model() -> None:
    stt = make(FakeWhisperModel([]))
    assert isinstance(stt, STTProvider)
    assert stt.name == "faster-whisper:kotoba-whisper-v2.0-faster"


async def test_transcribe_joins_segments_and_reports_language() -> None:
    model = FakeWhisperModel([Seg(" おはよう"), Seg("ございます。 ")], language="ja")
    result = await make(model).transcribe(pcm(0.1, 0.2), 16000)
    assert result.text == "おはようございます。"
    assert result.lang == "ja"
    assert result.confidence == pytest.approx(math.exp(-0.1))


async def test_transcribe_passes_the_benchmarked_options() -> None:
    model = FakeWhisperModel([Seg("はい")])
    await make(model, hotwords="リーチ").transcribe(pcm(0.0) * 100, 16000)
    call = model.calls[0]
    assert call["audio"] == (400, 16000)
    assert call["language"] == "ja"
    assert call["beam_size"] == 1
    assert call["vad_filter"] is False  # 区間切りはロボット側の VAD
    assert call["condition_on_previous_text"] is False
    assert call["hotwords"] == "リーチ"


async def test_silence_gives_empty_text_without_confidence() -> None:
    result = await make(FakeWhisperModel([])).transcribe(pcm(0.0) * 100, 16000)
    assert result.text == ""
    assert result.confidence is None


async def test_model_is_loaded_once_and_only_when_needed() -> None:
    loads: list[int] = []
    model = FakeWhisperModel([Seg("はい")])

    def factory() -> FakeWhisperModel:
        loads.append(1)
        return model

    stt = FasterWhisperSTT(model_factory=factory, to_audio=lambda b, sr: b)
    assert loads == []  # 作っただけでは読み込まない
    await stt.transcribe(pcm(0.0), 16000)
    await stt.transcribe(pcm(0.0), 16000)
    stt.load()
    assert loads == [1]


async def test_inference_runs_one_at_a_time() -> None:
    model = FakeWhisperModel([Seg("はい")])
    stt = make(model)
    await asyncio.gather(*(stt.transcribe(pcm(0.0), 16000) for _ in range(5)))
    assert len(model.calls) == 5
    assert model.max_active == 1


async def test_does_not_block_the_event_loop() -> None:
    ticks: list[int] = []

    class SlowModel(FakeWhisperModel):
        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
            import time

            time.sleep(0.2)
            return super().transcribe(audio, **kwargs)

    async def ticker() -> None:
        for i in range(5):
            ticks.append(i)
            await asyncio.sleep(0.02)

    await asyncio.gather(make(SlowModel([Seg("はい")])).transcribe(pcm(0.0), 16000), ticker())
    assert len(ticks) == 5


def test_compute_type_follows_the_device_unless_overridden() -> None:
    assert FasterWhisperSTT(device="cuda")._compute_type == "float16"
    assert FasterWhisperSTT(device="cpu")._compute_type == "int8"  # CPU は float16 非対応
    assert FasterWhisperSTT(device="cpu", compute_type="float32")._compute_type == "float32"


def test_confidence_is_duration_weighted() -> None:
    segs = [Seg("a", 0.0, 3.0, -0.2), Seg("b", 3.0, 4.0, -1.0)]
    assert confidence_from_segments(segs) == pytest.approx(math.exp((3 * -0.2 + 1 * -1.0) / 4))
    assert confidence_from_segments([]) is None
    assert confidence_from_segments([Seg("x", 0, 1, 0.5)]) == 1.0  # 確率は 1 を超えない


def test_importing_the_provider_does_not_pull_ml_dependencies() -> None:
    assert "faster_whisper" not in sys.modules


def test_registry_picks_a_free_gpu_then_loads_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_server import gpu
    from reachy_server.config import ServerSettings
    from reachy_server.providers import build_providers
    from reachy_server.providers import faster_whisper_stt as mod

    order: list[str] = []
    monkeypatch.setattr(
        gpu, "pin_free_gpu", lambda *, min_free_mib, env=None: order.append(f"pin:{min_free_mib}")
    )
    monkeypatch.setattr(mod.FasterWhisperSTT, "load", lambda self: order.append("load"))

    settings = ServerSettings(  # pyright: ignore[reportCallIssue]
        token=SecretStr("x" * 16), stt="faster-whisper", gpu_min_free_mib=9000, stt_beam_size=5
    )
    providers = build_providers(settings)
    assert order == ["pin:9000", "load"]  # CUDA を使う前に GPU を選び、起動時に読み込む
    assert providers.stt.name == "faster-whisper:kotoba-whisper-v2.0-faster"
    assert providers.stt._beam_size == 5  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]


def test_registry_skips_gpu_selection_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_server import gpu
    from reachy_server.config import ServerSettings
    from reachy_server.providers import build_providers
    from reachy_server.providers import faster_whisper_stt as mod

    def fail(**_kw: object) -> None:
        raise AssertionError("CPU 実行では GPU を選ばない")

    monkeypatch.setattr(gpu, "pin_free_gpu", fail)
    monkeypatch.setattr(mod.FasterWhisperSTT, "load", lambda self: None)
    settings = ServerSettings(  # pyright: ignore[reportCallIssue]
        token=SecretStr("x" * 16), stt="faster-whisper", stt_device="cpu"
    )
    build_providers(settings)
