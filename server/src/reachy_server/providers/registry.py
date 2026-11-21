"""プロバイダの登録簿。設定の名前から実装を作る。

本物の実装は ``@register_stt("faster-whisper")`` のように登録する。
GPU やクラウドの依存は生成関数の中で import し、使わない構成（CI・Fake）では読み込まない。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from .base import LLMProvider, STTProvider, TTSProvider
from .fakes import FakeLLM, FakeSTT, FakeTTS

if TYPE_CHECKING:
    from ..config import ServerSettings

log = logging.getLogger(__name__)

T = TypeVar("T")
Factory = Callable[["ServerSettings"], T]

_STT: dict[str, Factory[STTProvider]] = {}
_TTS: dict[str, Factory[TTSProvider]] = {}
_LLM: dict[str, Factory[LLMProvider]] = {}


class UnknownProviderError(ValueError):
    pass


def _register(table: dict[str, Factory[T]], name: str) -> Callable[[Factory[T]], Factory[T]]:
    def deco(factory: Factory[T]) -> Factory[T]:
        table[name] = factory
        return factory

    return deco


def register_stt(name: str) -> Callable[[Factory[STTProvider]], Factory[STTProvider]]:
    return _register(_STT, name)


def register_tts(name: str) -> Callable[[Factory[TTSProvider]], Factory[TTSProvider]]:
    return _register(_TTS, name)


def register_llm(name: str) -> Callable[[Factory[LLMProvider]], Factory[LLMProvider]]:
    return _register(_LLM, name)


def available() -> dict[str, list[str]]:
    return {"stt": sorted(_STT), "tts": sorted(_TTS), "llm": sorted(_LLM)}


def _create(kind: str, table: dict[str, Factory[T]], name: str, settings: ServerSettings) -> T:
    factory = table.get(name)
    if factory is None:
        raise UnknownProviderError(
            f"{kind} プロバイダ {name!r} は未登録（選択肢: {', '.join(sorted(table))}）"
        )
    return factory(settings)


@dataclass(frozen=True)
class Providers:
    stt: STTProvider
    tts: TTSProvider
    llm: LLMProvider

    def describe(self) -> dict[str, str]:
        """welcome.models に載せる名前。"""
        return {"stt": self.stt.name, "tts": self.tts.name, "llm": self.llm.name}


def build_providers(settings: ServerSettings) -> Providers:
    providers = Providers(
        stt=_create("STT", _STT, settings.stt, settings),
        tts=_create("TTS", _TTS, settings.tts, settings),
        llm=_create("LLM", _LLM, settings.llm, settings),
    )
    fakes = [k for k, v in providers.describe().items() if v == "fake"]
    if fakes:
        log.warning(
            "偽のプロバイダで動いています: %s（本番では REACHY_SERVER_STT 等を設定）", fakes
        )
    return providers


# ---------------------------------------------------------------- 組み込み（Fake）


@register_stt("fake")
def _fake_stt(_settings: ServerSettings) -> STTProvider:
    return FakeSTT()


@register_tts("fake")
def _fake_tts(_settings: ServerSettings) -> TTSProvider:
    return FakeTTS()


@register_llm("fake")
def _fake_llm(_settings: ServerSettings) -> LLMProvider:
    return FakeLLM()


# ---------------------------------------------------------------- 組み込み（クラウド）


@register_llm("openai")
def _openai_llm(settings: ServerSettings) -> LLMProvider:
    from .openai_llm import OpenAILLM  # 使うときだけ読み込む

    return OpenAILLM(
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


# ---------------------------------------------------------------- 組み込み（GPU）


@register_stt("faster-whisper")
def _faster_whisper_stt(settings: ServerSettings) -> STTProvider:
    from ..gpu import pin_free_gpu
    from .faster_whisper_stt import FasterWhisperSTT  # 使うときだけ読み込む

    if settings.stt_device == "cuda":
        pin_free_gpu(min_free_mib=settings.gpu_min_free_mib)  # CUDA の初期化より前に選ぶ
    stt = FasterWhisperSTT(
        model=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        beam_size=settings.stt_beam_size,
        hotwords=settings.stt_hotwords,
    )
    stt.load()  # 起動時に読み込む。最初の発話で待たせない
    return stt
