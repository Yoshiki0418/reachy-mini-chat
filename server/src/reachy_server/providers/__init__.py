"""STT / TTS / LLM のプロバイダ（差し替え可能な実装）と登録簿。"""

from __future__ import annotations

from .base import LLMProvider, STTProvider, Transcription, TTSProvider
from .fakes import FakeLLM, FakeSTT, FakeTTS
from .registry import (
    Providers,
    UnknownProviderError,
    available,
    build_providers,
    register_llm,
    register_stt,
    register_tts,
)

__all__ = [
    "FakeLLM",
    "FakeSTT",
    "FakeTTS",
    "LLMProvider",
    "Providers",
    "STTProvider",
    "TTSProvider",
    "Transcription",
    "UnknownProviderError",
    "available",
    "build_providers",
    "register_llm",
    "register_stt",
    "register_tts",
]
