"""プロバイダの口（Protocol）。設計書 §4.1。

セッションやエージェントはこの口だけを使い、実装（Fake / faster-whisper / SBV2 / OpenAI ...）は
設定で選ぶ（:mod:`.registry`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

from reachy_protocol.messages import Emotion


@dataclass(frozen=True)
class Transcription:
    text: str
    confidence: float | None = None
    lang: str = "ja"


@runtime_checkable
class STTProvider(Protocol):
    name: str

    async def transcribe(self, pcm_f32le: bytes, sr: int) -> Transcription:
        """float32 little-endian mono の音声を文字起こしする。"""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, emotion: Emotion = "neutral") -> bytes:
        """1 文を合成し、WAV（PCM16 mono）のバイト列を返す。"""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def chat_model(self) -> BaseChatModel:
        """LangChain / LangGraph でそのまま使えるチャットモデルを返す。"""
        ...
