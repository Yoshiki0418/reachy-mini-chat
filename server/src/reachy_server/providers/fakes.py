"""テスト・開発用の偽プロバイダ。外部 API も GPU も使わない。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from reachy_protocol.messages import Emotion

from .audio import pcm_f32le_duration, silent_wav
from .base import Transcription


@dataclass
class FakeSTT:
    """台本どおりに文字起こしする。台本を使い切ったら最後の文を返し続ける。

    ``script`` に関数を渡すと、(音声, sr) から文字列を作れる。
    """

    script: Sequence[str] | Callable[[bytes, int], str] = ("こんにちは",)
    delay_s: float = 0.0
    name: str = "fake"
    calls: list[float] = field(default_factory=list[float])  # 受け取った音声の秒数

    async def transcribe(self, pcm_f32le: bytes, sr: int) -> Transcription:
        self.calls.append(pcm_f32le_duration(pcm_f32le, sr))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if callable(self.script):
            return Transcription(self.script(pcm_f32le, sr), confidence=1.0)
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        return Transcription(self.script[idx] if self.script else "", confidence=1.0)


@dataclass
class FakeTTS:
    """本物の WAV（無音、長さは文字数に比例）を返す。ロボット側でそのまま再生できる。"""

    sr: int = 24000
    seconds_per_char: float = 0.02
    delay_s: float = 0.0
    name: str = "fake"
    calls: list[tuple[str, Emotion]] = field(default_factory=list[tuple[str, Emotion]])

    async def synthesize(self, text: str, emotion: Emotion = "neutral") -> bytes:
        self.calls.append((text, emotion))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return silent_wav(len(text) * self.seconds_per_char, self.sr)


@dataclass
class FakeLLM:
    """台本どおりの返答をストリーミングするチャットモデルを作る。

    台本を使い切ったら最後の返答を繰り返す。ツール呼び出しの台本は #20 で拡張する。
    """

    replies: Sequence[str] = ("こんにちは！今日も来てくれてうれしいな。",)
    name: str = "fake"

    def chat_model(self) -> BaseChatModel:
        replies = list(self.replies) or [""]

        def forever() -> object:
            i = 0
            while True:
                yield AIMessage(content=replies[min(i, len(replies) - 1)])
                i += 1

        return GenericFakeChatModel(messages=forever())  # pyright: ignore[reportArgumentType]
