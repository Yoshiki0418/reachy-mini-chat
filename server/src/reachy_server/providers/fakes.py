"""テスト・開発用の偽プロバイダ。外部 API も GPU も使わない。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

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


class ScriptedChatModel(BaseChatModel):
    """台本どおりの AIMessage を順に返すチャットモデル。ツール呼び出しも台本に書ける。

    - ストリーミングではテキストを数文字ずつ、ツール呼び出しは最後のチャンクで返す
    - 台本を使い切ったら最後のメッセージを繰り返す
    - 受け取った入力（プロンプト）を ``inputs`` に記録する（テストでプロンプトを検査するため）
    """

    script: list[AIMessage]
    chunk_chars: int = 3
    inputs: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        return self  # 台本が決まっているのでツール定義は使わない

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.inputs.append(list(messages))
        msg = self.script[min(self.cursor, len(self.script) - 1)] if self.script else AIMessage("")
        self.cursor += 1
        return msg

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next(messages)
        text = msg.text
        for i in range(0, len(text), self.chunk_chars):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=text[i : i + self.chunk_chars])
            )
        if msg.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        ToolCallChunk(
                            name=tc["name"], args=json.dumps(tc["args"]), id=tc["id"], index=i
                        )
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                )
            )


def tool_call(
    name: str, args: dict[str, Any] | None = None, call_id: str | None = None
) -> AIMessage:
    """台本用: ツールを呼ぶだけの AI メッセージ。"""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id or f"call_{name}"}]
    )


@dataclass
class FakeLLM:
    """台本どおりの返答（テキスト・ツール呼び出し）をストリーミングするチャットモデルを作る。

    文字列はそのままの返答、``tool_call()`` で作ったメッセージはツール呼び出しになる。
    ``chat_model()`` を呼ぶたびに台本の先頭から始まる。
    """

    replies: Sequence[str | AIMessage] = ("こんにちは！今日も来てくれてうれしいな。",)
    name: str = "fake"
    models: list[ScriptedChatModel] = field(default_factory=list[ScriptedChatModel])

    def chat_model(self) -> BaseChatModel:
        script = [AIMessage(r) if isinstance(r, str) else r for r in self.replies]
        model = ScriptedChatModel(script=script)
        self.models.append(model)
        return model
