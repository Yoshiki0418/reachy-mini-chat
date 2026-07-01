"""会話パイプライン: STT → LangGraph エージェント(stream) → 文分割 → SBV2 TTS。

1ターンを、順序付きイベントの async ジェネレータとして返す:
  {"type":"stt",      "text": <認識文>}
  {"type":"motion",   "name": <motion名>}
  {"type":"audio",    "sr": int, "audio": np.ndarray(float32), "text": <合成した文>}
  {"type":"turn_end"}

文単位で TTS するので、最初の文が読み上がるまでのレイテンシを短縮できる。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import numpy as np
from langchain_core.messages import AIMessageChunk, HumanMessage

from pc_server.agent import build_agent, motion_sink_var
from pc_server.models import WhisperSTT, get_tts

# 文の区切りとみなす文字。
_SENTENCE_END = "。.!?！？\n"


def _split_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """buffer から「区切り文字で終わる完結文」を切り出す。

    戻り値: (完結した文のリスト, 未完の残り)
    """
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_END:
            piece = buffer[start : i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    return sentences, buffer[start:]


class ConversationPipeline:
    def __init__(self) -> None:
        self.stt = WhisperSTT()
        self.tts = get_tts()
        self.agent = build_agent()

    async def _tts(self, text: str) -> tuple[int, np.ndarray]:
        # SBV2 の合成は CPU バウンドなのでスレッドに逃がしてイベントループを塞がない。
        return await asyncio.to_thread(self.tts.synthesize, text)

    async def run_turn(
        self, audio: np.ndarray, session_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        # --- STT ---
        user_text = await asyncio.to_thread(self.stt.transcribe, audio, "ja")
        yield {"type": "stt", "text": user_text}
        if not user_text:
            yield {"type": "turn_end"}
            return

        # --- モーション side-channel をこのターン用に用意 ---
        sink: list[dict] = []
        motion_sink_var.set(sink)

        def drain_motions() -> list[dict]:
            out = list(sink)
            sink.clear()
            return out

        # --- エージェントをトークンストリームで実行し、文単位で TTS ---
        config = {"configurable": {"thread_id": session_id}}
        buffer = ""
        async for chunk, _meta in self.agent.astream(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
            stream_mode="messages",
        ):
            if not (isinstance(chunk, AIMessageChunk) and chunk.content):
                continue
            buffer += chunk.content
            sentences, buffer = _split_completed_sentences(buffer)
            for sentence in sentences:
                for m in drain_motions():
                    yield m
                sr, wav = await self._tts(sentence)
                yield {"type": "audio", "sr": sr, "audio": wav, "text": sentence}

        # 残りの未完文を flush
        tail = buffer.strip()
        if tail:
            for m in drain_motions():
                yield m
            sr, wav = await self._tts(tail)
            yield {"type": "audio", "sr": sr, "audio": wav, "text": tail}

        # ツールだけ呼ばれて本文が無い等でも、残ったモーションを送る
        for m in drain_motions():
            yield m

        yield {"type": "turn_end"}
