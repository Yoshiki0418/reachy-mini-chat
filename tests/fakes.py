"""テスト用のフェイク（STT / TTS / エージェント）。外部 API・ML モデルを使わない。"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.messages import AIMessageChunk

from pc_server.agent.tools.motion_tool import motion_sink_var
from pc_server.pipeline import ConversationPipeline


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio: np.ndarray, language: str = "ja") -> str:
        return self.text


class FakeTTS:
    SR = 16000

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        self.spoken.append(text)
        return self.SR, np.zeros(160, dtype=np.float32)


class FakeAgent:
    """LangGraph の astream(stream_mode="messages") を模倣する。

    chunks: ストリームするテキスト片。motions: 最初のチャンク前に「ツールが呼ばれた」扱いで
    motion sink へ積むモーション名。
    """

    def __init__(self, chunks: list[str], motions: list[str] | None = None) -> None:
        self.chunks = chunks
        self.motions = motions or []
        self.calls: list[dict[str, Any]] = []

    async def astream(self, inputs: dict, config: dict, stream_mode: str):
        self.calls.append({"inputs": inputs, "config": config})
        sink = motion_sink_var.get()
        for m in self.motions:
            if sink is not None:
                sink.append({"type": "motion", "name": m})
        for c in self.chunks:
            yield AIMessageChunk(content=c), {}


def make_pipeline(
    stt_text: str, agent: FakeAgent, tmp_path
) -> tuple[ConversationPipeline, FakeTTS]:
    from pc_server.person.store import PersonStore

    p = ConversationPipeline.__new__(ConversationPipeline)
    tts = FakeTTS()
    p.stt = FakeSTT(stt_text)  # type: ignore[assignment]
    p.tts = tts
    p.agent = agent  # type: ignore[assignment]
    p.persons = PersonStore(path=str(tmp_path / "people.json"))
    p.face = None
    return p, tts
