"""STT (音声認識) 層。現状はローカル Whisper。差替可能なように base を挟む。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSTT(ABC):
    """STT の基底クラス。"""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, language: str = "ja") -> str:
        """float32 mono @16kHz の音声を文字起こしして返す。"""
        ...


class WhisperSTT(BaseSTT):
    """OpenAI Whisper (ローカル) による日本語 STT。"""

    def __init__(self, model_name: str | None = None) -> None:
        # 遅延 import（起動高速化）。whisper は pc-ml グループなので CI には無い
        import whisper  # pyright: ignore[reportMissingImports]

        self.model_name = model_name or os.environ.get("WHISPER_MODEL", "base")
        self._model: Any = whisper.load_model(self.model_name)

    def transcribe(self, audio: np.ndarray, language: str = "ja") -> str:
        audio = np.asarray(audio, dtype=np.float32)
        result = self._model.transcribe(audio, language=language, fp16=False)
        return str(result.get("text", "")).strip()
