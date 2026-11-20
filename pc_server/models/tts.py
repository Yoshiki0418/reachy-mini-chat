"""TTS 層。現状は Style-Bert-VITS2。env で差替可能にするファクトリを提供。

SBV2 実装本体は pc_server.sbv2_tts.StyleBertVITS2TTS（Phase 0 で numpy ABI 解消済み）。
`synthesize(text) -> (sample_rate, float32[-1,1] mono)` の契約に統一している。
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class BaseTTS(Protocol):
    """TTS の構造的インターフェース。"""

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """テキストを (sample_rate, float32[-1,1] mono) に合成する。"""
        ...


def get_tts(provider: str | None = None) -> BaseTTS:
    """env の TTS_PROVIDER に応じて TTS 実装を返す。"""
    provider = provider or os.environ.get("TTS_PROVIDER", "style-bert-vits2")
    if provider == "style-bert-vits2":
        from pc_server.sbv2_tts import StyleBertVITS2TTS

        model_dir = os.environ.get("SBV2_MODEL_DIR", "models/sbv2")
        length = float(os.environ.get("SBV2_LENGTH", "1.1"))
        return StyleBertVITS2TTS(model_dir=model_dir, length=length)
    raise ValueError(f"Unknown TTS provider: {provider}")
