"""研究室 GPU マシン上で動く生成バックエンド。

サーバー側の責務: STT・応答生成（LangGraph）・TTS・会話履歴。
骨格・WS・認証は #15 以降で追加する。設計書 §4。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
