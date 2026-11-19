"""研究室 GPU マシン上で動く生成バックエンド。設計書 §4。

サーバー側の責務: STT・応答生成（LangGraph）・TTS・会話履歴。

- :mod:`reachy_server.app`: FastAPI アプリ（WS ``/v1/session``、``/healthz``）
- :mod:`reachy_server.session`: 1 接続の処理（ハンドシェイク・死活監視。ターンは #16 以降）
- :mod:`reachy_server.config`: 設定（環境変数 ``REACHY_SERVER_*``）
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
