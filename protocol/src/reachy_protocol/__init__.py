"""ロボット（フロントエンド）と GPU サーバー間のプロトコル定義。設計書 §2。

- :mod:`reachy_protocol.messages`: 制御メッセージ（WebSocket の text フレーム）
- :mod:`reachy_protocol.version`: 版と互換判定
- バイナリフレーミング（音声・画像）は #10 で追加する
"""

from __future__ import annotations

from .version import PROTOCOL_VERSION, is_compatible, parse_version

__version__ = "0.1.0"

__all__ = ["PROTOCOL_VERSION", "__version__", "is_compatible", "parse_version"]
