"""サーバーの設定。環境変数（接頭辞 REACHY_SERVER_）と .env から読む。"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REACHY_SERVER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    token: SecretStr = Field(description="ロボットと共有する認証トークン（必須）", min_length=16)
    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    server_version: str = __version__
    handshake_timeout_s: float = Field(default=10.0, gt=0, description="接続から hello までの猶予")

    decision_timeout_s: float = Field(
        default=15.0, gt=0, description="文字起こし後に判定を待つ時間"
    )
    max_utterance_s: float = Field(default=30.0, gt=0, description="1 発話の音声の上限")

    # プロバイダの選択（reachy_server.providers の登録名）。本物が揃うまで既定は fake
    stt: str = "fake"
    tts: str = "fake"
    llm: str = "fake"

    # llm=openai のとき（API キーは標準の OPENAI_API_KEY）
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_max_tokens: int = Field(default=200, gt=0, description="1 回の応答の上限（音声なので短く）")

    def masked(self) -> dict[str, Any]:
        """ログ用の設定ダンプ（トークンは伏せる）。"""
        return self.model_dump(mode="json") | {"token": "**********"}
