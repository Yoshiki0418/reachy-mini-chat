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
    memory_ack_timeout_s: float = Field(
        default=3.0, gt=0, description="memory.op に対するロボットの memory.ack を待つ時間"
    )

    # 会話履歴（SQLite。空ならメモリ上）と保持期限
    history_db: str = "data/server/history.sqlite"
    history_retention_days: int = Field(default=14, ge=1)
    # 場の文脈バッファ（自分宛てでない発話を短時間だけ覚える。0 で無効）
    overheard_ttl_s: float = Field(default=600.0, ge=0)
    overheard_max: int = Field(default=20, ge=1)

    # プロバイダの選択（reachy_server.providers の登録名）。本物が揃うまで既定は fake
    stt: str = "fake"
    tts: str = "fake"
    llm: str = "fake"

    # GPU は研究室の共用機。起動時に、空きメモリが足りる GPU のうち使用率の低いものを選ぶ
    # （手で選ぶときは CUDA_VISIBLE_DEVICES）。STT・TTS を合わせて載せる分の目安
    gpu_min_free_mib: int = Field(default=8000, ge=0)

    # stt=faster-whisper のとき（既定は docs/benchmarks.md の結論）
    stt_model: str = "kotoba-tech/kotoba-whisper-v2.0-faster"
    stt_device: str = "cuda"
    stt_compute_type: str | None = Field(
        default=None, description="未指定なら GPU は float16、CPU は int8（CPU は float16 非対応）"
    )
    stt_beam_size: int = Field(default=1, ge=1)
    stt_hotwords: str | None = Field(default=None, description="認識させたい語（例: リーチ）")

    # tts=irodori のとき。Irodori-TTS は専用の環境（<irodori_dir>/.venv）でワーカーとして動かす
    irodori_dir: str = "~/Irodori-TTS"
    tts_checkpoint: str = "Aratako/Irodori-TTS-v4.1-Small-MF"
    tts_precision: str = Field(default="fp32", pattern="^(fp32|bf16)$")
    tts_num_steps: int | None = Field(default=None, ge=1, description="未指定ならモデルの既定")
    tts_ref_wav: str = Field(
        default="data/server/voice/reference.wav",
        description="話者の参照音声。声を固定する。実在の人の声は使わない",
    )
    tts_caption: str = ""
    tts_sample_rate: int = Field(default=24000, ge=8000, le=48000)
    tts_timeout_s: float = Field(default=10.0, gt=0, description="1 文の合成の上限")
    tts_startup_timeout_s: float = Field(default=300.0, gt=0, description="モデルの読み込み")

    # llm=openai のとき（API キーは標準の OPENAI_API_KEY）
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0, le=2)
    llm_max_tokens: int = Field(default=200, gt=0, description="1 回の応答の上限（音声なので短く）")

    def masked(self) -> dict[str, Any]:
        """ログ用の設定ダンプ（トークンは伏せる）。"""
        return self.model_dump(mode="json") | {"token": "**********"}
