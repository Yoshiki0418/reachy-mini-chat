"""事前に合成した音声（フィラー・定型挨拶）の配布。設計書 §4.3。

サーバーが起動時に合成し、ロボットが HTTP で取りに来てキャッシュする。
  GET /v1/clips        → :class:`ClipManifest`（版つきの一覧）
  GET /v1/clips/{id}   → WAV（PCM16 mono）
どちらも ``Authorization: Bearer <トークン>`` が要る。一覧の ``version`` が前回と同じなら、
音声は取り直さなくてよい（``If-None-Match`` に版を付けると 304 が返る）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .messages import Emotion

ClipKind = Literal["filler", "greeting", "notice"]


class ClipInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="音声の識別子。URL の一部になる")
    kind: ClipKind = Field(
        description="filler=間つなぎ / greeting=入室者への定型挨拶 / notice=会話できない旨"
    )
    text: str
    emotion: Emotion = "neutral"
    sha256: str = Field(description="WAV の SHA-256（16 進）。キャッシュの検証に使う")
    nbytes: int = Field(ge=0)


class ClipManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str = Field(description="全体の版。声や文言が変わると変わる")
    clips: list[ClipInfo]


__all__ = ["ClipInfo", "ClipKind", "ClipManifest"]
