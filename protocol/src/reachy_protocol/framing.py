"""バイナリフレーム（WebSocket の binary フレーム）。設計書 §2.1。

形式::

    [ヘッダ長: 4 バイト・ビッグエンディアン][JSON ヘッダ (UTF-8)][ペイロード]

kind と format の組み合わせは固定:

==========  ============  =====================================================
kind        format        内容
==========  ============  =====================================================
audio_in    pcm_f32le     ロボット → サーバー。発話中のマイク音声（mono, 16 kHz）
audio_out   wav_pcm16     サーバー → ロボット。合成音声（1 文 = 1 フレーム）
frame       jpeg          ロボット → サーバー。VLM 用の縮小カメラ画像
==========  ============  =====================================================
"""

from __future__ import annotations

import json
import struct
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .messages import Id

__all__ = [
    "MAX_FRAME_BYTES",
    "MAX_HEADER_BYTES",
    "BinaryHeader",
    "FrameFormat",
    "FrameKind",
    "FramingError",
    "decode_frame",
    "encode_frame",
]

MAX_HEADER_BYTES = 4 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024

_LEN = struct.Struct(">I")

FrameKind: TypeAlias = Literal["audio_in", "audio_out", "frame"]
FrameFormat: TypeAlias = Literal["pcm_f32le", "wav_pcm16", "jpeg"]

_FORMAT_OF: dict[str, str] = {
    "audio_in": "pcm_f32le",
    "audio_out": "wav_pcm16",
    "frame": "jpeg",
}


class FramingError(ValueError):
    """バイナリフレームが壊れている・規約に合わない。"""


class BinaryHeader(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: FrameKind
    ref: Id = Field(description="audio_in / frame は utt_id、audio_out は turn_id")
    seq: int = Field(ge=0, description="audio_in はチャンク番号、audio_out は Sentence.idx")
    format: FrameFormat
    sr: int | None = Field(default=None, gt=0, description="pcm_f32le のサンプリング周波数")

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = _FORMAT_OF[self.kind]
        if self.format != expected:
            raise ValueError(f"kind={self.kind} の format は {expected}（{self.format} ではない）")
        if self.format == "pcm_f32le" and self.sr is None:
            raise ValueError("pcm_f32le には sr が必要")
        return self


def _check_payload(header: BinaryHeader, payload: bytes) -> None:
    if header.format == "pcm_f32le" and len(payload) % 4 != 0:
        raise FramingError(f"pcm_f32le の長さが 4 の倍数でない: {len(payload)} バイト")


def encode_frame(header: BinaryHeader, payload: bytes) -> bytes:
    """ヘッダとペイロードを 1 つのバイナリフレームにする。"""
    _check_payload(header, payload)
    head = header.model_dump_json(exclude_none=True).encode("utf-8")
    if len(head) > MAX_HEADER_BYTES:
        raise FramingError(f"ヘッダが大きすぎる: {len(head)} バイト")
    frame = _LEN.pack(len(head)) + head + payload
    if len(frame) > MAX_FRAME_BYTES:
        raise FramingError(f"フレームが大きすぎる: {len(frame)} バイト")
    return frame


def decode_frame(data: bytes) -> tuple[BinaryHeader, bytes]:
    """バイナリフレームを (ヘッダ, ペイロード) に分ける。壊れていれば FramingError。"""
    if len(data) > MAX_FRAME_BYTES:
        raise FramingError(f"フレームが大きすぎる: {len(data)} バイト")
    if len(data) < _LEN.size:
        raise FramingError(f"短すぎる: {len(data)} バイト")
    (head_len,) = _LEN.unpack_from(data)
    if head_len == 0 or head_len > MAX_HEADER_BYTES:
        raise FramingError(f"ヘッダ長が不正: {head_len}")
    end = _LEN.size + head_len
    if end > len(data):
        raise FramingError(f"ヘッダが途中で切れている（ヘッダ長 {head_len}, 全体 {len(data)}）")
    try:
        raw = json.loads(data[_LEN.size : end].decode("utf-8"))
        header = BinaryHeader.model_validate(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as e:
        raise FramingError(f"ヘッダが不正: {e}") from e
    payload = data[end:]
    _check_payload(header, payload)
    return header, payload
