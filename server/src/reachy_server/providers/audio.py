"""音声バイト列の小さな変換（numpy に依存しない）。"""

from __future__ import annotations

import array
import io
import sys
import wave


def pcm_f32le_duration(pcm_f32le: bytes, sr: int) -> float:
    return len(pcm_f32le) / 4 / sr


def silent_wav(seconds: float, sr: int = 24000) -> bytes:
    """無音の WAV（PCM16 mono）。"""
    n = max(0, int(seconds * sr))
    return pcm16_to_wav(b"\x00\x00" * n, sr)


def pcm16_to_wav(pcm16le: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16le)
    return buf.getvalue()


def f32le_to_pcm16le(pcm_f32le: bytes) -> bytes:
    """float32[-1, 1] を PCM16 に変換する（範囲外はクリップ）。"""
    samples = array.array("f")
    samples.frombytes(pcm_f32le)
    if sys.byteorder == "big":
        samples.byteswap()
    out = array.array("h", (int(max(-1.0, min(1.0, x)) * 32767) for x in samples))
    if sys.byteorder == "big":
        out.byteswap()
    return out.tobytes()
