"""Reachy Mini 会話バックエンド (PC側)。

Robotから16kHz mono float32 のPCMを受け取り:
  STT(Whisper local) → OpenAI Chat → TTS(Style-Bert-VITS2) → WAV を返す

起動:
    uv run uvicorn pc_server.server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import soundfile as sf
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from openai import OpenAI


def _log(*args: Any) -> None:
    print(*args, flush=True, file=sys.stderr)


load_dotenv()

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SBV2_MODEL_DIR = os.environ.get("SBV2_MODEL_DIR", "models/sbv2")
SBV2_LENGTH = float(os.environ.get("SBV2_LENGTH", "1.1"))
SAMPLE_RATE = 16000  # Robotとmini.mediaの整合性のため16kHzに揃える

SYSTEM_PROMPT = """あなたはReachy Miniという小型ロボットです。
日本語で、親しみやすく、フランクに話してください。
返答は1〜2文以内、短くテンポよく。
ロボットらしくユーモアを交えても構いません。"""


_state_lock = threading.Lock()
_whisper_model: Any = None
_openai_client: OpenAI | None = None
_tts: Any = None  # StyleBertVITS2TTS
_messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _whisper_model, _openai_client, _tts
    _log(f"Whisper '{WHISPER_MODEL}' をロード中…")
    t = time.monotonic()
    _whisper_model = whisper.load_model(WHISPER_MODEL)
    _log(f"  → {time.monotonic() - t:.1f}s")

    _openai_client = OpenAI()
    _log(f"OpenAI モデル: {OPENAI_MODEL}")

    _log(f"Style-Bert-VITS2 をロード中… ({SBV2_MODEL_DIR})")
    t = time.monotonic()
    from pc_server.sbv2_tts import StyleBertVITS2TTS  # 遅延 import (起動高速化)

    _tts = StyleBertVITS2TTS(model_dir=SBV2_MODEL_DIR, length=SBV2_LENGTH)
    _log(f"  → {time.monotonic() - t:.1f}s")
    _log("Ready.")
    yield


app = FastAPI(lifespan=lifespan)


def _synthesize_tts(text: str) -> bytes:
    """Style-Bert-VITS2でテキスト合成 → 16kHz mono PCM WAVバイト列を返す。"""
    sr, audio = _tts.synthesize(text)
    # int16 (一般的なSBV2出力) または float32 を float32 に正規化
    if audio.dtype.kind == "i":
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    # mono化
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # 16kHzへリサンプリング (numpy線形補間)
    if sr != SAMPLE_RATE:
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len, dtype=np.float64),
            np.arange(len(audio), dtype=np.float64),
            audio,
        ).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/turn")
async def turn(request: Request) -> Response:
    raw = await request.body()
    if not raw:
        return Response(status_code=400, content="empty body")

    # 期待: float32 mono @ 16kHz
    audio = np.frombuffer(raw, dtype=np.float32)
    duration_sec = len(audio) / SAMPLE_RATE
    _log(f"\n[受信] {duration_sec:.1f}s  ({len(audio)}サンプル)")

    # STT
    t = time.monotonic()
    result = _whisper_model.transcribe(audio, language="ja", fp16=False)
    user_text = result["text"].strip()
    _log(f"[STT {time.monotonic() - t:.1f}s] User: {user_text}")

    if not user_text:
        return Response(status_code=204)

    # LLM
    with _state_lock:
        _messages.append({"role": "user", "content": user_text})
        msgs = list(_messages)

    t = time.monotonic()
    resp = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,  # type: ignore[arg-type]
        max_tokens=150,
        temperature=0.7,
    )
    assistant_text = (resp.choices[0].message.content or "").strip()
    _log(f"[LLM {time.monotonic() - t:.1f}s] Reachy: {assistant_text}")

    with _state_lock:
        _messages.append({"role": "assistant", "content": assistant_text})

    if not assistant_text:
        return Response(status_code=204)

    # TTS
    t = time.monotonic()
    wav_bytes = _synthesize_tts(assistant_text)
    _log(f"[TTS {time.monotonic() - t:.1f}s] {len(wav_bytes)}バイト")

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/reset")
def reset() -> dict[str, str]:
    """会話履歴をリセット。"""
    with _state_lock:
        _messages.clear()
        _messages.append({"role": "system", "content": SYSTEM_PROMPT})
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "reachy-mini-chat",
        "whisper_model": WHISPER_MODEL,
        "openai_model": OPENAI_MODEL,
        "tts": "style-bert-vits2",
        "tts_model_dir": SBV2_MODEL_DIR,
        "messages_count": len(_messages),
    }
