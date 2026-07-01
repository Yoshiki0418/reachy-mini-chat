"""Reachy Mini 会話バックエンド (WebSocket 版)。

ロボット ⇄ PC を常時接続の双方向チャネルでつなぐ。1 接続 = 1 会話セッション。

プロトコル (robot → server):
  - binary: 1 発話ぶんの PCM (float32 mono @16kHz)
  - text(JSON): {"type":"reset"} 等の制御

プロトコル (server → robot):
  - text(JSON): {"type":"stt","text":...} / {"type":"motion","name":...} / {"type":"turn_end"}
  - binary: 応答音声チャンク (WAV bytes, PCM16)。文単位で逐次送る。

起動:
    uv run uvicorn pc_server.ws_server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pc_server.pipeline import ConversationPipeline

load_dotenv()


def _log(*args: Any) -> None:
    print(*args, flush=True, file=sys.stderr)


_pipeline: ConversationPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    t = time.monotonic()
    _log("モデルをロード中… (Whisper / SBV2 / LangGraph agent)")
    _pipeline = ConversationPipeline()
    _log(f"Ready. ({time.monotonic() - t:.1f}s)")
    yield


app = FastAPI(lifespan=lifespan)


def _wav_bytes(sr: int, audio: np.ndarray) -> bytes:
    """float32[-1,1] mono を WAV(PCM16) バイト列に。play_sound が任意 sr を再生できる。"""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.websocket("/ws/session")
async def session(ws: WebSocket) -> None:
    assert _pipeline is not None
    await ws.accept()
    session_id = uuid4().hex[:8]
    _log(f"[WS] connected session={session_id}")

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if "text" in msg and msg["text"]:
                # 制御メッセージ（今は reset のみ）
                _log(f"[WS] control: {msg['text']}")
                continue

            if "bytes" not in msg or not msg["bytes"]:
                continue

            audio = np.frombuffer(msg["bytes"], dtype=np.float32)
            _log(f"\n[WS] 受信 {len(audio) / 16000:.1f}s ({len(audio)} samples)")

            t = time.monotonic()
            async for ev in _pipeline.run_turn(audio, session_id):
                etype = ev["type"]
                if etype == "audio":
                    await ws.send_bytes(_wav_bytes(ev["sr"], ev["audio"]))
                    _log(f"  [audio] '{ev['text']}'")
                elif etype == "stt":
                    _log(f"  [stt {time.monotonic() - t:.1f}s] User: {ev['text']}")
                    await ws.send_json(ev)
                elif etype == "motion":
                    _log(f"  [motion] {ev['name']}")
                    await ws.send_json(ev)
                elif etype == "turn_end":
                    await ws.send_json(ev)

    except WebSocketDisconnect:
        _log(f"[WS] disconnected session={session_id}")
    except Exception as e:  # noqa: BLE001
        _log(f"[WS] error session={session_id}: {e}")
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "reachy-mini-chat", "transport": "websocket", "endpoint": "/ws/session"}
