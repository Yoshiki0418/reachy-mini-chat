"""Reachy Mini 会話クライアント (WebSocket 版)。

旧 conversation.py (HTTP 1往復) を置き換える常時接続クライアント。
VAD で発話区間を検出 → PC へ送信 → 文単位で返る音声チャンクを逐次再生し、
motion 指示を Motion Executor で実行する。

実行:
    PC_HOST=yoshikinoMacBook-Air.local /venvs/apps_venv/bin/python robot/conversation_ws.py

依存: websockets (apps_venv に導入済み: 15.x)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
from websockets.sync.client import connect

from reachy_mini import ReachyMini

# 同ディレクトリの既存 VAD 実装と Motion Controller を再利用
from conversation import collect_utterance, to_mono_float32  # noqa: E402
from motion import MotionController  # noqa: E402

PC_HOST = os.environ.get("PC_HOST", "yoshikinoMacBook-Air.local")
PC_PORT = int(os.environ.get("PC_PORT", "8765"))
WS_URL = f"ws://{PC_HOST}:{PC_PORT}/ws/session"

POST_PLAYBACK_DELAY = 0.4  # 再生後のエコー収まり待ち
RECV_TIMEOUT = 60.0
FRAME_MAX_W = 512          # 送信フレームの最大幅（VLM は detail=low なので小さくてよい）
FRAME_JPEG_QUALITY = 70


def capture_frame_b64(mini: ReachyMini) -> str | None:
    """カメラ画像を取得し JPEG(base64) で返す。get_frame は BGR 順なのでそのまま encode。"""
    try:
        frame = None
        for _ in range(3):
            frame = mini.media.get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > FRAME_MAX_W:
            nh = int(h * FRAME_MAX_W / w)
            frame = cv2.resize(frame, (FRAME_MAX_W, nh))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        print(f"(frame capture skip: {e})", flush=True)
        return None


def play_wav_bytes(mini: ReachyMini, wav_bytes: bytes) -> None:
    """1 チャンクの WAV を再生し、再生完了までブロックする。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = Path(f.name)
    try:
        with wave.open(str(path), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        mini.media.play_sound(str(path))
        time.sleep(duration)
    finally:
        path.unlink(missing_ok=True)


def handle_turn(mini: ReachyMini, ws, motion: MotionController) -> None:
    """turn_end まで受信し、音声を逐次再生・motion を実行する。"""
    # 再生中のフィードバック回避のため録音を止める
    mini.media.stop_recording()
    motion.set_mode("talking")  # 応答中は少し活発に動く
    try:
        while True:
            msg = ws.recv(timeout=RECV_TIMEOUT)
            if isinstance(msg, (bytes, bytearray)):
                print("🔊 再生中...", flush=True)
                play_wav_bytes(mini, bytes(msg))
                continue
            ev = json.loads(msg)
            etype = ev.get("type")
            if etype == "stt":
                print(f"🗣  User: {ev.get('text')}", flush=True)
            elif etype == "motion":
                print(f"🤖 motion: {ev.get('name')}", flush=True)
                motion.gesture(ev["name"])
            elif etype == "turn_end":
                break
    finally:
        motion.set_mode("idle")
        time.sleep(POST_PLAYBACK_DELAY)
        mini.media.start_recording()


def main() -> None:
    print(f"接続先: {WS_URL}")
    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        in_sr = mini.media.get_input_audio_samplerate()
        motion = MotionController(mini)
        print(f"準備完了 (mic sr={in_sr}). 話しかけてください。Ctrl+C で終了。\n")

        with connect(WS_URL, max_size=None) as ws:
            try:
                while True:
                    audio = collect_utterance(mini, in_sr)
                    if audio is None:
                        print("(短すぎてスキップ)", flush=True)
                        continue
                    mono = to_mono_float32(audio)
                    # 発話に合わせて直近のカメラ画像を添えて送る
                    frame_b64 = capture_frame_b64(mini)
                    if frame_b64:
                        ws.send(json.dumps({"type": "frame", "jpeg_b64": frame_b64}))
                    print(
                        f"📡 送信中 ({len(mono) / in_sr:.1f}s"
                        f"{', 📷' if frame_b64 else ''})...",
                        flush=True,
                    )
                    ws.send(mono.tobytes())
                    handle_turn(mini, ws, motion)
                    print()
            except KeyboardInterrupt:
                print("\n終了します。")
            finally:
                motion.stop()
                mini.media.stop_recording()
                mini.media.stop_playing()


if __name__ == "__main__":
    main()
