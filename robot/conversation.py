"""Reachy Mini 会話クライアント (ロボット側)。

mini.media のDoA `is_speech` を使って発話区間を検出 → PC側サーバーに送信 → 返却されたWAVを再生。

実行:
    PC_HOST=yoshikinoMacBook-Air.local /venvs/apps_venv/bin/python robot/conversation.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini

PC_HOST = os.environ.get("PC_HOST", "yoshikinoMacBook-Air.local")
PC_PORT = int(os.environ.get("PC_PORT", "8765"))
SERVER_URL = f"http://{PC_HOST}:{PC_PORT}/turn"

# VAD パラメータ
SILENCE_END_SEC = 0.8       # 発話終了判定: 連続無音
MIN_SPEECH_SEC = 0.4         # 短すぎは無視
MAX_SPEECH_SEC = 20.0        # 長すぎは強制終了
POST_PLAYBACK_DELAY = 0.4    # 再生後、エコー収まり待ち


def collect_utterance(mini: ReachyMini, in_sr: int) -> np.ndarray | None:
    """発話を検出して終端までのオーディオを返す。短すぎる場合は None。"""
    audio_chunks: list[np.ndarray] = []
    speech_started = False
    silent_samples = 0
    total_samples = 0

    silence_limit = int(SILENCE_END_SEC * in_sr)
    max_limit = int(MAX_SPEECH_SEC * in_sr)
    min_limit = int(MIN_SPEECH_SEC * in_sr)

    while True:
        try:
            sample = mini.media.get_audio_sample()
        except Exception as e:  # 一過性の USB I/O エラー等でループを落とさない
            print(f"(audio read skip: {e})", flush=True)
            time.sleep(0.01)
            continue
        if sample is None or len(sample) == 0:
            time.sleep(0.01)
            continue

        try:
            doa = mini.media.get_DoA()
        except Exception:  # DoA の一過性 USB エラーは「発話なし」として継続
            doa = None
        is_speech = doa is not None and doa[1]

        if not speech_started:
            if is_speech:
                speech_started = True
                audio_chunks.append(sample)
                total_samples = len(sample)
                silent_samples = 0
                print("🎤 発話中...", flush=True)
            # else: ドロップ
        else:
            audio_chunks.append(sample)
            total_samples += len(sample)
            if is_speech:
                silent_samples = 0
            else:
                silent_samples += len(sample)
                if silent_samples >= silence_limit:
                    break
            if total_samples >= max_limit:
                print("(最大時間到達)")
                break

    if total_samples < min_limit:
        return None
    return np.concatenate(audio_chunks, axis=0)


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)


def send_to_server(audio_mono: np.ndarray) -> bytes | None:
    """PC側に送信。レスポンスのWAVバイト列、または None (空応答時)。"""
    body = audio_mono.tobytes()
    req = urllib.request.Request(
        SERVER_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 204:
                return None
            return resp.read()
    except Exception as e:
        print(f"❌ 通信エラー: {e}", file=sys.stderr)
        return None


def play_wav_bytes(mini: ReachyMini, wav_bytes: bytes) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = Path(f.name)
    try:
        with wave.open(str(path), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        # 再生中は録音停止（フィードバック回避）
        mini.media.stop_recording()
        mini.media.play_sound(str(path))
        time.sleep(duration + POST_PLAYBACK_DELAY)
        mini.media.start_recording()
        # エコー収まり待ち
        time.sleep(0.2)
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    print(f"接続先: {SERVER_URL}")
    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        in_sr = mini.media.get_input_audio_samplerate()
        print(f"準備完了 (mic sr={in_sr}). 話しかけてください。Ctrl+C で終了。\n")

        try:
            while True:
                audio = collect_utterance(mini, in_sr)
                if audio is None:
                    print("(短すぎてスキップ)", flush=True)
                    continue
                mono = to_mono_float32(audio)
                print(f"📡 送信中 ({len(mono) / in_sr:.1f}s)...", flush=True)

                wav_bytes = send_to_server(mono)
                if wav_bytes is None:
                    print("(応答なし)", flush=True)
                    continue

                print("🔊 再生中...", flush=True)
                play_wav_bytes(mini, wav_bytes)
                print()
        except KeyboardInterrupt:
            print("\n終了します。")
        finally:
            mini.media.stop_recording()
            mini.media.stop_playing()


if __name__ == "__main__":
    main()
