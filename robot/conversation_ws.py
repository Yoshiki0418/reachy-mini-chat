"""Reachy Mini 会話クライアント (WebSocket 版)。

旧 conversation.py (HTTP 1往復) を置き換える常時接続クライアント。
VAD で発話区間を検出 → PC へ送信 → 文単位で返る音声チャンクを逐次再生し、
motion 指示を Motion Executor で実行する。

実行:
    PC_HOST=<PC のホスト名>.local /venvs/apps_venv/bin/python robot/conversation_ws.py

依存: websockets (apps_venv に導入済み: 15.x)
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import wave
from pathlib import Path

import cv2

# 同ディレクトリの既存 VAD 実装と Motion Controller / Gaze を再利用
from conversation import collect_utterance, to_mono_float32
from gaze import GazeTracker
from motion import MotionController
from reachy_mini import ReachyMini
from websockets.sync.client import connect

GAZE_ENABLED = os.environ.get("GAZE_ENABLED", "1") == "1"

PC_HOST = os.environ.get("PC_HOST", "localhost")  # PC の mDNS 名などを環境変数で渡す
PC_PORT = int(os.environ.get("PC_PORT", "8765"))
WS_URL = f"ws://{PC_HOST}:{PC_PORT}/ws/session"

POST_PLAYBACK_DELAY = 0.4  # 再生後のエコー収まり待ち
RECV_TIMEOUT = 60.0
FRAME_MAX_W = 512  # 送信フレームの最大幅（VLM は detail=low なので小さくてよい）
FRAME_JPEG_QUALITY = 70

# --- 自発発話 / エンゲージメント (Phase 5+) ---
# タイマー式の話題ふりは廃止。相手がこちらを「じっと見ている（持続アイコンタクト）」時だけ
# ロボットの側から声をかける。安全のため顔検出(gaze)必須。
PROACTIVE_ENABLED = os.environ.get("PROACTIVE_ENABLED", "1") == "1"
# この秒数 目が合い続けたら声かけ
ENGAGE_EYE_SEC = float(os.environ.get("PROACTIVE_ENGAGE_SEC", "2.0"))
PROACTIVE_MIN_GAP = float(os.environ.get("PROACTIVE_MIN_GAP", "15"))  # 自発発話どうしの最小間隔
POLL_SEC = float(os.environ.get("PROACTIVE_POLL_SEC", "0.5"))  # アイコンタクト判定の粒度
# 発話を「自分に向けられた」とみなすのに必要な、発話時点までの継続アイコンタクト秒数。
# 一瞬こちらを向いただけ・傍聴中のちら見では応答しない。
GATE_EYE_SEC = float(os.environ.get("GATE_EYE_SEC", "0.5"))


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
    except Exception as e:
        print(f"(frame capture skip: {e})", flush=True)
        return None


def send_frame(ws, mini: ReachyMini) -> bool:
    """直近のカメラフレームを送る。送れたら True。"""
    frame_b64 = capture_frame_b64(mini)
    if frame_b64:
        ws.send(json.dumps({"type": "frame", "jpeg_b64": frame_b64}))
    return bool(frame_b64)


class ProactivePolicy:
    """持続アイコンタクトを見て、ロボットから声をかけるか判定する状態機械。

    相手が ENGAGE_EYE_SEC 以上こちらを見続けたら1回だけ "engage"（声かけ）を出す。
    同じアイコンタクトが続く間は再発火しない（相手が目を逸らすとリセット）。
    タイマー式の話題ふりは廃止。gaze が無い時は常に None（誰もいない/読めない時は喋らない）。
    """

    def __init__(self, gaze) -> None:
        self.gaze = gaze
        self.last_proactive = 0.0  # 最後の自発発話/やり取り時刻
        self.engaged = False  # 今のアイコンタクト継続中に既に関与したか

    def note_turn(self) -> None:
        """相手と実際にやり取りした（応答した）ことを記録。二重に割り込まないため。"""
        self.last_proactive = time.monotonic()
        self.engaged = True

    def note_proactive(self, reason: str) -> None:
        self.last_proactive = time.monotonic()
        self.engaged = True

    def decide(self) -> str | None:
        if self.gaze is None:
            return None
        if not self.gaze.eye_contact():
            # 目が逸れたら次のアイコンタクトで再び声をかけられるようリセット
            self.engaged = False
            return None
        now = time.monotonic()
        if (
            not self.engaged
            and self.gaze.eye_contact(min_duration=ENGAGE_EYE_SEC)
            and now - self.last_proactive >= PROACTIVE_MIN_GAP
        ):
            return "engage"
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


def handle_turn(mini: ReachyMini, ws, motion: MotionController) -> bool:
    """turn_end まで受信し、音声を逐次再生・motion を実行する。

    戻り値: このターンでロボットが実際に応答した（音声を受け取った）か。
    サーバが「割り込まない」と判断して無音で返した場合は False。
    """
    # 再生中のフィードバック回避のため録音を止める
    mini.media.stop_recording()
    motion.set_mode("talking")  # 応答中は少し活発に動く
    responded = False
    try:
        while True:
            msg = ws.recv(timeout=RECV_TIMEOUT)
            if isinstance(msg, (bytes, bytearray)):
                responded = True
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
    return responded


def main() -> None:
    print(f"接続先: {WS_URL}")
    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()
        in_sr = mini.media.get_input_audio_samplerate()
        motion = MotionController(mini)
        gaze = GazeTracker(mini, motion) if GAZE_ENABLED else None
        print(
            f"準備完了 (mic sr={in_sr}, gaze={'on' if gaze else 'off'}). "
            "話しかけてください。Ctrl+C で終了。\n"
        )

        proactive_on = PROACTIVE_ENABLED and gaze is not None
        policy = ProactivePolicy(gaze) if proactive_on else None
        # gaze がある間は POLL_SEC ごとに制御を戻し、アイコンタクトを判定する
        idle_timeout = POLL_SEC if proactive_on else None

        with connect(WS_URL, max_size=None) as ws:
            try:
                while True:
                    audio = collect_utterance(mini, in_sr, idle_timeout=idle_timeout)

                    if audio is not None:
                        # --- ユーザー発話ターン ---
                        mono = to_mono_float32(audio)
                        has_frame = send_frame(ws, mini)  # 発話に直近フレームを添える
                        # 発話時に「こちらを一定時間見続けていたか」＝自分に向けられた発話か
                        eye_contact = (
                            bool(gaze.eye_contact(min_duration=GATE_EYE_SEC)) if gaze else True
                        )
                        ws.send(json.dumps({"type": "utt_meta", "eye_contact": eye_contact}))
                        print(
                            f"📡 送信中 ({len(mono) / in_sr:.1f}s"
                            f"{', 📷' if has_frame else ''}"
                            f"{', 👁' if eye_contact else ''})...",
                            flush=True,
                        )
                        ws.send(mono.tobytes())
                        responded = handle_turn(mini, ws, motion)
                        # 実際に応答したやり取りだけを記録（無視した発話では割り込み判定を変えない）
                        if policy is not None and responded:
                            policy.note_turn()
                        print()
                        continue

                    # --- 発話なし: 持続アイコンタクトなら自分から声かけ ---
                    if policy is None:
                        continue
                    reason = policy.decide()
                    if reason is None:
                        continue
                    has_frame = send_frame(ws, mini)
                    print(
                        f"💡 声かけ ({reason}{', 📷' if has_frame else ''})...",
                        flush=True,
                    )
                    ws.send(json.dumps({"type": "proactive", "reason": reason}))
                    handle_turn(mini, ws, motion)
                    policy.note_proactive(reason)
                    print()
            except KeyboardInterrupt:
                print("\n終了します。")
            finally:
                if gaze is not None:
                    gaze.stop()
                motion.stop()
                mini.media.stop_recording()
                mini.media.stop_playing()


if __name__ == "__main__":
    main()
