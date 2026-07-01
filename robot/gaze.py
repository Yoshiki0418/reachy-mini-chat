"""Gaze Tracker (ロボット側 reflex 層)。

カメラ画像から顔をローカル検出（cv2 Haar）し、顔が画面中央へ来るように
MotionController の視線 DC 成分（set_gaze）を更新する閉ループ。カメラは頭部に
載っているので、頭を回すと顔が中央へ寄る → 積分的な比例制御で収束する。

- 検出は幅を縮小して高速化（全解像度は ~320ms/frame と遅いため）。
- 顔を見失ったら視線をゆっくり正面へ戻す。
- 符号はハード依存で誤りやすいので env (GAZE_YAW_SIGN / GAZE_PITCH_SIGN) で反転可能。
"""

from __future__ import annotations

import os
import threading
import time

import cv2

from reachy_mini import ReachyMini

DETECT_W = 320            # 検出用に縮小する幅
RATE_HZ = 10.0
KP_YAW = 6.0             # 1 ステップあたりの補正ゲイン（度 / 正規化オフセット）
KP_PITCH = 5.0
YAW_LIMIT = 30.0
PITCH_LIMIT = 18.0
LOST_GRACE_SEC = 0.5     # この時間 顔を見失ったら正面へ戻し始める
DECAY = 0.9              # 見失い時に視線を 0 へ戻す係数（/ステップ）

YAW_SIGN = float(os.environ.get("GAZE_YAW_SIGN", "1"))
PITCH_SIGN = float(os.environ.get("GAZE_PITCH_SIGN", "1"))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class GazeTracker:
    def __init__(self, mini: ReachyMini, motion) -> None:
        self.mini = mini
        self.motion = motion
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._yaw = 0.0
        self._pitch = 0.0
        self._last_seen = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.motion.set_gaze(0.0, 0.0)

    def _detect_offset(self, frame) -> tuple[float, float] | None:
        """最大の顔を検出し、中央からの正規化オフセット (dx, dy) を返す。無ければ None。

        dx, dy は半幅/半高で割った [-1, 1]（右/下が正）。
        """
        h, w = frame.shape[:2]
        scale = DETECT_W / w
        small = cv2.resize(frame, (DETECT_W, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return None
        # 面積最大の顔を採用
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        cx = fx + fw / 2
        cy = fy + fh / 2
        sh, sw = gray.shape[:2]
        dx = (cx - sw / 2) / (sw / 2)
        dy = (cy - sh / 2) / (sh / 2)
        return dx, dy

    def _run(self) -> None:
        dt = 1.0 / RATE_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                frame = self.mini.media.get_frame()
            except Exception:
                frame = None

            if frame is not None:
                off = self._detect_offset(frame)
            else:
                off = None

            if off is not None:
                dx, dy = off
                self._yaw = _clamp(
                    self._yaw + YAW_SIGN * KP_YAW * dx, -YAW_LIMIT, YAW_LIMIT
                )
                self._pitch = _clamp(
                    self._pitch + PITCH_SIGN * KP_PITCH * dy, -PITCH_LIMIT, PITCH_LIMIT
                )
                self._last_seen = time.monotonic()
                self.motion.set_gaze(self._yaw, self._pitch)
            elif time.monotonic() - self._last_seen > LOST_GRACE_SEC:
                # 見失ったら正面へゆっくり戻す
                self._yaw *= DECAY
                self._pitch *= DECAY
                self.motion.set_gaze(self._yaw, self._pitch)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
