"""Gaze Tracker (ロボット側 reflex 層)。

カメラ画像から顔をローカル検出（cv2 の YuNet DNN = FaceDetectorYN）し、顔（目の中点）が
画面中央へ来るように MotionController の視線 DC 成分（set_gaze）を更新する閉ループ。
カメラは頭部に載っているので、頭を回すと顔が中央へ寄る → 積分的な比例制御で収束する。

- YuNet は角度/照明に強く高速（~10ms）。Haar より取りこぼしが少なく滑らかに追従できる。
- 検出は幅を縮小して高速化。目の中点を狙うことで安定させる。
- 顔を見失ったら視線をゆっくり正面へ戻す。
- 符号はハード依存で誤りやすいので env (GAZE_YAW_SIGN / GAZE_PITCH_SIGN) で反転可能。
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request

import cv2

from reachy_mini import ReachyMini

DETECT_W = 320            # 検出用に縮小する幅
RATE_HZ = 15.0
# 補正ゲイン（度 / 正規化オフセット / ステップ）。閉ループなので小さめで滑らかに収束。
# 符号・ゲインはハード依存・体感依存なので env で再起動なしに調整できるようにする。
KP_YAW = float(os.environ.get("GAZE_KP_YAW", "3.0"))
KP_PITCH = float(os.environ.get("GAZE_KP_PITCH", "2.5"))
YAW_LIMIT = 30.0
PITCH_LIMIT = 18.0
DEADZONE = 0.06          # 中央付近は動かさない（微振動防止）
LOST_GRACE_SEC = 0.5     # この時間 顔を見失ったら正面へ戻し始める
DECAY = 0.9              # 見失い時に視線を 0 へ戻す係数（/ステップ）

YAW_SIGN = float(os.environ.get("GAZE_YAW_SIGN", "1"))
PITCH_SIGN = float(os.environ.get("GAZE_PITCH_SIGN", "1"))

# YuNet モデル（OpenCV Zoo, ~340KB）。未配置なら初回に自動ダウンロードする。
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_YUNET_PATH = os.path.join(
    os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"
)


def _ensure_yunet_model() -> str:
    if not os.path.exists(_YUNET_PATH):
        os.makedirs(os.path.dirname(_YUNET_PATH), exist_ok=True)
        print("[gaze] YuNet モデルをダウンロード中…", flush=True)
        urllib.request.urlretrieve(_YUNET_URL, _YUNET_PATH)
    return _YUNET_PATH


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class GazeTracker:
    def __init__(self, mini: ReachyMini, motion) -> None:
        self.mini = mini
        self.motion = motion
        model_path = _ensure_yunet_model()
        # 入力サイズは最初のフレームで確定させる
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (DETECT_W, DETECT_W), score_threshold=0.7
        )
        self._input_size: tuple[int, int] | None = None
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
        """最もスコアの高い顔を検出し、目の中点の正規化オフセット (dx, dy) を返す。

        dx, dy は半幅/半高で割った [-1, 1]（右/下が正）。顔が無ければ None。
        """
        h, w = frame.shape[:2]
        scale = DETECT_W / w
        sw, sh = DETECT_W, int(h * scale)
        small = cv2.resize(frame, (sw, sh))
        if self._input_size != (sw, sh):
            self._detector.setInputSize((sw, sh))
            self._input_size = (sw, sh)
        _, faces = self._detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        # スコア最大の顔（faces[:, -1] がスコア）
        face = max(faces, key=lambda f: f[-1])
        # 目の中点: cols 4,5=右目, 6,7=左目。無効なら bbox 中心にフォールバック。
        rex, rey, lex, ley = face[4], face[5], face[6], face[7]
        cx = (rex + lex) / 2
        cy = (rey + ley) / 2
        if not (0 <= cx <= sw and 0 <= cy <= sh):
            cx = face[0] + face[2] / 2
            cy = face[1] + face[3] / 2
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
                if abs(dx) < DEADZONE:
                    dx = 0.0
                if abs(dy) < DEADZONE:
                    dy = 0.0
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
