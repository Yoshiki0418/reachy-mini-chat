"""Gaze Tracker + アイコンタクト推定 (ロボット側 reflex 層)。

カメラ画像から顔をローカル検出（cv2 の YuNet DNN = FaceDetectorYN）し、
1) 顔（目の中点）が画面中央へ来るように MotionController の視線 DC 成分を更新する閉ループ、
2) 相手がこちらを見ているか（アイコンタクト）を推定して公開する、の2役を担う。

アイコンタクトは「顔が正面向き（鼻が両目の中点付近）× 十分近い × 画面中央寄り」で近似する。
真の視線ではないが、ロボットは相手の方へ頭を向ける（gaze 追従）ので、相手がこちらを見れば
カメラには正面・中央・大きめの顔が映る、という前提で実用的に判定する。
"""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.request

import cv2

from reachy_mini import ReachyMini

DETECT_W = 320            # 検出用に縮小する幅
RATE_HZ = 15.0
# 視線補正ゲイン（度 / 正規化オフセット / ステップ）。閉ループなので小さめで滑らかに収束。
KP_YAW = float(os.environ.get("GAZE_KP_YAW", "3.0"))
KP_PITCH = float(os.environ.get("GAZE_KP_PITCH", "2.5"))
YAW_LIMIT = 30.0
PITCH_LIMIT = 25.0
DEADZONE = 0.06
LOST_GRACE_SEC = 0.5
DECAY = 0.9

# 実機で確認した正しい向き: yaw は反転、pitch はそのまま（env で上書き可）。
YAW_SIGN = float(os.environ.get("GAZE_YAW_SIGN", "-1"))
PITCH_SIGN = float(os.environ.get("GAZE_PITCH_SIGN", "1"))
DEBUG = os.environ.get("GAZE_DEBUG", "0") == "1"

# --- アイコンタクト判定のしきい値（実機で要調整・env で上書き可）---
# 「相手が本当にこちらを見ている」時だけ True になるよう保守的に。作業中の下向き顔などは弾く。
EC_FRONTAL_MAX = float(os.environ.get("EC_FRONTAL_MAX", "0.35"))   # |鼻-両目中点|/両目間（左右の正面度）
EC_LEVEL_MAX = float(os.environ.get("EC_LEVEL_MAX", "0.35"))      # 両目の高さズレ/両目間（傾き）
EC_FACE_MIN_FRAC = float(os.environ.get("EC_FACE_MIN_FRAC", "0.14"))  # 顔幅/画面幅（近接＝関心の表れ）
EC_CENTER_MAX = float(os.environ.get("EC_CENTER_MAX", "0.35"))    # |dx|,|dy|（こちらへ顔を向けている）
# 鼻が両目より下にある度合い (nose_y-eye_y)/両目間。正面upright なら概ね 0.4〜1.1。
# 下を向く（作業中）と鼻が目に近づき小さくなる → 弾く。上向きすぎも弾く。
EC_VERT_MIN = float(os.environ.get("EC_VERT_MIN", "0.15"))
EC_VERT_MAX = float(os.environ.get("EC_VERT_MAX", "1.6"))
EC_GRACE = float(os.environ.get("EC_GRACE", "0.4"))              # この時間 生検知が無いとアイコンタクト解除

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
        self._detector = cv2.FaceDetectorYN.create(
            model_path, "", (DETECT_W, DETECT_W), score_threshold=0.7
        )
        self._input_size: tuple[int, int] | None = None
        self._yaw = 0.0
        self._pitch = 0.0
        self._last_seen = 0.0
        # アイコンタクト状態
        self._ec_since = 0.0     # 現在のアイコンタクト継続の開始時刻
        self._ec_last = 0.0      # 最後にアイコンタクト「生」検知した時刻
        self._num_faces = 0
        self._last_metrics = (0.0, 0.0, 0.0, 0.0)  # (nose_ratio, level, vert, face_frac) デバッグ用
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.motion.set_gaze(0.0, 0.0)

    def seconds_since_face_seen(self) -> float:
        return time.monotonic() - self._last_seen

    def face_present(self, within: float = 1.5) -> bool:
        return self.seconds_since_face_seen() < within

    def num_faces(self) -> int:
        return self._num_faces

    def eye_contact(self, min_duration: float = 0.0) -> bool:
        """相手がこちらを見ているか。min_duration を渡すと、その秒数以上継続していれば True。"""
        now = time.monotonic()
        if now - self._ec_last > EC_GRACE:
            return False
        return (now - self._ec_since) >= min_duration

    def _analyze(self, frame):
        """最大顔の (dx, dy, frontal, face_frac, num_faces) を返す。顔が無ければ None。"""
        h, w = frame.shape[:2]
        scale = DETECT_W / w
        sw, sh = DETECT_W, int(h * scale)
        small = cv2.resize(frame, (sw, sh))
        if self._input_size != (sw, sh):
            self._detector.setInputSize((sw, sh))
            self._input_size = (sw, sh)
        _, faces = self._detector.detect(small)
        n = 0 if faces is None else len(faces)
        self._num_faces = n
        if n == 0:
            return None
        # 最も面積の大きい顔（＝手前の相手）
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        rex, rey, lex, ley, nx, ny = (face[4], face[5], face[6], face[7], face[8], face[9])
        cx = (rex + lex) / 2
        cy = (rey + ley) / 2
        if not (0 <= cx <= sw and 0 <= cy <= sh):
            cx = face[0] + face[2] / 2
            cy = face[1] + face[3] / 2
        dx = (cx - sw / 2) / (sw / 2)
        dy = (cy - sh / 2) / (sh / 2)
        eye_dist = math.hypot(lex - rex, ley - rey)
        frontal = False
        if eye_dist > 1.0:
            nose_ratio = abs(nx - cx) / eye_dist       # 左右の正面度
            level = abs(rey - ley) / eye_dist          # 顔の傾き
            vert = (ny - cy) / eye_dist                # 鼻が両目より下にある度合い（上下の向き）
            frontal = (
                nose_ratio < EC_FRONTAL_MAX
                and level < EC_LEVEL_MAX
                and EC_VERT_MIN <= vert <= EC_VERT_MAX
            )
            self._last_metrics = (nose_ratio, level, vert, float(face[2]) / sw)
        face_frac = float(face[2]) / sw
        return dx, dy, frontal, face_frac, n

    def _run(self) -> None:
        dt = 1.0 / RATE_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                frame = self.mini.media.get_frame()
            except Exception:
                frame = None

            res = self._analyze(frame) if frame is not None else None
            now = time.monotonic()

            if res is not None:
                dx, dy, frontal, face_frac, n = res
                off = (dx, dy)
                # アイコンタクト「生」判定
                ec_raw = (
                    frontal
                    and face_frac >= EC_FACE_MIN_FRAC
                    and abs(dx) < EC_CENTER_MAX
                    and abs(dy) < EC_CENTER_MAX
                )
                if ec_raw:
                    if now - self._ec_last > EC_GRACE:
                        self._ec_since = now  # 新しいアイコンタクト episode 開始
                    self._ec_last = now
            else:
                off = None

            if DEBUG and (int(now * 3) != int((t0 - dt) * 3)):
                ec = self.eye_contact()
                nr, lv, vt, ff = self._last_metrics
                print(
                    f"[gaze] faces={self._num_faces} "
                    f"nose={nr:.2f} level={lv:.2f} vert={vt:.2f} facefrac={ff:.2f} "
                    f"eye_contact={ec}",
                    flush=True,
                )

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
                self._last_seen = now
                self.motion.set_gaze(self._yaw, self._pitch)
            elif now - self._last_seen > LOST_GRACE_SEC:
                self._yaw *= DECAY
                self._pitch *= DECAY
                self.motion.set_gaze(self._yaw, self._pitch)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
