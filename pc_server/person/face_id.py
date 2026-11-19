"""顔の埋め込み抽出 (YuNet 検出 + SFace 認識)。

受信 JPEG から最も大きい（＝手前にいる）顔を検出し、SFace の 128 次元特徴を返す。
特徴は L2 正規化して返すので、内積がそのままコサイン類似度になる。
モデル(onnx, 各〜数百KB)は OpenCV Zoo から初回に自動ダウンロードする。
"""

from __future__ import annotations

import base64
import os
import urllib.request

import cv2
import numpy as np

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def _ensure_model(url: str, name: str) -> str:
    path = os.path.join(_MODEL_DIR, name)
    if not os.path.exists(path):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        print(f"[person] モデルをダウンロード中… {name}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


class FaceIdentifier:
    def __init__(self, score_threshold: float = 0.7) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            _ensure_model(_YUNET_URL, "yunet.onnx"),
            "",
            (320, 320),
            score_threshold=score_threshold,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(
            _ensure_model(_SFACE_URL, "sface.onnx"), ""
        )

    def embed_bgr(self, frame: np.ndarray) -> np.ndarray | None:
        """BGR フレームから最大顔の L2 正規化済み 128 次元特徴を返す。顔が無ければ None。"""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        # 最も面積の大きい顔（＝手前の相手）を選ぶ。cols 2,3 が bbox の w,h。
        face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        aligned = self._recognizer.alignCrop(frame, face)
        feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat

    def embed_jpeg_b64(self, jpeg_b64: str) -> np.ndarray | None:
        """base64 JPEG から顔特徴を返す。デコード不能・顔なしは None。"""
        try:
            arr = np.frombuffer(base64.b64decode(jpeg_b64), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
            if frame is None:
                return None
            return self.embed_bgr(frame)
        except Exception as e:  # noqa: BLE001
            print(f"[person] 顔埋め込み失敗: {e}", flush=True)
            return None
