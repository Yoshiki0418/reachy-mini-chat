"""ロボットの入出力の抽象（マイク・DoA・カメラ・スピーカー・首・アンテナ）。設計書 §3.1。

実機は :class:`ReachyRobotIO`（``reachy_mini`` SDK。実機の環境でだけ使う）、
テスト・再生は :class:`FakeRobotIO`。ほかのモジュールは :class:`RobotIO` だけを見る。
"""

from __future__ import annotations

from .base import Antennas, AudioFrame, CameraFrame, DoAReading, HeadPose, RobotIO
from .fake import FakeRobotIO
from .reachy import ReachyRobotIO

__all__ = [
    "Antennas",
    "AudioFrame",
    "CameraFrame",
    "DoAReading",
    "FakeRobotIO",
    "HeadPose",
    "ReachyRobotIO",
    "RobotIO",
]
