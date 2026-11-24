"""ロボットの入出力の抽象（設計書 §3.1）。SDK への依存を 1 か所（:mod:`.reachy`）に閉じ込める。

ほかのモジュールは ``RobotIO`` だけを見る。実機は :class:`~reachy_front.io.reachy.ReachyRobotIO`、
テストと再生は :class:`~reachy_front.io.fake.FakeRobotIO`（実機なしで CI で動く）。

- メソッドはすべて async（SDK の同期呼び出しは実装側でスレッドに逃がす）
- 値は標準ライブラリだけの型（numpy や cv2 は知覚の側で ``bytes`` から作る）
- 読み取りは「いまのぶん」を返し、無ければ ``None``（USB の一過性のエラーも ``None``。例外で落とさない）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AudioFrame:
    """マイクの 1 チャンク。モノラル、float32 リトルエンディアン（プロトコルの audio_in と同じ形式）。"""

    pcm_f32le: bytes
    sample_rate: int

    @property
    def n_samples(self) -> int:
        return len(self.pcm_f32le) // 4

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass(frozen=True)
class DoAReading:
    """音の到来方向。``angle_deg`` は SDK の角度（ラジアン）を度に直しただけ（座標系の対応は実機で確認）。"""

    angle_deg: float | None
    speech: bool = False


@dataclass(frozen=True)
class CameraFrame:
    """カメラの 1 枚。BGR・8 bit・行優先（``len(bgr) == width * height * 3``）。"""

    width: int
    height: int
    bgr: bytes

    def __post_init__(self) -> None:
        if len(self.bgr) != self.width * self.height * 3:
            raise ValueError("画像のサイズと bgr の長さが合わない")


@dataclass(frozen=True)
class HeadPose:
    """頭の姿勢。回転は度、``z_mm`` は ミリメートル。すべて 0 が中立。"""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    z_mm: float = 0.0


Antennas = tuple[float, float]
"""左右のアンテナの角度（度）。"""


@runtime_checkable
class RobotIO(Protocol):
    async def start(self) -> None:
        """マイク・スピーカーを開始する。"""
        ...

    async def stop(self) -> None: ...

    @property
    def input_sample_rate(self) -> int:
        """マイクのサンプルレート（``start`` の後で確定する）。"""
        ...

    # ---- 入力
    async def read_audio(self) -> AudioFrame | None: ...

    async def read_doa(self) -> DoAReading | None: ...

    async def read_frame(self) -> CameraFrame | None: ...

    # ---- 出力
    async def play_wav(self, wav: bytes) -> None:
        """WAV（PCM16）を再生し、**再生が終わるまで**待つ。"""
        ...

    async def stop_playback(self) -> None:
        """再生中の音を止める（割り込み）。"""
        ...

    async def set_mic_enabled(self, enabled: bool) -> None:
        """マイクの入／切。自分の声を拾わないよう、再生の間は切る使い方を想定。"""
        ...

    async def set_pose(self, head: HeadPose, antennas: Antennas = (0.0, 0.0)) -> None:
        """目標の姿勢を即座に与える（20 Hz 程度で連続して呼ぶ）。"""
        ...

    async def goto_pose(
        self, head: HeadPose, antennas: Antennas = (0.0, 0.0), duration_s: float = 0.3
    ) -> None:
        """``duration_s`` かけて姿勢へ動かす（終わるまで待つ）。"""
        ...


__all__ = [
    "Antennas",
    "AudioFrame",
    "CameraFrame",
    "DoAReading",
    "HeadPose",
    "RobotIO",
]
