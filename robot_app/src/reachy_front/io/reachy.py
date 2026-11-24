"""ReachyRobotIO: Reachy Mini の SDK（``reachy_mini``）を使う :class:`~reachy_front.io.base.RobotIO`。

**SDK に依存するのはこのファイルだけ。** 実機の ``/venvs/apps_venv`` にだけ SDK と numpy があるので、
import は ``start()`` まで遅らせる（CI やテストでは ``mini`` と変換関数を差し込める）。SDK の同期
呼び出しはスレッドに逃がし、イベントループを止めない。呼び方は移行前の ``robot/`` に合わせてある。

実機でまだ確認していない点（#35 で確認する）:
- 再生を止める ``stop_playback`` の動き（``media.stop_playing`` の後に再開が要るか）
- DoA の角度の座標系
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from .base import Antennas, AudioFrame, CameraFrame, DoAReading, HeadPose
from .fake import wav_duration_s

log = logging.getLogger(__name__)

PoseFactory = Callable[[HeadPose], Any]
AntennaFactory = Callable[[Antennas], Any]


class ReachyRobotIO:
    def __init__(
        self,
        mini: Any | None = None,
        *,
        pose_factory: PoseFactory | None = None,
        antenna_factory: AntennaFactory | None = None,
    ) -> None:
        self._mini = mini
        self._owns_mini = mini is None
        self._pose_factory = pose_factory
        self._antenna_factory = antenna_factory
        self._stack = ExitStack()
        self._sample_rate = 0

    # ---------------------------------------------------------------- 開始と終了

    async def start(self) -> None:
        await asyncio.to_thread(self._start_sync)

    def _start_sync(self) -> None:
        if self._mini is None:
            from reachy_mini import ReachyMini  # pyright: ignore[reportMissingImports]

            self._mini = self._stack.enter_context(ReachyMini())
        if self._pose_factory is None:
            from reachy_mini.utils import create_head_pose  # pyright: ignore[reportMissingImports]

            self._pose_factory = lambda p: create_head_pose(  # pyright: ignore[reportUnknownLambdaType]
                roll=p.roll, pitch=p.pitch, yaw=p.yaw, z=p.z_mm, mm=True, degrees=True
            )
        if self._antenna_factory is None:
            import numpy as np  # pyright: ignore[reportMissingImports]

            self._antenna_factory = lambda a: np.deg2rad(list(a))  # pyright: ignore[reportUnknownLambdaType]
        media = self._mini.media
        media.start_recording()
        media.start_playing()
        self._sample_rate = int(media.get_input_audio_samplerate())

    async def stop(self) -> None:
        await asyncio.to_thread(self._stop_sync)

    def _stop_sync(self) -> None:
        if self._mini is None:
            return
        media = self._mini.media
        for fn in (media.stop_recording, media.stop_playing):
            with contextlib.suppress(Exception):
                fn()
        if self._owns_mini:
            self._stack.close()
            self._mini = None

    @property
    def input_sample_rate(self) -> int:
        return self._sample_rate

    # ---------------------------------------------------------------- 入力（一過性の失敗は None）

    async def read_audio(self) -> AudioFrame | None:
        return await asyncio.to_thread(self._read_audio_sync)

    def _read_audio_sync(self) -> AudioFrame | None:
        try:
            sample = self._media().get_audio_sample()
        except Exception as e:  # 一過性の USB I/O エラーでループを落とさない
            log.debug("audio read skip: %s", e)
            return None
        if sample is None or len(sample) == 0:
            return None
        mono = sample.mean(axis=1) if sample.ndim == 2 else sample
        return AudioFrame(mono.astype("<f4").tobytes(), self._sample_rate)

    async def read_doa(self) -> DoAReading | None:
        return await asyncio.to_thread(self._read_doa_sync)

    def _read_doa_sync(self) -> DoAReading | None:
        try:
            doa = self._media().get_DoA()
        except Exception:  # 一過性の USB エラーは「発話なし」として継続
            return None
        if doa is None:
            return None
        angle, speech = doa[0], doa[1]
        return DoAReading(None if angle is None else math.degrees(float(angle)), bool(speech))

    async def read_frame(self) -> CameraFrame | None:
        return await asyncio.to_thread(self._read_frame_sync)

    def _read_frame_sync(self) -> CameraFrame | None:
        try:
            frame = self._media().get_frame()
        except Exception:
            return None
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return CameraFrame(int(w), int(h), frame.tobytes())  # get_frame は BGR 順

    # ---------------------------------------------------------------- 出力

    async def play_wav(self, wav: bytes) -> None:
        duration = wav_duration_s(wav)
        path = await asyncio.to_thread(self._write_temp, wav)
        try:
            await asyncio.to_thread(self._media().play_sound, path)
            await asyncio.sleep(duration)  # play_sound は戻ってくるので、長さの分だけ待つ
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    @staticmethod
    def _write_temp(wav: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            return f.name

    async def stop_playback(self) -> None:
        def stop() -> None:
            media = self._media()
            media.stop_playing()
            media.start_playing()  # 次の再生のために開け直す（実機で要確認）

        with contextlib.suppress(Exception):
            await asyncio.to_thread(stop)

    async def set_mic_enabled(self, enabled: bool) -> None:
        media = self._media()
        await asyncio.to_thread(media.start_recording if enabled else media.stop_recording)

    async def set_pose(self, head: HeadPose, antennas: Antennas = (0.0, 0.0)) -> None:
        await asyncio.to_thread(self._set_pose_sync, head, antennas)

    def _set_pose_sync(self, head: HeadPose, antennas: Antennas) -> None:
        assert self._mini is not None and self._pose_factory and self._antenna_factory
        self._mini.set_target(
            head=self._pose_factory(head), antennas=self._antenna_factory(antennas)
        )

    async def goto_pose(
        self, head: HeadPose, antennas: Antennas = (0.0, 0.0), duration_s: float = 0.3
    ) -> None:
        await asyncio.to_thread(self._goto_pose_sync, head, antennas, duration_s)

    def _goto_pose_sync(self, head: HeadPose, antennas: Antennas, duration_s: float) -> None:
        assert self._mini is not None and self._pose_factory and self._antenna_factory
        self._mini.goto_target(
            head=self._pose_factory(head),
            antennas=self._antenna_factory(antennas),
            body_yaw=0.0,
            duration=duration_s,
        )

    def _media(self) -> Any:
        if self._mini is None:
            raise RuntimeError("start() の前に呼ばれた")
        return self._mini.media


__all__ = ["ReachyRobotIO"]
