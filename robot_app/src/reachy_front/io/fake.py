"""FakeRobotIO: 実機なしで :class:`~reachy_front.io.base.RobotIO` の代わりをする（テスト・再生用）。

- 入力は台本どおり（WAV・画像列・DoA）。尽きたら、音声は ``None``、DoA と画像は最後の値を返す
- 出力は ``commands`` に記録するだけ（再生の長さは待たない。待たせたいときは ``play_delay_s``）
"""

from __future__ import annotations

import asyncio
import io
import struct
import wave
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .base import Antennas, AudioFrame, CameraFrame, DoAReading, HeadPose


@dataclass(frozen=True)
class Started:
    pass


@dataclass(frozen=True)
class Stopped:
    pass


@dataclass(frozen=True)
class PlayWav:
    nbytes: int
    duration_s: float


@dataclass(frozen=True)
class StopPlayback:
    pass


@dataclass(frozen=True)
class MicEnabled:
    enabled: bool


@dataclass(frozen=True)
class SetPose:
    head: HeadPose
    antennas: Antennas


@dataclass(frozen=True)
class GotoPose:
    head: HeadPose
    antennas: Antennas
    duration_s: float


Command = Started | Stopped | PlayWav | StopPlayback | MicEnabled | SetPose | GotoPose


def wav_to_frames(wav: bytes, chunk_s: float = 0.05) -> list[AudioFrame]:
    """WAV（PCM16、モノラルまたはステレオ）を、``chunk_s`` 秒ごとのモノラル float32 に切る。"""
    with wave.open(io.BytesIO(wav)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("PCM16 の WAV だけ扱える")
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    ints = struct.unpack(f"<{len(raw) // 2}h", raw)
    mono = [sum(ints[i : i + ch]) / ch / 32768.0 for i in range(0, len(ints), ch)]
    step = max(1, int(sr * chunk_s))
    return [
        AudioFrame(struct.pack(f"<{len(c)}f", *c), sr)
        for i in range(0, len(mono), step)
        if (c := mono[i : i + step])
    ]


def wav_duration_s(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


@dataclass
class FakeRobotIO:
    audio: Sequence[AudioFrame] = ()
    doa: Sequence[DoAReading] = ()
    """``audio`` の i 番目を読んだ直後の ``read_doa`` が、``doa[i]`` を返す（足りなければ最後の値、無ければ無音）。"""
    frames: Sequence[CameraFrame] = ()
    """``read_frame`` ごとに 1 枚ずつ進み、尽きたら最後の 1 枚を返し続ける（カメラは常に絵がある）。"""
    sample_rate: int = 16000
    play_delay_s: float = 0.0

    commands: list[Command] = field(default_factory=list[Command])
    mic_enabled: bool = True
    started: bool = False
    _audio_pos: int = field(default=0, init=False)
    _frame_pos: int = field(default=0, init=False)

    @classmethod
    def from_wav(
        cls, wav: bytes, *, chunk_s: float = 0.05, speech: bool = True, **kw: object
    ) -> FakeRobotIO:
        """WAV をマイクの入力として流す。``speech`` なら、全チャンクを「発話中」の DoA にする。"""
        frames = wav_to_frames(wav, chunk_s)
        sr = frames[0].sample_rate if frames else 16000
        io_ = cls(
            audio=frames,
            doa=[DoAReading(90.0, speech)] * len(frames),
            sample_rate=sr,
            **kw,  # pyright: ignore[reportArgumentType]
        )
        return io_

    @property
    def input_sample_rate(self) -> int:
        return self.sample_rate

    @property
    def audio_exhausted(self) -> bool:
        return self._audio_pos >= len(self.audio)

    def queue_audio(self, frames: Iterable[AudioFrame], doa: Iterable[DoAReading] = ()) -> None:
        """あとから入力を足す（会話の続きのテスト用）。"""
        # 既存の音声に DoA が足りなければ、まず ``read_doa`` の既定（最後の値、無ければ無音）で埋める。
        # そうしないと、足した DoA が先の音声のぶんにずれて当たってしまう
        pad = self.doa[-1] if self.doa else DoAReading(None, False)
        self.doa = [*self.doa, *([pad] * max(0, len(self.audio) - len(self.doa)))]
        self.audio = [*self.audio, *frames]
        self.doa = [*self.doa, *doa]

    async def start(self) -> None:
        self.started = True
        self.commands.append(Started())

    async def stop(self) -> None:
        self.started = False
        self.commands.append(Stopped())

    async def read_audio(self) -> AudioFrame | None:
        if not self.mic_enabled or self.audio_exhausted:
            return None
        frame = self.audio[self._audio_pos]
        self._audio_pos += 1
        return frame

    async def read_doa(self) -> DoAReading | None:
        if not self.doa:
            return DoAReading(None, False)
        i = min(max(self._audio_pos - 1, 0), len(self.doa) - 1)
        return self.doa[i]

    async def read_frame(self) -> CameraFrame | None:
        if not self.frames:
            return None
        frame = self.frames[min(self._frame_pos, len(self.frames) - 1)]
        self._frame_pos += 1
        return frame

    async def play_wav(self, wav: bytes) -> None:
        self.commands.append(PlayWav(len(wav), wav_duration_s(wav)))
        await asyncio.sleep(self.play_delay_s)

    async def stop_playback(self) -> None:
        self.commands.append(StopPlayback())

    async def set_mic_enabled(self, enabled: bool) -> None:
        self.mic_enabled = enabled
        self.commands.append(MicEnabled(enabled))

    async def set_pose(self, head: HeadPose, antennas: Antennas = (0.0, 0.0)) -> None:
        self.commands.append(SetPose(head, antennas))

    async def goto_pose(
        self, head: HeadPose, antennas: Antennas = (0.0, 0.0), duration_s: float = 0.3
    ) -> None:
        self.commands.append(GotoPose(head, antennas, duration_s))

    def commands_of(self, kind: type[Command]) -> list[Command]:
        return [c for c in self.commands if isinstance(c, kind)]


__all__ = [
    "Command",
    "FakeRobotIO",
    "GotoPose",
    "MicEnabled",
    "PlayWav",
    "SetPose",
    "Started",
    "StopPlayback",
    "Stopped",
    "wav_duration_s",
    "wav_to_frames",
]
