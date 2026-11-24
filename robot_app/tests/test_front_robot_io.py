"""RobotIO: FakeRobotIO の振る舞いと、ReachyRobotIO の SDK 呼び出し（SDK の代役で検証）。"""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import Any

import pytest

from reachy_front.io import (
    AudioFrame,
    CameraFrame,
    DoAReading,
    FakeRobotIO,
    HeadPose,
    ReachyRobotIO,
    RobotIO,
)
from reachy_front.io.fake import (
    GotoPose,
    MicEnabled,
    PlayWav,
    SetPose,
    StopPlayback,
    wav_duration_s,
    wav_to_frames,
)


def make_wav(samples: list[int], sr: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def floats(frame: AudioFrame) -> list[float]:
    return list(struct.unpack(f"<{frame.n_samples}f", frame.pcm_f32le))


# ---------------------------------------------------------------- 型


def test_audio_frame_duration() -> None:
    f = AudioFrame(struct.pack("<4f", 0, 0, 0, 0), 8)
    assert (f.n_samples, f.duration_s) == (4, 0.5)


def test_camera_frame_checks_its_size() -> None:
    CameraFrame(2, 1, b"\x00" * 6)
    with pytest.raises(ValueError):
        CameraFrame(2, 2, b"\x00" * 6)


def test_both_implementations_satisfy_the_protocol() -> None:
    assert isinstance(FakeRobotIO(), RobotIO)
    assert isinstance(ReachyRobotIO(object()), RobotIO)


# ---------------------------------------------------------------- WAV → 入力


def test_wav_is_cut_into_mono_float_chunks() -> None:
    frames = wav_to_frames(make_wav([16384, -16384] * 800), chunk_s=0.05)  # 0.1 秒
    assert [f.n_samples for f in frames] == [800, 800]
    assert floats(frames[0])[:2] == [0.5, -0.5]
    assert frames[0].sample_rate == 16000


def test_stereo_wav_is_mixed_down() -> None:
    frames = wav_to_frames(make_wav([16384, 0, 0, -16384], channels=2), chunk_s=1)
    assert floats(frames[0]) == [0.25, -0.25]


def test_wav_duration() -> None:
    assert wav_duration_s(make_wav([0] * 8000)) == 0.5


# ---------------------------------------------------------------- FakeRobotIO


async def test_fake_streams_audio_with_matching_doa_then_runs_out() -> None:
    fake = FakeRobotIO.from_wav(make_wav([0] * 1600), chunk_s=0.05)  # 2 チャンク
    await fake.start()
    assert fake.input_sample_rate == 16000
    seen: list[tuple[bool, bool]] = []
    while (a := await fake.read_audio()) is not None:
        d = await fake.read_doa()
        assert d is not None
        seen.append((a.n_samples == 800, d.speech))
    assert seen == [(True, True), (True, True)]
    assert fake.audio_exhausted


async def test_fake_doa_follows_the_audio_chunk_just_read() -> None:
    chunk = AudioFrame(struct.pack("<1f", 0), 16000)
    fake = FakeRobotIO(
        audio=[chunk, chunk, chunk],
        doa=[DoAReading(10, False), DoAReading(20, True)],  # 3 つ目は最後の値のまま
    )
    out: list[DoAReading | None] = []
    for _ in range(3):
        await fake.read_audio()
        out.append(await fake.read_doa())
    assert [d.angle_deg for d in out if d] == [10, 20, 20]
    assert (await FakeRobotIO().read_doa()) == DoAReading(None, False)


async def test_fake_camera_advances_then_repeats_the_last_frame() -> None:
    a, b = CameraFrame(1, 1, b"\x01\x01\x01"), CameraFrame(1, 1, b"\x02\x02\x02")
    fake = FakeRobotIO(frames=[a, b])
    assert [await fake.read_frame() for _ in range(4)] == [a, b, b, b]
    assert await FakeRobotIO().read_frame() is None


async def test_muted_mic_returns_nothing() -> None:
    fake = FakeRobotIO.from_wav(make_wav([0] * 800))
    await fake.set_mic_enabled(False)
    assert await fake.read_audio() is None
    await fake.set_mic_enabled(True)
    assert await fake.read_audio() is not None


async def test_fake_records_commands_in_order() -> None:
    fake = FakeRobotIO()
    wav = make_wav([0] * 8000)
    await fake.set_mic_enabled(False)
    await fake.play_wav(wav)
    await fake.set_mic_enabled(True)
    await fake.stop_playback()
    await fake.set_pose(HeadPose(yaw=5), (1.0, -1.0))
    await fake.goto_pose(HeadPose(), duration_s=0.5)
    assert fake.commands == [
        MicEnabled(False),
        PlayWav(len(wav), 0.5),
        MicEnabled(True),
        StopPlayback(),
        SetPose(HeadPose(yaw=5), (1.0, -1.0)),
        GotoPose(HeadPose(), (0.0, 0.0), 0.5),
    ]
    assert len(fake.commands_of(PlayWav)) == 1


# ---------------------------------------------------------------- ReachyRobotIO（SDK の代役）


class Arr:
    """numpy の配列の代役（CI に numpy は無い）。2 次元（サンプル×チャンネル）か 1 次元。"""

    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    @property
    def ndim(self) -> int:
        return 2 if isinstance(self.rows[0], list) else 1

    def __len__(self) -> int:
        return len(self.rows)

    def mean(self, axis: int) -> Arr:
        return Arr([sum(r) / len(r) for r in self.rows])

    def astype(self, _dtype: str) -> Arr:
        return self

    def tobytes(self) -> bytes:
        return struct.pack(f"<{len(self.rows)}f", *self.rows)


class Img:
    def __init__(self, w: int, h: int) -> None:
        self.shape = (h, w, 3)
        self._w, self._h = w, h

    def tobytes(self) -> bytes:
        return b"\x07" * (self._w * self._h * 3)


class StubMedia:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sample: Any = Arr([[0.5, -0.5], [0.25, 0.25]])
        self.doa: Any = (math.pi / 2, True)
        self.frame: Any = Img(2, 1)
        self.fail = False
        self.played: list[tuple[str, bytes]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith(("start_", "stop_")):
            return lambda: self.calls.append(name)
        raise AttributeError(name)

    def get_input_audio_samplerate(self) -> int:
        return 16000

    def get_audio_sample(self) -> Any:
        if self.fail:
            raise OSError("usb")
        return self.sample

    def get_DoA(self) -> Any:
        if self.fail:
            raise OSError("usb")
        return self.doa

    def get_frame(self) -> Any:
        if self.fail:
            raise OSError("usb")
        return self.frame

    def play_sound(self, path: str) -> None:
        with open(path, "rb") as f:
            self.played.append((path, f.read()))


class StubMini:
    def __init__(self) -> None:
        self.media = StubMedia()
        self.targets: list[tuple[str, dict[str, Any]]] = []

    def set_target(self, **kw: Any) -> None:
        self.targets.append(("set", kw))

    def goto_target(self, **kw: Any) -> None:
        self.targets.append(("goto", kw))


async def started() -> tuple[ReachyRobotIO, StubMini]:
    mini = StubMini()
    rio = ReachyRobotIO(
        mini,
        pose_factory=lambda p: ("pose", p.roll, p.pitch, p.yaw, p.z_mm),
        antenna_factory=lambda a: ("ant", *a),
    )
    await rio.start()
    return rio, mini


async def test_start_opens_the_media_and_reads_the_sample_rate() -> None:
    rio, mini = await started()
    assert mini.media.calls == ["start_recording", "start_playing"]
    assert rio.input_sample_rate == 16000
    await rio.stop()
    assert mini.media.calls[-2:] == ["stop_recording", "stop_playing"]


async def test_audio_is_mixed_to_mono_float32() -> None:
    rio, _ = await started()
    frame = await rio.read_audio()
    assert frame is not None
    assert floats(frame) == [0.0, 0.25]
    assert frame.sample_rate == 16000


async def test_transient_read_errors_are_none_not_exceptions() -> None:
    rio, mini = await started()
    mini.media.fail = True
    assert await rio.read_audio() is None
    assert await rio.read_doa() is None
    assert await rio.read_frame() is None
    mini.media.fail = False
    mini.media.sample = Arr([])  # 空のサンプル
    mini.media.sample.rows = []
    assert await rio.read_audio() is None


async def test_doa_is_converted_to_degrees_and_frame_to_bgr_bytes() -> None:
    rio, _ = await started()
    doa = await rio.read_doa()
    assert doa is not None and doa.speech is True
    assert doa.angle_deg == pytest.approx(90.0)
    frame = await rio.read_frame()
    assert frame == CameraFrame(2, 1, b"\x07" * 6)


async def test_play_wav_writes_a_temp_file_waits_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rio, mini = await started()
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr("reachy_front.io.reachy.asyncio.sleep", fake_sleep)
    wav = make_wav([0] * 4000)
    await rio.play_wav(wav)
    path, content = mini.media.played[0]
    assert content == wav
    assert slept == [0.25]  # 再生の長さの分だけ待つ
    import os

    assert not os.path.exists(path)


async def test_mic_toggle_and_poses_use_the_sdk_conventions() -> None:
    rio, mini = await started()
    await rio.set_mic_enabled(False)
    await rio.set_mic_enabled(True)
    assert mini.media.calls[-2:] == ["stop_recording", "start_recording"]
    await rio.set_pose(HeadPose(pitch=3, yaw=-2, z_mm=1), (4.0, -4.0))
    await rio.goto_pose(HeadPose(), duration_s=0.3)
    assert mini.targets == [
        ("set", {"head": ("pose", 0, 3, -2, 1), "antennas": ("ant", 4.0, -4.0)}),
        (
            "goto",
            {
                "head": ("pose", 0, 0, 0, 0),
                "antennas": ("ant", 0.0, 0.0),
                "body_yaw": 0.0,
                "duration": 0.3,
            },
        ),
    ]


async def test_calls_before_start_fail_clearly() -> None:
    with pytest.raises(RuntimeError):
        await ReachyRobotIO().set_mic_enabled(True)
    with pytest.raises(RuntimeError):
        ReachyRobotIO(None)._media()  # pyright: ignore[reportPrivateUsage]
