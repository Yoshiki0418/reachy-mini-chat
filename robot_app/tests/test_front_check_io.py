"""実機チェックの道具。Fake の入出力で、全項目が最後まで走り、例外で止まらないことを確かめる。"""

from __future__ import annotations

import io
import struct
import wave

from reachy_front.io import AudioFrame, CameraFrame, FakeRobotIO, RobotIO
from reachy_front.io.fake import GotoPose, PlayWav
from reachy_front.tools.check_io import beep_wav, run_checks


async def no_sleep(_s: float) -> None:
    return None


class Endless(FakeRobotIO):
    """マイクは尽きず、カメラの絵もある、実機に近い Fake。"""

    async def read_audio(self) -> AudioFrame | None:
        if not self.mic_enabled:
            return None
        return AudioFrame(struct.pack("<160f", *([0.0] * 160)), 16000)


def long_input() -> FakeRobotIO:
    return Endless(frames=[CameraFrame(2, 1, b"\x00" * 6)])


def test_beep_is_a_valid_short_wav() -> None:
    with wave.open(io.BytesIO(beep_wav(0.6))) as w:
        assert (w.getnchannels(), w.getsampwidth()) == (1, 2)
        assert abs(w.getnframes() / w.getframerate() - 0.6) < 0.01


async def test_all_checks_run_against_a_fake_robot() -> None:
    fake = long_input()
    lines: list[str] = []
    results = await run_checks(fake, out=lines.append, listen_s=0.05, doa_s=0.05, sleep=no_sleep)
    assert isinstance(fake, RobotIO)
    names = [r.name for r in results]
    assert "開始・マイクのサンプルレート" in names
    assert "首・アンテナ" in names and "再生の停止" in names
    assert all(r.status != "NG" for r in results), [r for r in results if r.status == "NG"]
    # ビープを鳴らし、首を動かした
    assert any(isinstance(c, PlayWav) for c in fake.commands)
    assert sum(isinstance(c, GotoPose) for c in fake.commands) >= 4
    assert lines[-1].strip().startswith("結果:")


async def test_a_failing_device_is_reported_as_ng_not_raised() -> None:
    class Broken(FakeRobotIO):
        async def read_frame(self):  # type: ignore[no-untyped-def]
            raise OSError("camera gone")

    results = await run_checks(
        Broken(), out=lambda _s: None, listen_s=0.01, doa_s=0.01, sleep=no_sleep
    )
    camera = next(r for r in results if r.name == "カメラ")
    assert camera.status == "NG" and "camera gone" in camera.detail
    assert any(r.name == "首・アンテナ" for r in results)  # 1 つ壊れても、残りは続ける
