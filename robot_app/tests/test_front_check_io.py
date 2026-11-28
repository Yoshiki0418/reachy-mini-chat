"""実機チェックの道具。Fake の入出力で、全項目が最後まで走り、例外で止まらないことを確かめる。"""

from __future__ import annotations

import io
import struct
import wave
from typing import ClassVar

from reachy_front.core import ManualClock
from reachy_front.io import AudioFrame, CameraFrame, DoAReading, FakeRobotIO, RobotIO
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


# ---------------------------------------------------------------- DoA の方向ごとの記録（Codex の指摘 #133）


class Directional(Endless):
    """話しかけた方向（プロンプトの【正面】【左】【右】）に応じて、DoA の角度を返す Fake。"""

    ANGLES: ClassVar[dict[str, float]] = {"正面": 90.0, "左": 150.0, "右": 30.0}

    def __init__(self, clock: ManualClock, silent_direction: str | None = None) -> None:
        super().__init__(frames=[CameraFrame(2, 1, b"\x00" * 6)])
        self.clock = clock
        self.direction = ""
        self.silent_direction = silent_direction

    async def read_audio(self) -> AudioFrame | None:
        self.clock.advance(0.01)  # 読むたびに時間が進む（実機のマイクの待ちの代わり）
        return await super().read_audio()

    async def read_doa(self) -> DoAReading | None:
        if self.direction == self.silent_direction or self.direction not in self.ANGLES:
            return DoAReading(None, False)
        return DoAReading(self.ANGLES[self.direction], True)


async def test_doa_is_recorded_per_direction_so_the_coordinate_mapping_is_known() -> None:
    clock = ManualClock()
    fake = Directional(clock)

    def out(line: str) -> None:
        for name in Directional.ANGLES:
            if f"【{name}】" in line:
                fake.direction = name  # オペレーターが、指示された方向から話しかける

    async def sleep(s: float) -> None:
        clock.advance(s)

    results = await run_checks(fake, out=out, listen_s=0.1, doa_s=1.0, sleep=sleep, clock=clock.now)
    doa = next(r for r in results if r.name.startswith("DoA"))
    # 方向ごとの角度が、方向の名前つきで残る（最小・最大だけでは、どの方向か分からない）
    assert "正面 90°" in doa.detail and "左 150°" in doa.detail and "右 30°" in doa.detail
    assert doa.detail.index("正面") < doa.detail.index("左") < doa.detail.index("右")


async def test_a_direction_without_detected_speech_is_reported_as_such() -> None:
    clock = ManualClock()
    fake = Directional(clock, silent_direction="左")

    def out(line: str) -> None:
        for name in Directional.ANGLES:
            if f"【{name}】" in line:
                fake.direction = name

    async def sleep(s: float) -> None:
        clock.advance(s)

    results = await run_checks(fake, out=out, listen_s=0.1, doa_s=1.0, sleep=sleep, clock=clock.now)
    doa = next(r for r in results if r.name.startswith("DoA"))
    assert "正面 90°" in doa.detail and "右 30°" in doa.detail
    assert "左 発話と判定されず" in doa.detail
