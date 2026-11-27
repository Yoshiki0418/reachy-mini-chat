"""サーバー断のフォールバック。単体（Fake の入出力）と、サーバーを落として戻す障害注入テスト。"""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr

from reachy_front.client import ClipCache, LinkConfig, ServerClient
from reachy_front.core import EventBus, ManualClock
from reachy_front.core import events as ev
from reachy_front.core.events import Event
from reachy_front.expression import FallbackConfig, FallbackController
from reachy_front.io import FakeRobotIO, HeadPose
from reachy_front.io.fake import GotoPose, MicEnabled, PlayWav, StopPlayback
from reachy_front.world import World
from reachy_protocol.clips import ClipInfo, ClipManifest
from reachy_protocol.testing.fake_server import FakeServer, FakeServerConfig
from reachy_protocol.transport import Transport, memory_pair

CLIPS = {
    "greeting-hello": ("greeting", b"RIFF-hello"),
    "greeting-morning": ("greeting", b"RIFF-morning"),
    "greeting-evening": ("greeting", b"RIFF-evening"),
    "notice-offline": ("notice", b"RIFF-notice"),
    "filler-etto": ("filler", b"RIFF-etto"),
}


class _Fetcher:
    def __init__(self, clips: dict[str, tuple[str, bytes]]) -> None:
        self.clips = clips

    async def manifest(self, if_none_match: str | None) -> ClipManifest | None:
        return ClipManifest(
            version="v1",
            clips=[
                ClipInfo.model_validate(
                    {
                        "id": i,
                        "kind": k,
                        "text": i,
                        "sha256": hashlib.sha256(w).hexdigest(),
                        "nbytes": len(w),
                    }
                )
                for i, (k, w) in self.clips.items()
            ],
        )

    async def clip(self, clip_id: str) -> bytes:
        return self.clips[clip_id][1]


async def make_cache(
    tmp_path: Path, clips: dict[str, tuple[str, bytes]] | None = None
) -> ClipCache:
    cache = ClipCache(tmp_path, _Fetcher(CLIPS if clips is None else clips))
    await cache.refresh()
    return ClipCache(tmp_path)  # ディスクから（サーバーなしで）読み直す


@dataclass
class Rig:
    clock: ManualClock
    bus: EventBus
    io: RecordingIO
    world: World
    ctrl: FallbackController
    events: list[Event]
    hour: list[int]

    async def settle(self) -> None:
        await self.bus.drain(1)
        for _ in range(200):
            if not self.ctrl.busy:
                break
            await asyncio.sleep(0.005)
        await self.bus.drain(1)

    def publish(self, e: Event) -> None:
        self.bus.publish(e)


class RecordingIO(FakeRobotIO):
    def __init__(self) -> None:
        super().__init__()
        self.played_wavs: list[bytes] = []

    async def play_wav(self, wav: bytes) -> None:
        self.played_wavs.append(wav)
        self.commands.append(PlayWav(len(wav), 0.0))
        await asyncio.sleep(self.play_delay_s)


@pytest.fixture
async def rig(tmp_path: Path) -> AsyncIterator[Rig]:
    """再生した WAV の中身を見られる版。"""
    cache = await make_cache(tmp_path)
    cache.load()
    clock = ManualClock(0.0)
    bus = EventBus(clock)
    io = RecordingIO()
    world = World()
    world.attach(bus)
    hour = [14]
    ctrl = FallbackController(
        io, bus, world, cache, clock=clock, config=FallbackConfig(), hour=lambda: hour[0]
    )
    ctrl.attach()
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    yield Rig(clock, bus, io, world, ctrl, events, hour)
    await ctrl.close()
    await bus.close()


# ---------------------------------------------------------------- 入室への反応


async def test_nothing_happens_during_the_grace_period(rig: Rig) -> None:
    rig.clock.advance(1.0)  # 切れて 1 秒（まだ通常の接続中かもしれない）
    rig.publish(ev.PersonEntered(track_id=1))
    await rig.settle()
    assert rig.io.commands == []
    assert not rig.ctrl.active


async def test_a_person_entering_during_an_outage_gets_a_greeting_and_a_nod(rig: Rig) -> None:
    rig.clock.advance(3.0)
    rig.publish(ev.PersonEntered(track_id=1))
    await rig.settle()
    assert rig.io.played_wavs == [b"RIFF-hello"]
    cmds = rig.io.commands
    # 自分の声を拾わないよう、再生の前後でマイクを切って戻す
    mic = [c.enabled for c in cmds if isinstance(c, MicEnabled)]
    assert mic == [False, True]
    nods = [c for c in cmds if isinstance(c, GotoPose)]
    assert [n.head for n in nods] == [HeadPose(pitch=15.0), HeadPose()]
    assert any(isinstance(e, ev.RobotSpeechStarted) for e in rig.events)
    assert any(isinstance(e, ev.RobotSpeechEnded) for e in rig.events)
    assert rig.world.snapshot().robot.speaking is False


@pytest.mark.parametrize(
    ("hour", "expected"), [(7, b"RIFF-morning"), (14, b"RIFF-hello"), (19, b"RIFF-evening")]
)
async def test_the_greeting_depends_on_the_time_of_day(
    rig: Rig, hour: int, expected: bytes
) -> None:
    rig.hour[0] = hour
    rig.clock.advance(5.0)
    rig.publish(ev.PersonEntered(track_id=1))
    await rig.settle()
    assert rig.io.played_wavs == [expected]


async def test_one_greeting_per_person_and_spacing_between_people(rig: Rig) -> None:
    rig.clock.advance(5.0)
    rig.publish(ev.PersonEntered(track_id=1))
    await rig.settle()
    rig.publish(ev.PersonEntered(track_id=1))  # 同じ人の重複通知
    rig.publish(ev.PersonEntered(track_id=2))  # 続けて別の人（間隔内）
    await rig.settle()
    assert len(rig.io.played_wavs) == 1
    rig.clock.advance(10.0)
    rig.publish(ev.PersonEntered(track_id=2))  # 間隔が空けば、別の人には挨拶する
    await rig.settle()
    assert len(rig.io.played_wavs) == 2
    rig.clock.advance(10.0)
    rig.publish(ev.PersonLeft(track_id=1))
    rig.publish(ev.PersonEntered(track_id=1))  # 出て戻った人には、また挨拶する
    await rig.settle()
    assert len(rig.io.played_wavs) == 3


async def test_busy_reactions_are_dropped_not_queued(tmp_path: Path) -> None:
    cache = await make_cache(tmp_path)
    cache.load()
    clock = ManualClock(10.0)
    bus = EventBus(clock)
    io = RecordingIO()
    io.play_delay_s = 0.1
    world = World()
    world.attach(bus)
    ctrl = FallbackController(
        io,
        bus,
        world,
        cache,
        clock=clock,
        config=FallbackConfig(greeting_min_gap_s=0.0),
        hour=lambda: 14,
    )
    ctrl.attach()
    ctrl._down_since = 0.0  # pyright: ignore[reportPrivateUsage]
    bus.publish(ev.PersonEntered(track_id=1))
    await asyncio.sleep(0.03)  # 1 人目の再生の最中
    bus.publish(ev.PersonEntered(track_id=2))
    await bus.drain(1)
    for _ in range(100):
        if not ctrl.busy:
            break
        await asyncio.sleep(0.01)
    assert len(io.played_wavs) == 1
    await ctrl.close()
    await bus.close()


# ---------------------------------------------------------------- 「いまは話せない」


async def test_notice_only_when_someone_looking_at_the_robot_speaks(rig: Rig) -> None:
    rig.clock.advance(5.0)
    rig.publish(ev.PersonEntered(track_id=1))
    await rig.settle()
    rig.io.played_wavs.clear()
    rig.clock.advance(20.0)
    # 見ていない人の発話（人同士の会話かもしれない）には黙る
    rig.publish(ev.SpeechEnded(utt_id="u1", duration_s=2.0))
    await rig.settle()
    assert rig.io.played_wavs == []
    # 見ている人が話しかけたら、会話できないことを伝える
    rig.publish(ev.EyeContactChanged(track_id=1, active=True))
    await rig.bus.drain(1)
    rig.publish(ev.SpeechEnded(utt_id="u2", duration_s=0.2))  # 短い物音は無視
    await rig.settle()
    assert rig.io.played_wavs == []
    rig.publish(ev.SpeechEnded(utt_id="u3", duration_s=2.0))
    await rig.settle()
    assert rig.io.played_wavs == [b"RIFF-notice"]
    # 続けて話しかけられても、間隔内は繰り返さない
    rig.clock.advance(5.0)
    rig.publish(ev.SpeechEnded(utt_id="u4", duration_s=2.0))
    await rig.settle()
    assert len(rig.io.played_wavs) == 1
    rig.clock.advance(30.0)
    rig.publish(ev.SpeechEnded(utt_id="u5", duration_s=2.0))
    await rig.settle()
    assert len(rig.io.played_wavs) == 2


async def test_missing_clips_degrade_to_a_nod_or_silence(tmp_path: Path) -> None:
    cache = await make_cache(tmp_path, {})  # キャッシュが空（まだ一度もサーバーに繋がっていない）
    cache.load()
    clock = ManualClock(10.0)
    bus = EventBus(clock)
    io = RecordingIO()
    world = World()
    world.attach(bus)
    ctrl = FallbackController(io, bus, world, cache, clock=clock, hour=lambda: 14)
    ctrl.attach()
    ctrl._down_since = 0.0  # pyright: ignore[reportPrivateUsage]
    bus.publish(ev.PersonEntered(track_id=1))
    bus.publish(ev.EyeContactChanged(track_id=1, active=True))
    await bus.drain(1)
    for _ in range(100):
        if not ctrl.busy:
            break
        await asyncio.sleep(0.01)
    assert io.played_wavs == []
    assert [c.head for c in io.commands if isinstance(c, GotoPose)] == [
        HeadPose(pitch=15.0),
        HeadPose(),
    ]  # 声は出せないが、うなずきで気づいたことは示す
    bus.publish(ev.SpeechEnded(utt_id="u", duration_s=2.0))
    await bus.drain(1)
    await asyncio.sleep(0.05)
    assert io.played_wavs == []
    await ctrl.close()
    await bus.close()


# ---------------------------------------------------------------- 障害注入（サーバーを落として、戻す）


TOKEN = "test-token-0123456789"


class Net:
    """サーバーに繋がる／繋がらないを切り替えられる接続。"""

    def __init__(self) -> None:
        self.up = True
        self.server = FakeServer(FakeServerConfig(token=TOKEN))
        self.tasks: list[asyncio.Task[None]] = []

    async def connect(self) -> Transport:
        if not self.up:
            raise ConnectionRefusedError
        robot_side, server_side = memory_pair()
        self.tasks.append(asyncio.create_task(self.server.serve(server_side)))
        return robot_side

    def crash(self) -> None:
        self.up = False
        for t in self.tasks:
            t.cancel()


async def until(cond: Callable[[], bool], timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not cond():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def test_fault_injection_server_down_and_back(tmp_path: Path) -> None:
    cache = await make_cache(tmp_path)
    cache.load()
    clock = ManualClock(100.0)
    bus = EventBus(clock)
    io = RecordingIO()
    world = World()
    world.attach(bus)
    ctrl = FallbackController(
        io,
        bus,
        world,
        cache,
        clock=clock,
        config=FallbackConfig(greeting_min_gap_s=0.0),
        hour=lambda: 14,
    )
    ctrl.attach()
    net = Net()
    client = ServerClient(
        net.connect,
        bus,
        LinkConfig(
            robot_id="r",
            token=SecretStr(TOKEN),
            backoff_initial_s=0.02,
            backoff_max_s=0.05,
            ping_interval_s=10,
        ),
        rng=random.Random(0),
    )
    client.start()
    await asyncio.wait_for(client.ready.wait(), 2)
    await bus.drain(1)

    async def enter(track: int) -> None:
        bus.publish(ev.PersonEntered(track_id=track))
        await bus.drain(1)
        for _ in range(200):
            if not ctrl.busy:
                break
            await asyncio.sleep(0.005)

    # 1. 繋がっている間は、フォールバックは何もしない
    clock.advance(10.0)
    await enter(1)
    assert io.played_wavs == [] and not ctrl.active

    # 2. サーバーが落ちる。猶予の間は何もしない
    net.crash()
    await until(lambda: not world.snapshot().robot.server_connected)
    await enter(2)
    assert io.played_wavs == []

    # 3. 猶予をすぎても落ちたままなら、入室した人にキャッシュの挨拶とうなずきで反応する
    clock.advance(4.0)
    assert ctrl.active
    await enter(3)
    assert io.played_wavs == [b"RIFF-hello"]
    assert any(isinstance(c, GotoPose) for c in io.commands)

    # 4. サーバーが戻ると、通常に戻る（フォールバックは何もしない）
    net.up = True
    await until(lambda: world.snapshot().robot.server_connected)
    assert not ctrl.active
    io.played_wavs.clear()
    await enter(4)
    assert io.played_wavs == []

    await client.close()
    await ctrl.close()
    await bus.close()


# ---------------------------------------------------------------- 失敗・取消のあと始末（Codex の指摘 #132）


class FlakyIO(RecordingIO):
    def __init__(self) -> None:
        super().__init__()
        self.fail_mic_off = False
        self.fail_nod = False

    async def set_mic_enabled(self, enabled: bool) -> None:
        self.commands.append(MicEnabled(enabled))
        if self.fail_mic_off and not enabled:
            raise OSError("usb")
        self.mic_enabled = enabled

    async def goto_pose(
        self, head: HeadPose, antennas: tuple[float, float] = (0.0, 0.0), duration_s: float = 0.3
    ) -> None:
        if self.fail_nod:
            raise OSError("servo")
        await super().goto_pose(head, antennas, duration_s)


async def flaky_rig(
    tmp_path: Path,
) -> tuple[FlakyIO, EventBus, World, FallbackController, list[Event], ManualClock]:
    cache = await make_cache(tmp_path)
    cache.load()
    clock = ManualClock(10.0)
    bus = EventBus(clock)
    io = FlakyIO()
    world = World()
    world.attach(bus)
    ctrl = FallbackController(
        io,
        bus,
        world,
        cache,
        clock=clock,
        config=FallbackConfig(greeting_min_gap_s=0.0),
        hour=lambda: 14,
    )
    ctrl.attach()
    ctrl._down_since = 0.0  # pyright: ignore[reportPrivateUsage]
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    return io, bus, world, ctrl, events, clock


async def wait_idle(ctrl: FallbackController) -> None:
    for _ in range(300):
        if not ctrl.busy:
            return
        await asyncio.sleep(0.005)


async def test_a_mic_failure_still_balances_the_speech_events(tmp_path: Path) -> None:
    io, bus, world, ctrl, events, _ = await flaky_rig(tmp_path)
    io.fail_mic_off = True
    bus.publish(ev.PersonEntered(track_id=1))
    await bus.drain(1)
    await wait_idle(ctrl)
    await bus.drain(1)
    started = sum(isinstance(e, ev.RobotSpeechStarted) for e in events)
    ended = sum(isinstance(e, ev.RobotSpeechEnded) for e in events)
    assert started == ended == 1  # 開始と終了がそろう
    assert world.snapshot().robot.speaking is False  # 「話し中」のまま固まらない
    assert [c.enabled for c in io.commands if isinstance(c, MicEnabled)][-1] is True  # マイクを戻す
    io.fail_mic_off = False
    bus.publish(ev.PersonEntered(track_id=2))  # 次の反応も受け付ける
    await bus.drain(1)
    await wait_idle(ctrl)
    assert len(io.played_wavs) == 1
    await ctrl.close()
    await bus.close()


async def test_a_failed_nod_does_not_leave_playback_running_outside_the_controller(
    tmp_path: Path,
) -> None:
    io, bus, _, ctrl, events, _ = await flaky_rig(tmp_path)
    io.fail_nod = True
    io.play_delay_s = 0.1
    bus.publish(ev.PersonEntered(track_id=1))
    await bus.drain(1)
    await wait_idle(ctrl)
    # 反応が終わった（busy でなくなった）時点で、再生も終わってあと始末されている
    await bus.drain(1)
    assert [c.enabled for c in io.commands if isinstance(c, MicEnabled)] == [False, True]
    assert sum(isinstance(e, ev.RobotSpeechEnded) for e in events) == 1
    await ctrl.close()
    await bus.close()


async def test_close_during_playback_cancels_and_cleans_up_both_jobs(tmp_path: Path) -> None:
    io, bus, world, ctrl, events, _ = await flaky_rig(tmp_path)
    io.play_delay_s = 5.0
    bus.publish(ev.PersonEntered(track_id=1))
    await bus.drain(1)
    for _ in range(100):
        if io.played_wavs:
            break
        await asyncio.sleep(0.005)
    await ctrl.close()  # 再生の最中に止める
    await bus.drain(1)
    assert [c.enabled for c in io.commands if isinstance(c, MicEnabled)] == [False, True]
    assert sum(isinstance(e, ev.RobotSpeechEnded) for e in events) == 1
    assert world.snapshot().robot.speaking is False
    await bus.close()


async def test_cancelling_stops_the_speaker_before_the_mic_is_restored(tmp_path: Path) -> None:
    # 実機の play_wav は「再生を始めて、長さの分だけ待つ」だけなので、待ちを取り消しても音は止まらない。
    # 先に再生を止めてからマイクを戻さないと、自分の声を拾い、「話し終えた」と言いながら鳴り続ける
    io, bus, _, ctrl, events, _ = await flaky_rig(tmp_path)
    io.play_delay_s = 5.0
    bus.publish(ev.PersonEntered(track_id=1))
    await bus.drain(1)
    for _ in range(100):
        if io.played_wavs:
            break
        await asyncio.sleep(0.005)
    await ctrl.close()
    kinds = [
        type(c).__name__ + (str(c.enabled) if isinstance(c, MicEnabled) else "")
        for c in io.commands
    ]
    mic_off, stop, mic_on = (
        kinds.index("MicEnabledFalse"),
        kinds.index("StopPlayback"),
        kinds.index("MicEnabledTrue"),
    )
    assert mic_off < stop < mic_on
    await bus.drain(1)
    assert sum(isinstance(e, ev.RobotSpeechEnded) for e in events) == 1
    await bus.close()


async def test_a_normal_playback_does_not_call_stop(tmp_path: Path) -> None:
    io, bus, _, ctrl, _, _ = await flaky_rig(tmp_path)
    bus.publish(ev.PersonEntered(track_id=1))
    await bus.drain(1)
    await wait_idle(ctrl)
    assert not [c for c in io.commands if isinstance(c, StopPlayback)]
    await ctrl.close()
    await bus.close()
