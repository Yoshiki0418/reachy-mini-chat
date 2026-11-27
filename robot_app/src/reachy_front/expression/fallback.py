"""サーバー断のフォールバック（設計書 §1.2・§3.1）。

サーバーが落ちていても、ロボットは黙って固まらない。知覚・反射・記憶は続け、入室した人には、
キャッシュした短い挨拶音声（:mod:`reachy_front.client.clips`）とうなずきで反応して、会話はできない
ことを示す。サーバーが戻ったら通常の動作に戻る（このコントローラーは何もしなくなる）。

**「空気を読む」を守る**（タイマー式の自発発話はしない）:
- 反応は入室・発話のイベントがあったときだけ。タイマーで話し始めない
- 切れて ``grace_s`` 秒たつまでは縮退とみなさない（起動直後・瞬断で、通常の挨拶と二重にならない）
- 挨拶は 1 回の断につき、同じ人に 1 回。複数人が続けて入っても、間隔を空ける
- 「いまは話せない」は、ロボットを見ている人が話しかけたときだけ（人同士の会話には割り込まない）。間隔も空ける
- 話している最中・うなずき中は、次を重ねない（取りこぼしは捨てる。溜めない）
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..client.clips import CachedClip, ClipCache
from ..core import events as ev
from ..core.bus import EventBus
from ..core.clock import Clock, SystemClock
from ..core.events import Event
from ..io.base import HeadPose, RobotIO
from ..world import World

log = logging.getLogger(__name__)


@dataclass
class FallbackConfig:
    grace_s: float = 3.0
    greeting_min_gap_s: float = 10.0
    notice_min_gap_s: float = 30.0
    min_speech_s: float = 0.5
    """これより短い物音は「話しかけられた」とみなさない。"""


def _local_hour() -> int:
    return time.localtime().tm_hour


class FallbackController:
    def __init__(
        self,
        io: RobotIO,
        bus: EventBus,
        world: World,
        clips: ClipCache,
        *,
        clock: Clock | None = None,
        config: FallbackConfig | None = None,
        hour: Callable[[], int] = _local_hour,
    ) -> None:
        self._io = io
        self._bus = bus
        self._world = world
        self._clips = clips
        self._clock = clock or SystemClock()
        self._cfg = config or FallbackConfig()
        self._hour = hour
        self._down_since: float | None = self._clock.now()  # 起動直後は、まだ繋がっていない
        self._greeted: set[int] = set()
        self._last_greeting: float | None = None
        self._last_notice: float | None = None
        self._busy = False
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def active(self) -> bool:
        """縮退中（切れて ``grace_s`` 秒以上たっている）か。"""
        return (
            self._down_since is not None
            and self._clock.now() - self._down_since >= self._cfg.grace_s
        )

    @property
    def busy(self) -> bool:
        return self._busy

    def attach(self) -> None:
        self._bus.subscribe(Event, self._on_event, name="fallback")

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ---------------------------------------------------------------- イベント

    def _on_event(self, event: Event) -> None:
        match event:
            case ev.ServerConnectionChanged():
                if event.connected:
                    self._down_since = None
                    self._greeted.clear()  # 通常の動作に戻る
                elif self._down_since is None:
                    self._down_since = self._clock.now()
            case ev.PersonLeft() | ev.PersonPassedBy():
                self._greeted.discard(event.track_id)
            case ev.PersonEntered():
                self._on_entered(event.track_id)
            case ev.SpeechEnded():
                self._on_speech_ended(event)
            case _:
                pass

    def _on_entered(self, track_id: int) -> None:
        if not self.active or track_id in self._greeted:
            return
        now = self._clock.now()
        if (
            self._last_greeting is not None
            and now - self._last_greeting < self._cfg.greeting_min_gap_s
        ):
            return
        clip = self._pick_greeting()
        if not self._begin(clip, nod=True):
            return
        self._greeted.add(track_id)
        self._last_greeting = now

    def _on_speech_ended(self, event: ev.SpeechEnded) -> None:
        if not self.active or event.duration_s < self._cfg.min_speech_s:
            return
        snap = self._world.snapshot()
        if not snap.gazing():  # ロボットを見ている人がいなければ、人同士の会話かもしれない。黙る
            return
        now = self._clock.now()
        if self._last_notice is not None and now - self._last_notice < self._cfg.notice_min_gap_s:
            return
        notices = self._clips.of_kind("notice")
        if self._begin(notices[0] if notices else None, nod=False):
            self._last_notice = now

    # ---------------------------------------------------------------- 反応

    def _pick_greeting(self) -> CachedClip | None:
        h = self._hour()
        wanted = (
            "greeting-morning"
            if 5 <= h < 11
            else "greeting-evening"
            if h >= 17
            else "greeting-hello"
        )
        return self._clips.get(wanted) or next(iter(self._clips.of_kind("greeting")), None)

    def _begin(self, clip: CachedClip | None, *, nod: bool) -> bool:
        """反応を始める。ロボットが話している・反応の最中なら始めない（溜めない）。"""
        if self._busy or self._world.snapshot().robot.speaking:
            return False
        if clip is None and not nod:
            return False  # 音声がキャッシュに無ければ、うなずきだけ（nod）か、何もしない
        self._busy = True
        task = asyncio.get_running_loop().create_task(self._react(clip, nod))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _react(self, clip: CachedClip | None, nod: bool) -> None:
        jobs: list[asyncio.Task[None]] = []
        try:
            if nod:
                jobs.append(asyncio.ensure_future(self._nod()))
            if clip is not None:
                jobs.append(asyncio.ensure_future(self._speak(clip)))
            # どちらかが失敗しても、もう片方が終わる（あと始末する）まで待つ。放置しない
            for result in await asyncio.gather(*jobs, return_exceptions=True):
                if isinstance(result, BaseException):
                    log.error("フォールバックの反応に失敗", exc_info=result)
        except asyncio.CancelledError:
            for job in jobs:  # 止めるときは、両方を止めてあと始末が終わるまで待つ
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
            raise
        finally:
            self._busy = False

    async def _speak(self, clip: CachedClip) -> None:
        self._bus.publish(ev.RobotSpeechStarted())
        try:
            await self._io.set_mic_enabled(False)  # 自分の声を拾わない
            await self._io.play_wav(clip.wav)
        finally:
            # マイクを切れなかった・取り消されたときも、マイクを戻し、開始と終了をそろえる
            with contextlib.suppress(Exception):
                await self._io.set_mic_enabled(True)
            self._bus.publish(ev.RobotSpeechEnded())

    async def _nod(self) -> None:
        await self._io.goto_pose(HeadPose(pitch=15.0), duration_s=0.22)
        await self._io.goto_pose(HeadPose(), duration_s=0.22)


__all__ = ["FallbackConfig", "FallbackController"]
