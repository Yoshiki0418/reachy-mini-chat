"""Motion Executor (ロボット側)。

PC のエージェントが express_motion ツールで送ってくるプリセット名を、
reachy_mini の goto_target(head / antennas / body_yaw) 系列に変換して再生する。

- 各モーションは短い系列で、最後に neutral へ戻す。
- 再生は専用ワーカースレッドで直列化（音声再生とは別サブシステムなので並行してよい）。
- head の回転は degrees、antennas / body_yaw は radians（SDK ドキュメントの慣例に合わせる）。
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

_ANT_NEUTRAL = np.deg2rad([0.0, 0.0])


def _pose(roll: float = 0, pitch: float = 0, yaw: float = 0, z: float = 0):
    """degrees 指定で頭部ポーズを作る（z は mm）。"""
    return create_head_pose(roll=roll, pitch=pitch, yaw=yaw, z=z, mm=True, degrees=True)


class MotionExecutor:
    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self._q: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def play(self, name: str) -> None:
        """モーションをキューに積む（非ブロッキング）。"""
        self._q.put(name)

    def _worker(self) -> None:
        while True:
            name = self._q.get()
            try:
                fn = getattr(self, f"_m_{name}", None)
                if fn is None:
                    print(f"[motion] 未知のプリセット: {name}", flush=True)
                    continue
                fn()
                self._neutral()
            except Exception as e:  # noqa: BLE001
                print(f"[motion] {name} 再生エラー: {e}", flush=True)
            finally:
                self._q.task_done()

    def _neutral(self) -> None:
        self.mini.goto_target(
            head=_pose(), antennas=_ANT_NEUTRAL, body_yaw=0.0, duration=0.35
        )

    # ---- プリセット ----
    def _m_nod(self) -> None:  # うなずく
        for _ in range(2):
            self.mini.goto_target(head=_pose(pitch=15), duration=0.22)
            self.mini.goto_target(head=_pose(pitch=0), duration=0.22)

    def _m_shake(self) -> None:  # 首を横に振る
        for yaw in (20, -20, 15, -15):
            self.mini.goto_target(head=_pose(yaw=yaw), duration=0.18)

    def _m_tilt(self) -> None:  # 首をかしげる
        self.mini.goto_target(head=_pose(roll=22), duration=0.4)
        time.sleep(0.5)

    def _m_look_away(self) -> None:  # 照れて顔を逸らす
        self.mini.goto_target(
            head=_pose(yaw=35, pitch=10), antennas=np.deg2rad([-20, -20]), duration=0.5
        )
        time.sleep(0.9)

    def _m_happy(self) -> None:  # 嬉しそうにアンテナを動かす
        self.mini.goto_target(head=_pose(z=8), duration=0.2)
        for a in ([30, -30], [-30, 30], [30, -30], [0, 0]):
            self.mini.goto_target(antennas=np.deg2rad(a), duration=0.16, method="cartoon")

    def _m_greet(self) -> None:  # 挨拶
        self.mini.goto_target(
            head=_pose(pitch=12), antennas=np.deg2rad([40, 40]), duration=0.3
        )
        self.mini.goto_target(head=_pose(pitch=0), duration=0.3)

    def _m_think(self) -> None:  # 考えるしぐさ
        self.mini.goto_target(
            head=_pose(pitch=-10, roll=12), antennas=np.deg2rad([15, -15]), duration=0.5
        )
        time.sleep(0.8)
