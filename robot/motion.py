"""Motion Controller (ロボット側)。

全モーションを 1 スレッドで調停する。2 層構成:
  - ベース層: 待機(idle)/発話(talking) モードに応じた微小な揺れを set_target で常時付与
             （ロボットが「生きている」ように見せる自律モーション）
  - ジェスチャー層: エージェントの express_motion で来たプリセットを goto_target 系列で再生。
                   再生中はベースを止め、終了後 neutral へ戻してベースへ復帰する。

こうすることで、常時微小モーションとトリガー型ジェスチャーが goto_target/set_target を
取り合って競合するのを防ぐ。

head の回転は degrees、antennas / body_yaw は radians（SDK ドキュメントの慣例）。
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

_ANT_NEUTRAL = np.deg2rad([0.0, 0.0])

# ベース微小モーションの更新レート（USB 負荷を抑えつつ十分滑らか）
_BASE_RATE_HZ = 20.0


def _pose(roll: float = 0, pitch: float = 0, yaw: float = 0, z: float = 0):
    """degrees 指定で頭部ポーズを作る（z は mm）。"""
    return create_head_pose(roll=roll, pitch=pitch, yaw=yaw, z=z, mm=True, degrees=True)


def _base_offsets(t: float, mode: str) -> tuple[float, float, float, float, list[float]]:
    """時刻 t（秒）とモードから、neutral まわりの微小オフセットを返す。

    戻り値: (roll_deg, pitch_deg, yaw_deg, z_mm, antennas_deg[2])
    振幅は小さく抑える（待機時は「呼吸」、発話時は少し活発に）。
    """
    if mode == "talking":
        # 話している間は頷き気味に、少し速く動く
        pitch = 3.5 * math.sin(2 * math.pi * 0.9 * t)
        yaw = 3.0 * math.sin(2 * math.pi * 0.35 * t)
        roll = 1.5 * math.sin(2 * math.pi * 0.5 * t)
        z = 3.0 * math.sin(2 * math.pi * 0.7 * t)
        ant = [6.0 * math.sin(2 * math.pi * 0.8 * t), -6.0 * math.sin(2 * math.pi * 0.8 * t)]
    else:  # idle: ゆっくりした呼吸のような揺れ
        pitch = 1.8 * math.sin(2 * math.pi * 0.08 * t)
        yaw = 2.5 * math.sin(2 * math.pi * 0.05 * t)
        roll = 1.0 * math.sin(2 * math.pi * 0.06 * t)
        z = 2.5 * math.sin(2 * math.pi * 0.15 * t)  # 呼吸
        ant = [3.0 * math.sin(2 * math.pi * 0.12 * t), -3.0 * math.sin(2 * math.pi * 0.1 * t)]
    return roll, pitch, yaw, z, ant


class MotionController:
    def __init__(self, mini: ReachyMini) -> None:
        self.mini = mini
        self._mode = "idle"
        self._gesture_q: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- 公開 API ----
    def set_mode(self, mode: str) -> None:
        """"idle" / "talking" を切り替える。"""
        self._mode = mode

    def gesture(self, name: str) -> None:
        """トリガー型ジェスチャーをキューに積む（非ブロッキング）。"""
        self._gesture_q.put(name)

    def stop(self) -> None:
        """スレッドを止めて neutral に戻す。"""
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.mini.goto_target(head=_pose(), antennas=_ANT_NEUTRAL, body_yaw=0.0, duration=0.3)
        except Exception:
            pass

    # ---- 内部ループ ----
    def _run(self) -> None:
        dt = 1.0 / _BASE_RATE_HZ
        while not self._stop.is_set():
            try:
                name = self._gesture_q.get_nowait()
            except queue.Empty:
                name = None

            if name is not None:
                self._play_gesture(name)
                continue

            t = time.monotonic() - self._t0
            roll, pitch, yaw, z, ant = _base_offsets(t, self._mode)
            # 微小なジッタで機械的な繰り返し感を消す
            jitter = 0.3
            roll += random.uniform(-jitter, jitter)
            pitch += random.uniform(-jitter, jitter)
            try:
                self.mini.set_target(
                    head=_pose(roll=roll, pitch=pitch, yaw=yaw, z=z),
                    antennas=np.deg2rad(ant),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[motion] base set_target エラー: {e}", flush=True)
            time.sleep(dt)

    def _play_gesture(self, name: str) -> None:
        fn = getattr(self, f"_m_{name}", None)
        if fn is None:
            print(f"[motion] 未知のプリセット: {name}", flush=True)
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[motion] {name} 再生エラー: {e}", flush=True)
        finally:
            # ベースへ滑らかに復帰
            try:
                self.mini.goto_target(head=_pose(), antennas=_ANT_NEUTRAL, body_yaw=0.0, duration=0.3)
            except Exception:
                pass

    # ---- ジェスチャープリセット ----
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
