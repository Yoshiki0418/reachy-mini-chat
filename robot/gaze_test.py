"""視線追従 (Gaze) 単体テスト。

PC サーバや STT/TTS を介さず、MotionController + GazeTracker だけを起動して
一定時間動かし、GAZE_DEBUG ログで顔追従が働いているかを確認する診断スクリプト。

実行:
    GAZE_DEBUG=1 /venvs/apps_venv/bin/python robot/gaze_test.py [秒数]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("GAZE_DEBUG", "1")

from reachy_mini import ReachyMini  # noqa: E402

from motion import MotionController  # noqa: E402
from gaze import GazeTracker  # noqa: E402


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    with ReachyMini() as mini:
        motion = MotionController(mini)
        gaze = GazeTracker(mini, motion)
        print(f"[gaze_test] {seconds:.0f}s 追従を開始。顔を見せてください。", flush=True)
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass
        finally:
            gaze.stop()
            motion.stop()
        print("[gaze_test] 終了。", flush=True)


if __name__ == "__main__":
    main()
