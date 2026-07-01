"""スピーカー動作テスト。440Hzの正弦波を1秒鳴らす。

`/venvs/apps_venv/bin/python robot/test_speaker.py` で実行する想定。
"""

from __future__ import annotations

import time

import numpy as np

from reachy_mini import ReachyMini


def main() -> None:
    with ReachyMini() as mini:
        mini.media.start_playing()

        sr = mini.media.get_output_audio_samplerate()
        ch = mini.media.get_output_channels()
        duration = 1.0
        freq = 440.0

        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        samples = np.tile(wave[:, None], (1, ch))  # (N, ch)

        print(f"440Hz {duration}s を再生 (sr={sr}, ch={ch})")
        mini.media.push_audio_sample(samples)

        # push_audio_sample は non-blocking。再生完了まで待つ
        time.sleep(duration + 0.3)

        mini.media.stop_playing()
        print("完了")


if __name__ == "__main__":
    main()
