"""マイクの診断スクリプト。

- 公式ドキュメント通りの呼び出し順で start_recording() / get_audio_sample()
- 各チャンクの形状とpeakを表示
- DoA (Direction of Arrival) でも音声検出を試みる
"""

from __future__ import annotations

import time

import numpy as np
from reachy_mini import ReachyMini


def main() -> None:
    with ReachyMini() as mini:
        # 公式ドキュメント通りの順序
        mini.media.start_recording()
        mini.media.start_playing()

        in_sr = mini.media.get_input_audio_samplerate()
        in_ch = mini.media.get_input_channels()
        print(f"mic: sr={in_sr}, ch={in_ch}")

        # 5秒間、ループでget_audio_sampleを叩いて中身をログ
        print("5秒間、マイクから取得を試行 (話してください)…")
        t_start = time.monotonic()
        chunks_info: list[str] = []
        none_count = 0
        empty_count = 0
        nonzero_count = 0
        total_samples = 0
        max_peak = 0.0

        while time.monotonic() - t_start < 5.0:
            s = mini.media.get_audio_sample()
            if s is None:
                none_count += 1
            elif len(s) == 0:
                empty_count += 1
            else:
                total_samples += len(s)
                peak = float(np.abs(s).max())
                max_peak = max(max_peak, peak)
                if peak > 1e-6:
                    nonzero_count += 1
                if len(chunks_info) < 10:
                    chunks_info.append(f"  shape={s.shape} dtype={s.dtype} peak={peak:.6f}")
            time.sleep(0.02)

            # DoAも並行で確認
            doa_result = mini.media.get_DoA()
            if doa_result is not None and len(chunks_info) < 10:
                chunks_info.append(f"  DoA: angle={doa_result[0]:.2f}rad is_speech={doa_result[1]}")

        print("\n結果:")
        print(f"  None 返却回数  : {none_count}")
        print(f"  空配列 返却回数: {empty_count}")
        print(f"  非ゼロチャンク : {nonzero_count}")
        print(f"  合計サンプル   : {total_samples} (期待 ~{int(5 * in_sr)})")
        print(f"  最大peak       : {max_peak:.6f}")
        print("\n最初の10件のチャンク情報:")
        for c in chunks_info:
            print(c)

        mini.media.stop_recording()
        mini.media.stop_playing()


if __name__ == "__main__":
    main()
