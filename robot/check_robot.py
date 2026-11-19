"""実機 Reachy Mini Wireless のメディア環境を確認する。

`/venvs/apps_venv/bin/python robot/check_robot.py` で実行する想定。
"""

from __future__ import annotations

from reachy_mini import ReachyMini


def main() -> None:
    with ReachyMini() as mini:
        print(f"SDK 接続OK (host={mini._host if hasattr(mini, '_host') else 'auto'})")

        in_sr = mini.media.get_input_audio_samplerate()
        out_sr = mini.media.get_output_audio_samplerate()
        in_ch = mini.media.get_input_channels()
        out_ch = mini.media.get_output_channels()

        print(f"入力 (mic) : samplerate={in_sr}Hz, channels={in_ch}")
        print(f"出力 (spk) : samplerate={out_sr}Hz, channels={out_ch}")


if __name__ == "__main__":
    main()
