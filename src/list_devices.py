"""利用可能なオーディオデバイスを一覧表示する。

Reachy Mini が接続されているか、どのデバイス ID で見えているかの確認用。
"""

from __future__ import annotations

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    default_in, default_out = sd.default.device

    print(f"デフォルト入力: index={default_in}")
    print(f"デフォルト出力: index={default_out}")
    print()
    print(f"{'idx':>3}  {'in':>3} {'out':>3}  name")
    print("-" * 70)
    for i, d in enumerate(devices):
        in_ch = d["max_input_channels"]
        out_ch = d["max_output_channels"]
        name = d["name"]
        marker = ""
        if "reachy" in name.lower():
            marker = "  <- Reachy"
        print(f"{i:>3}  {in_ch:>3} {out_ch:>3}  {name}{marker}")


if __name__ == "__main__":
    main()
