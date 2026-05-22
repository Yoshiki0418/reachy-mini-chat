"""Reachy Mini を優先しつつ入出力デバイスを解決するユーティリティ。"""

from __future__ import annotations

import sounddevice as sd


def find_device(kind: str, name_hint: str | None = None) -> int | None:
    """kind='input' or 'output' のデバイスを探す。

    name_hint が指定されればその文字列を含むデバイスを優先。
    見つからなければ Reachy を探し、それも無ければ None（デフォルト使用）。
    """
    assert kind in ("input", "output")
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()

    hints: list[str] = []
    if name_hint:
        hints.append(name_hint.lower())
    hints.append("reachy")

    for hint in hints:
        for i, d in enumerate(devices):
            if d[channel_key] > 0 and hint in d["name"].lower():
                return i
    return None
