"""テキストを macOS `say` で音声化して、指定の出力デバイス(=Reachy Mini)で再生する。

`say -o` で AIFF を書き出し、sounddevice 経由で任意の出力デバイスに流す。
こうしないと `say` は常にシステムのデフォルト出力に流れてしまうため。

例:
    uv run python src/speak.py "こんにちは、レイチーです"
    uv run python src/speak.py "Hello" --voice Samantha
    uv run python src/speak.py "テスト" --device 4
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from audio_devices import find_device

DEFAULT_VOICE_JA = "Kyoko"


def synthesize(text: str, voice: str, rate: int | None) -> Path:
    tmp = Path(tempfile.mkstemp(suffix=".aiff")[1])
    cmd = ["say", "-v", voice, "-o", str(tmp)]
    if rate is not None:
        cmd += ["-r", str(rate)]
    cmd += ["--", text]
    subprocess.run(cmd, check=True)
    return tmp


def play(path: Path, device: int | None) -> None:
    data, samplerate = sf.read(path, dtype="float32")
    info = sd.query_devices(device, "output") if device is not None else sd.query_devices(kind="output")
    print(f"再生デバイス: {info['name']} (sr={samplerate})")
    sd.play(data, samplerate=samplerate, device=device)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="読み上げるテキスト")
    parser.add_argument("--voice", default=DEFAULT_VOICE_JA, help=f"say の声 (デフォルト: {DEFAULT_VOICE_JA})")
    parser.add_argument("--rate", type=int, default=None, help="話速 (words/min)。未指定なら声のデフォルト。")
    parser.add_argument("--device", type=int, default=None, help="出力デバイス index (省略時は Reachy/デフォルト)")
    args = parser.parse_args()

    device = args.device if args.device is not None else find_device("output")

    aiff = synthesize(args.text, args.voice, args.rate)
    try:
        play(aiff, device)
    finally:
        aiff.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
