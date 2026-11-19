"""faster-whisper の各設定でベンチマークし、最適な構成を探す。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from faster_whisper import WhisperModel


def bench(model_size: str, compute_type: str, cpu_threads: int, vad_filter: bool, wav: Path, language: str) -> None:
    print(f"\n--- model={model_size} compute={compute_type} threads={cpu_threads} vad={vad_filter} ---")
    t = time.monotonic()
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
    print(f"  load: {time.monotonic() - t:.1f}s")

    for trial in range(2):
        t = time.monotonic()
        segments, info = model.transcribe(str(wav), language=language, vad_filter=vad_filter, beam_size=1)
        text = "".join(seg.text for seg in segments).strip()
        dur = time.monotonic() - t
        print(f"  trial{trial+1}: {dur:.1f}s  text='{text[:60]}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", default=None)
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    if args.wav:
        wav = Path(args.wav)
    else:
        wav = sorted(recordings_dir.glob("mic_*.wav"))[-1]
    print(f"WAV: {wav}")

    configs = [
        ("tiny", "int8", 4, False),
        ("tiny", "int8", 4, True),
        ("tiny", "int8_float32", 4, True),
        ("base", "int8", 4, True),
    ]
    for model_size, ct, threads, vad in configs:
        try:
            bench(model_size, ct, threads, vad, wav, args.language)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
