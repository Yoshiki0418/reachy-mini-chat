"""faster-whisper で既存の録音WAVを文字起こしして表示するテスト。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?", default=None, help="WAVファイル (省略時は最新の録音)")
    parser.add_argument("--model", default="tiny", help="tiny/base/small/medium/large-v3 など")
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()

    recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    if args.wav:
        wav = Path(args.wav)
    else:
        wavs = sorted(recordings_dir.glob("mic_*.wav"))
        if not wavs:
            print("録音WAVが見つかりません", file=sys.stderr)
            sys.exit(1)
        wav = wavs[-1]

    print(f"WAV: {wav}")
    print(f"モデル: {args.model}  言語: {args.language}")

    t = time.monotonic()
    print("モデルをロード中…")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(f"  → {time.monotonic() - t:.1f}s")

    t = time.monotonic()
    print("文字起こし中…")
    segments, _info = model.transcribe(str(wav), language=args.language, vad_filter=False)
    text = "".join(seg.text for seg in segments).strip()
    print(f"  → {time.monotonic() - t:.1f}s")

    print()
    print("=== 結果 ===")
    print(text or "(空)")


if __name__ == "__main__":
    main()
