"""マイクから録音 → Whisper (ローカル) で文字起こし。

例:
    uv run python src/record_and_transcribe.py --seconds 5
    uv run python src/record_and_transcribe.py --device 3 --model small
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_devices import find_device

SAMPLE_RATE = 16000  # Whisper は 16kHz 想定
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"


def record(seconds: float, device: int | None) -> np.ndarray:
    info = sd.query_devices(device, "input") if device is not None else sd.query_devices(kind="input")
    print(f"録音デバイス: {info['name']} (sr={SAMPLE_RATE})")
    print(f"{seconds}秒間 録音します…")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("録音完了。")
    return audio.flatten()


def save_wav(audio: np.ndarray) -> Path:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RECORDINGS_DIR / f"rec_{ts}.wav"
    sf.write(path, audio, SAMPLE_RATE)
    return path


def transcribe(audio: np.ndarray, model_name: str, language: str | None) -> str:
    import whisper  # 遅延 import (起動が重いため)

    print(f"Whisper モデル '{model_name}' をロード中…")
    model = whisper.load_model(model_name)
    print("文字起こし中…")
    result = model.transcribe(audio, language=language, fp16=False)
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0, help="録音秒数")
    parser.add_argument("--device", type=int, default=None, help="入力デバイス index (省略時は Reachy/デフォルト)")
    parser.add_argument("--model", default="base", help="whisper モデル名 (tiny/base/small/medium/large)")
    parser.add_argument("--language", default="ja", help="言語コード (例: ja, en)。auto なら未指定。")
    args = parser.parse_args()

    device = args.device if args.device is not None else find_device("input")
    language = None if args.language.lower() == "auto" else args.language

    audio = record(args.seconds, device)
    wav_path = save_wav(audio)
    print(f"保存: {wav_path}")

    text = transcribe(audio, args.model, language)
    print()
    print("=== 文字起こし結果 ===")
    print(text)


if __name__ == "__main__":
    main()
