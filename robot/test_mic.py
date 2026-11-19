"""マイク動作テスト。N秒間マイクから録音して WAV に保存し、その場で再生する。

`/venvs/apps_venv/bin/python robot/test_mic.py --seconds 5` で実行する想定。
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from reachy_mini import ReachyMini

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--no-playback", action="store_true", help="録音後の再生をスキップ")
    args = parser.parse_args()

    RECORDINGS_DIR.mkdir(exist_ok=True)

    with ReachyMini() as mini:
        mini.media.start_playing()

        in_sr = mini.media.get_input_audio_samplerate()
        in_ch = mini.media.get_input_channels()
        out_sr = mini.media.get_output_audio_samplerate()
        out_ch = mini.media.get_output_channels()

        # 開始ビープ (880Hz, 0.25秒)
        beep_t = np.linspace(0, 0.25, int(out_sr * 0.25), endpoint=False, dtype=np.float32)
        beep = (0.3 * np.sin(2 * np.pi * 880 * beep_t)).astype(np.float32)
        beep_stereo = np.tile(beep[:, None], (1, out_ch))
        print("録音まもなく開始 (ビープ後)…")
        mini.media.push_audio_sample(beep_stereo)
        time.sleep(0.25 + 0.15)  # ビープ再生完了 + マージン

        mini.media.start_recording()
        print(f"🔴 録音中: {args.seconds}秒 (sr={in_sr}, ch={in_ch}) — どうぞ話してください")

        # mini.media.get_audio_sample() は呼び出しごとに利用可能なバッファを返す
        # 必要時間ぶん溜まるまで繰り返し取得して連結する
        target_samples = int(args.seconds * in_sr)
        chunks: list[np.ndarray] = []
        collected = 0
        t_start = time.monotonic()
        timeout = args.seconds + 5.0
        while collected < target_samples:
            sample = mini.media.get_audio_sample()
            if sample is None or len(sample) == 0:
                time.sleep(0.02)
            else:
                chunks.append(sample)
                collected += len(sample)
            if time.monotonic() - t_start > timeout:
                print(f"タイムアウト: 期待 {target_samples} / 取得 {collected}")
                break

        audio = (
            np.concatenate(chunks, axis=0)[:target_samples]
            if chunks
            else np.zeros((0, in_ch), np.float32)
        )
        print(f"取得サンプル数: {len(audio)} (期待 {target_samples})")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = RECORDINGS_DIR / f"mic_{ts}.wav"
        sf.write(wav_path, audio, in_sr)
        print(f"保存: {wav_path}")

        mini.media.stop_recording()

        if not args.no_playback and len(audio) > 0:
            out_sr = mini.media.get_output_audio_samplerate()
            if out_sr != in_sr:
                from scipy.signal import resample

                n_out = int(len(audio) * out_sr / in_sr)
                playback = resample(audio, n_out).astype(np.float32)
            else:
                playback = audio
            print("再生中…")
            mini.media.push_audio_sample(playback)
            time.sleep(len(playback) / out_sr + 0.3)

        mini.media.stop_playing()
        print("完了")


if __name__ == "__main__":
    main()
