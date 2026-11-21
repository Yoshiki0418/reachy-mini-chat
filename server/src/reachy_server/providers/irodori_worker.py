"""Irodori-TTS を動かすワーカープロセス。Irodori-TTS 自身の Python 環境で起動する。

    ~/Irodori-TTS/.venv/bin/python irodori_worker.py --irodori-dir ~/Irodori-TTS --ref-wav voice.wav

Irodori-TTS は torch・torchcodec・transformers の版がサーバー本体の lock と合わないので、
別プロセスにして標準入出力でつなぐ（設計書 §4.3）。このファイルは reachy_server を import しない
（標準ライブラリと Irodori-TTS だけで動く）。

プロトコル（改行区切りの JSON。応答の本体は続けて ``nbytes`` バイト）:
  起動完了  worker → server  {"ready": true, "sr": 24000}
  合成      server → worker  {"id": 1, "text": "..."}
  取消      server → worker  {"cancel": 1}      # まだ始めていなければ捨てる
  応答      worker → server  {"id": 1, "ok": true, "nbytes": N, "sr": 24000, "synth_s": 0.2} + WAV
                             {"id": 1, "ok": false, "error": "..."}
標準出力はプロトコル専用にし、ライブラリの print は標準エラーへ回す。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import wave
from typing import Any

CANCELLED = "cancelled"


def _log(msg: str) -> None:
    print(f"[irodori-worker] {msg}", file=sys.stderr, flush=True)


class _Channel:
    """標準出力の本物をプロトコル用に確保し、fd 1 は標準エラーへ向ける。"""

    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(1), "wb")
        os.dup2(2, 1)
        self._lock = threading.Lock()

    def send(self, header: dict[str, Any], body: bytes = b"") -> None:
        with self._lock:
            self._out.write(json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n")
            if body:
                self._out.write(body)
            self._out.flush()


def to_wav_bytes(audio: Any, sr: int, out_sr: int) -> bytes:
    """float のテンソル（チャンネル × サンプル）を、モノラル PCM16 の WAV にする。"""
    import torch  # pyright: ignore[reportMissingImports]
    import torchaudio  # pyright: ignore[reportMissingImports]

    wav = audio.detach().to(device="cpu", dtype=torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != out_sr:
        wav = torchaudio.functional.resample(wav, sr, out_sr)
    pcm = (wav.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).numpy().tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(out_sr)
        w.writeframes(pcm)
    return buf.getvalue()


class Engine:
    def __init__(self, args: argparse.Namespace) -> None:
        sys.path.insert(0, os.path.expanduser(args.irodori_dir))
        from irodori_tts.inference_runtime import (  # pyright: ignore[reportMissingImports]
            InferenceRuntime,
            RuntimeKey,
            SamplingRequest,
            download_hf_checkpoint,
            resolve_cfg_scales,
        )

        self._request_cls = SamplingRequest
        self._resolve_cfg_scales = resolve_cfg_scales
        self._ref_wav = os.path.abspath(args.ref_wav) if args.ref_wav else None
        self._caption = args.caption or None
        self._out_sr = args.sample_rate
        self._num_steps = args.num_steps
        self._seed = args.seed
        self.runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=download_hf_checkpoint(args.checkpoint),
                model_device="cuda",
                model_precision=args.precision,
                codec_device="cuda",
                codec_precision="fp32",
            )
        )

    def synthesize(self, text: str) -> tuple[bytes, float]:
        use_caption = bool(self._caption) and bool(self.runtime.model_cfg.use_caption_condition)
        use_ref = self._ref_wav is not None
        cfg_text, cfg_caption, cfg_speaker, _ = self._resolve_cfg_scales(
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_caption_condition=use_caption,
            use_speaker_condition=use_ref,
        )
        req = self._request_cls(
            text=text,
            caption=self._caption if use_caption else None,
            ref_wav=self._ref_wav,
            no_ref=not use_ref,
            num_steps=self._num_steps,
            seed=self._seed,
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
        )
        t0 = time.perf_counter()
        result = self.runtime.synthesize(req)
        synth_s = time.perf_counter() - t0
        return to_wav_bytes(result.audio, result.sample_rate, self._out_sr), synth_s


def _stdin_reader(requests: queue.Queue[dict[str, Any] | None], cancelled: set[int]) -> None:
    """標準入力を読み続け、取消はその場で記録する（合成中でも受け取れるように別スレッド）。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if "cancel" in msg:
            cancelled.add(int(msg["cancel"]))
        else:
            requests.put(msg)
    requests.put(None)  # 親が閉じた


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--irodori-dir", default="~/Irodori-TTS")
    p.add_argument("--checkpoint", default="Aratako/Irodori-TTS-v4.1-Small-MF")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--ref-wav", default="", help="話者の参照音声。声を固定する")
    p.add_argument("--caption", default="", help="声と話し方の説明（参照音声と併用できる）")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="固定すると毎回同じ揺らぎで合成する")
    p.add_argument("--sample-rate", type=int, default=24000, help="出力の WAV のサンプルレート")
    args = p.parse_args()

    # bf16 + cuDNN の注意機構は、初めての長さの文で 0.6〜1.5 秒止まる（docs/benchmarks.md）
    os.environ.setdefault("IRODORI_SDPA_CUDNN", "0")
    channel = _Channel()
    engine = Engine(args)
    engine.synthesize("おはよう。")  # 参照音声の符号化とカーネルの準備を済ませておく
    channel.send({"ready": True, "sr": args.sample_rate})
    _log("ready")

    requests: queue.Queue[dict[str, Any] | None] = queue.Queue()
    cancelled: set[int] = set()
    threading.Thread(target=_stdin_reader, args=(requests, cancelled), daemon=True).start()
    while (msg := requests.get()) is not None:
        rid = int(msg["id"])
        if rid in cancelled:
            cancelled.discard(rid)
            channel.send({"id": rid, "ok": False, "error": CANCELLED})
            continue
        try:
            wav, synth_s = engine.synthesize(str(msg["text"]))
        except Exception as e:  # 1 文の失敗でワーカーを落とさない
            _log(f"synthesis failed: {e!r}")
            channel.send({"id": rid, "ok": False, "error": repr(e)})
            continue
        channel.send(
            {"id": rid, "ok": True, "nbytes": len(wav), "sr": args.sample_rate, "synth_s": synth_s},
            wav,
        )


if __name__ == "__main__":
    main()
