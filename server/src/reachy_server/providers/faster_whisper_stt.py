"""faster-whisper（GPU）の STT プロバイダ。設計書 §4.1・docs/benchmarks.md。

既定は kotoba-whisper-v2.0（beam 1）。#14 の計測で、1 発話 p50 0.11 s / p90 0.14 s、VRAM 2.4 GB。
重い依存（faster-whisper・numpy）は初回の ``load()`` で読み込むので、CI や Fake 構成では入れない。
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib.util
import logging
import math
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from .base import Transcription

log = logging.getLogger(__name__)

WHISPER_SR = 16000


class _Segment(Protocol):
    text: str
    start: float
    end: float
    avg_logprob: float


def default_compute_type(device: str) -> str:
    """CTranslate2 の CPU 実装は float16 に対応していないので、CPU では int8 にする。"""
    return "float16" if device == "cuda" else "int8"


def confidence_from_segments(segments: Iterable[_Segment]) -> float | None:
    """区間ごとの平均対数確率を、長さで重み付けして平均し、確率（0〜1）にする。

    Whisper の ``avg_logprob`` は 1 トークンあたりの対数確率なので、exp で戻すと
    「1 トークンあたりの確からしさ」になる。区間がなければ None。
    """
    total = 0.0
    weighted = 0.0
    for s in segments:
        dur = max(s.end - s.start, 1e-3)
        total += dur
        weighted += dur * s.avg_logprob
    if total == 0.0:
        return None
    return min(1.0, max(0.0, math.exp(weighted / total)))


def _preload_cublas12() -> None:
    """ctranslate2 の配布 wheel は CUDA 12 の cuBLAS を実行時に探す。

    この環境の torch は CUDA 13 版なので、pip の ``nvidia-cublas-cu12`` を先に読み込んで渡す。
    入っていなければ何もしない（システムに CUDA 12 があれば、そちらが使われる）。
    """
    spec = importlib.util.find_spec("nvidia.cublas")
    if spec is None or not spec.submodule_search_locations:
        return
    libdir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        path = libdir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _numpy_audio(pcm_f32le: bytes, sr: int) -> Any:
    """float32 little-endian mono を、16 kHz の numpy 配列にする（必要なら線形補間で変換）。"""
    import numpy as np

    audio = np.frombuffer(pcm_f32le, dtype="<f4")
    if sr != WHISPER_SR and audio.size:
        n_out = round(audio.size * WHISPER_SR / sr)
        audio = np.interp(
            np.linspace(0, audio.size - 1, n_out), np.arange(audio.size), audio
        ).astype(np.float32)
    return audio.astype(np.float32, copy=False)


class FasterWhisperSTT:
    """faster-whisper による文字起こし。GPU は 1 つのモデルを共有し、推論は 1 つずつ行う。"""

    def __init__(
        self,
        *,
        model: str = "kotoba-tech/kotoba-whisper-v2.0-faster",
        device: str = "cuda",
        compute_type: str | None = None,
        beam_size: int = 1,
        language: str = "ja",
        hotwords: str | None = None,
        model_factory: Callable[[], Any] | None = None,
        to_audio: Callable[[bytes, int], Any] = _numpy_audio,
    ) -> None:
        self.name = f"faster-whisper:{model.rsplit('/', 1)[-1]}"
        self._model_name = model
        self._device = device
        self._compute_type = compute_type or default_compute_type(device)
        self._beam_size = beam_size
        self._language = language
        self._hotwords = hotwords
        self._model_factory = model_factory
        self._to_audio = to_audio
        self._model: Any = None
        # CTranslate2 のモデルは同時に呼べるが、GPU を取り合うと両方遅くなるので 1 つずつにする
        self._lock = asyncio.Lock()

    def load(self) -> None:
        """モデルを読み込む（起動時に呼ぶと、最初の発話が遅れない）。"""
        if self._model is not None:
            return
        t0 = time.perf_counter()
        if self._model_factory is not None:
            self._model = self._model_factory()
        else:
            _preload_cublas12()
            from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]  # 使うときだけ

            self._model = WhisperModel(
                self._model_name, device=self._device, compute_type=self._compute_type
            )
        log.info("STT モデルを読み込んだ: %s (%.1f s)", self._model_name, time.perf_counter() - t0)

    async def transcribe(self, pcm_f32le: bytes, sr: int) -> Transcription:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, pcm_f32le, sr)

    def _transcribe_sync(self, pcm_f32le: bytes, sr: int) -> Transcription:
        self.load()
        audio = self._to_audio(pcm_f32le, sr)
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=False,  # 発話区間はロボット側の VAD で切って送ってくる
            condition_on_previous_text=False,
            hotwords=self._hotwords,
        )
        segs = list(segments)  # ジェネレータなので、ここで推論が走る
        text = "".join(s.text for s in segs).strip()
        return Transcription(
            text=text,
            confidence=confidence_from_segments(segs),
            lang=getattr(info, "language", self._language) or self._language,
        )
