# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "faster-whisper==1.2.1",
#     "av>=11,<19",  # 19 で av.open の metadata_errors が消え、faster-whisper の decode_audio が落ちる
#     "nvidia-cublas-cu12>=12.8,<13",
#     "huggingface-hub>=0.34",
#     "pyarrow>=18",
#     "numpy>=1.26",
# ]
# ///
"""faster-whisper の速度・日本語の精度・VRAM を測る（#14）。

ワークスペースの環境（CI と共有の lock）には入れず、使い捨ての環境で動かす:

    uv run --script scripts/bench/bench_stt.py --out bench_out/stt

評価データは Hugging Face の japanese-asr/* から、各データの先頭 N 発話を使う
（JSUT basic5000 = 朗読、ReazonSpeech = 放送、Common Voice 8 = いろいろなマイクでの朗読）。
GPU は起動時に空いているものを選ぶ（gpu.py）。手で選ぶときは CUDA_VISIBLE_DEVICES を付ける。

TTS の聞き取りやすさ（合成音声を STT で書き起こしたときの CER）は `--manifest` で測る。
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu import VramPeak, gpu_dict, pin_free_gpu, process_vram_mib
from metrics import corpus_cer, normalize_ja, summarize_seconds

SAMPLE_RATE = 16000

# 名前 → (HF のリポジトリ, 先頭のシャード)
DATASETS: dict[str, tuple[str, str]] = {
    "jsut": ("japanese-asr/ja_asr.jsut_basic5000", "data/test-00000-of-00004.parquet"),
    "reazon": ("japanese-asr/ja_asr.reazonspeech_test", "data/test-00000-of-00002.parquet"),
    "cv8": ("japanese-asr/ja_asr.common_voice_8_0", "data/test-00000-of-00001.parquet"),
}

DEFAULT_MODELS = [
    "small",
    "medium",
    # faster-whisper の別名 "large-v3-turbo" が指す mobiuslabsgmbh/ は、ここへ転送されている
    "dropbox-dash/faster-whisper-large-v3-turbo",
    "large-v3",
    "kotoba-tech/kotoba-whisper-v2.0-faster",
]


@dataclass
class Sample:
    id: str
    audio: Any  # float32 の 1 次元配列（16 kHz）
    text: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _preload_cublas12() -> None:
    """ctranslate2 の配布 wheel は CUDA 12 の cuBLAS を実行時に dlopen する。

    この端末の torch は CUDA 13 版なので、pip の nvidia-cublas-cu12 を先に読み込んでおく
    （同じ soname が読み込み済みなら、dlopen はそれを返す）。
    """
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia.cublas")
    if spec is None or not spec.submodule_search_locations:
        return
    libdir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        path = libdir / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def load_dataset_samples(name: str, n: int) -> list[Sample]:
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    from faster_whisper.audio import decode_audio  # pyright: ignore[reportMissingImports]
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

    repo, filename = DATASETS[name]
    path = hf_hub_download(repo, filename, repo_type="dataset")
    batch = next(
        pq.ParquetFile(path).iter_batches(batch_size=n, columns=["audio", "transcription"])
    )
    samples = []
    for i, row in enumerate(batch.to_pylist()):
        audio = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=SAMPLE_RATE)
        samples.append(Sample(f"{name}/{i:04d}", audio, row["transcription"]))
    return samples


def load_manifest_samples(path: Path) -> list[Sample]:
    """`{"id", "wav", "text"}` の JSONL（bench_tts.py が書く）を読む。"""
    from faster_whisper.audio import decode_audio  # pyright: ignore[reportMissingImports]

    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        wav = Path(item["wav"])
        if not wav.is_absolute():
            wav = path.parent / wav
        audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
        samples.append(Sample(item["id"], audio, item["text"]))
    return samples


def transcribe(model: Any, sample: Sample, *, beam_size: int, hotwords: str | None) -> str:
    segments, _info = model.transcribe(
        sample.audio,
        language="ja",
        beam_size=beam_size,
        vad_filter=False,  # 発話区間はロボット側の VAD で切ってから送る
        condition_on_previous_text=False,
        hotwords=hotwords,
    )
    return "".join(s.text for s in segments)  # ジェネレータなので、ここで推論が走る


def bench_model(
    model_name: str,
    compute_type: str,
    beams: list[int],
    data: dict[str, list[Sample]],
    *,
    hotwords: str | None,
    warmup: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from faster_whisper import WhisperModel  # pyright: ignore[reportMissingImports]

    rows: list[dict[str, Any]] = []
    hyps: list[dict[str, Any]] = []
    with VramPeak() as peak:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        load_s = time.perf_counter() - t0
        vram_loaded = process_vram_mib()
        first = next(iter(data.values()))[0]
        for _ in range(warmup):
            transcribe(model, first, beam_size=max(beams), hotwords=hotwords)
        for beam in beams:
            for ds_name, samples in data.items():
                pairs: list[tuple[str, str]] = []
                lat: list[float] = []
                audio_s = 0.0
                for s in samples:
                    t = time.perf_counter()
                    hyp = transcribe(model, s, beam_size=beam, hotwords=hotwords)
                    lat.append(time.perf_counter() - t)
                    audio_s += s.seconds
                    pairs.append((s.text, hyp))
                    hyps.append(
                        {
                            "model": model_name,
                            "compute_type": compute_type,
                            "beam": beam,
                            "id": s.id,
                            "ref": s.text,
                            "hyp": hyp,
                            "cer": corpus_cer([(s.text, hyp)]),
                            "norm_ref": normalize_ja(s.text),
                            "norm_hyp": normalize_ja(hyp),
                        }
                    )
                row = {
                    "model": model_name,
                    "compute_type": compute_type,
                    "beam": beam,
                    "dataset": ds_name,
                    "n": len(samples),
                    "audio_s_mean": audio_s / max(len(samples), 1),
                    "cer": corpus_cer(pairs),
                    "latency_s": summarize_seconds(lat),
                    "rtf": sum(lat) / audio_s if audio_s else float("nan"),
                    "load_s": load_s,
                    "vram_loaded_mib": vram_loaded,
                }
                rows.append(row)
                print(
                    f"  {model_name} {compute_type} beam={beam} {ds_name}: "
                    f"CER {row['cer']:.3f}  p50 {row['latency_s']['p50'] * 1000:.0f} ms  "
                    f"p90 {row['latency_s']['p90'] * 1000:.0f} ms  RTF {row['rtf']:.3f}",
                    flush=True,
                )
        del model
        gc.collect()
    for row in rows:
        row["vram_peak_mib"] = peak.peak_mib
    return rows, hyps


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("faster-whisper", "ctranslate2", "nvidia-cublas-cu12"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "-"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--compute-types", nargs="+", default=["float16"])
    p.add_argument("--beams", nargs="+", type=int, default=[1, 5])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--n", type=int, default=100, help="各データから使う発話数")
    p.add_argument("--manifest", type=Path, help="データの代わりに、この JSONL の音声を評価する")
    p.add_argument("--hotwords", default=None, help="例: 'リーチ'（認識させたい語を事前に渡す）")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--min-free-mib", type=int, default=8000)
    p.add_argument("--out", type=Path, default=Path("bench_out/stt"))
    args = p.parse_args()

    chosen, gpus = pin_free_gpu(min_free_mib=args.min_free_mib)
    print("GPU の状況:", *(g.describe() for g in gpus), sep="\n  ")
    print("使う GPU:", chosen.describe() if chosen else os.environ.get("CUDA_VISIBLE_DEVICES"))
    _preload_cublas12()

    if args.manifest:
        data = {"manifest": load_manifest_samples(args.manifest)}
    else:
        data = {name: load_dataset_samples(name, args.n) for name in args.datasets}
    for name, samples in data.items():
        secs = [s.seconds for s in samples]
        print(f"データ {name}: {len(samples)} 発話、平均 {sum(secs) / len(secs):.1f} 秒")

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_hyps: list[dict[str, Any]] = []
    for model_name in args.models:
        for ct in args.compute_types:
            print(f"== {model_name} ({ct})", flush=True)
            rows, hyps = bench_model(
                model_name, ct, args.beams, data, hotwords=args.hotwords, warmup=args.warmup
            )
            all_rows += rows
            all_hyps += hyps

    result = {
        "started_gpus": [gpu_dict(g) for g in gpus],
        "used_gpu": gpu_dict(chosen) if chosen else None,
        "versions": _versions(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "rows": all_rows,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"stt-{stamp}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.out / f"stt-{stamp}-hyps.jsonl").open("w", encoding="utf-8") as f:
        for h in all_hyps:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"保存: {args.out / f'stt-{stamp}.json'}")


if __name__ == "__main__":
    main()
