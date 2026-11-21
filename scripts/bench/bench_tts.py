"""Irodori-TTS の合成時間・VRAM の計測と、声の候補づくり（#14）。

Irodori-TTS は torch / torchcodec の版がワークスペースと合わないので、Irodori-TTS 自身の環境で動かす:

    git clone https://github.com/Aratako/Irodori-TTS ~/Irodori-TTS
    (cd ~/Irodori-TTS && uv sync --extra cu128)
    ~/Irodori-TTS/.venv/bin/python scripts/bench/bench_tts.py voices --out bench_out/voices
    ~/Irodori-TTS/.venv/bin/python scripts/bench/bench_tts.py speed \\
        --ref-wav bench_out/voices/v1_s0.wav --out bench_out/tts

- `voices`: キャプション（声の説明文）だけから声を作り、候補の音声を書き出す。実在の人の声は
  複製しない（モデルカードの利用条件）。
- `speed`: 設定（チェックポイント・精度・ステップ数）ごとに、受付で話す文を 1 文ずつ合成して
  時間と VRAM を測る。書き出した `manifest.jsonl` を `bench_stt.py --manifest` に渡すと、
  合成音声の聞き取りやすさ（CER）が測れる。

Irodori-TTS の場所は IRODORI_DIR（既定 ~/Irodori-TTS）。GPU は起動時に空いているものを選ぶ。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu import VramPeak, gpu_dict, pin_free_gpu
from metrics import summarize_seconds

SMALL = "Aratako/Irodori-TTS-v4.1-Small"
SMALL_MF = "Aratako/Irodori-TTS-v4.1-Small-MF"


@dataclass(frozen=True)
class TtsConfig:
    checkpoint: str
    precision: str  # fp32 / bf16
    num_steps: int | None = None  # None = チェックポイントの既定（RF 40, MeanFlow 4）
    schedule: str = "linear"  # linear / sway


CONFIGS: dict[str, TtsConfig] = {
    "small-rf40-fp32": TtsConfig(SMALL, "fp32"),
    "small-rf40-bf16": TtsConfig(SMALL, "bf16"),
    "small-sway12-bf16": TtsConfig(SMALL, "bf16", 12, "sway"),
    "small-sway6-bf16": TtsConfig(SMALL, "bf16", 6, "sway"),
    "mf4-fp32": TtsConfig(SMALL_MF, "fp32"),
    "mf4-bf16": TtsConfig(SMALL_MF, "bf16"),
}

# 声の候補（キャプションは声と話し方の説明。特定の人を指す言葉は入れない）
VOICE_CAPTIONS: dict[str, str] = {
    "v1": "明るく元気な若い女性。アニメのキャラクターのような高めで可愛らしい声で、親しい相手に楽しそうに話しかけている。",
    "v2": "甘く柔らかい声の少女。にこにこしながら、やさしく親しげに話している。",
    "v3": "元気いっぱいで弾むような声の女の子。配信で視聴者にあいさつするように、テンション高めに話している。",
    "v4": "透明感のある澄んだ声の若い女性。穏やかで、聞いていて癒される話し方。",
    "v5": "少し幼さの残る高い声の女の子。無邪気で、表情豊かに話している。",
    "v6": "明るくはきはきとした、可愛らしい声の受付の女性。丁寧で聞き取りやすく、笑顔で話している。",
}
VOICE_TEXT = "おはよう！今日も来てくれてうれしいな。わたしはリーチ、よろしくね！"

# 受付で話す文（エージェントは 1〜3 文で返し、サーバーは 1 文ずつ合成して送る）
SENTENCES = [
    "おはよう！",
    "おかえり！",
    "リーチだよ、よろしくね！",
    "五日連続だね、すごい！",
    "久しぶり！元気にしてた？",
    "今日は朝から来てくれたんだね、うれしいな。",
    "お名前を教えてもらってもいいかな？",
    "わかった、覚えておくね。",
    "今日は研究室に三人来てるよ。",
    "実験うまくいくといいね、応援してるよ！",
    "午後から雨が降るみたいだから、帰りは傘を忘れないでね。",
    "お疲れさま、今日もがんばったね。また明日ね！",
]


def _import_irodori(irodori_dir: Path) -> dict[str, Any]:
    """Irodori-TTS は仮想プロジェクト（パッケージとして入らない）なので、パスを足して import する。"""
    sys.path.insert(0, str(irodori_dir))
    from irodori_tts import length_check  # pyright: ignore[reportMissingImports]
    from irodori_tts.inference_runtime import (  # pyright: ignore[reportMissingImports]
        InferenceRuntime,
        RuntimeKey,
        SamplingRequest,
        download_hf_checkpoint,
        resolve_cfg_scales,
        save_wav,
    )

    return {
        "length_check": length_check,
        "InferenceRuntime": InferenceRuntime,
        "RuntimeKey": RuntimeKey,
        "SamplingRequest": SamplingRequest,
        "download_hf_checkpoint": download_hf_checkpoint,
        "resolve_cfg_scales": resolve_cfg_scales,
        "save_wav": save_wav,
    }


class Synth:
    """1 つの設定でモデルを読み込み、1 文ずつ合成する。"""

    def __init__(self, ir: dict[str, Any], cfg: TtsConfig, *, length_check: bool) -> None:
        self.ir = ir
        self.cfg = cfg
        self.length_check = length_check
        path = ir["download_hf_checkpoint"](cfg.checkpoint)
        self.runtime = ir["InferenceRuntime"].from_key(
            ir["RuntimeKey"](
                checkpoint=path,
                model_device="cuda",
                model_precision=cfg.precision,
                codec_device="cuda",
                codec_precision="fp32",
            )
        )

    def __call__(
        self, text: str, *, caption: str | None, ref_wav: str | None, seed: int
    ) -> tuple[Any, int, list[tuple[str, float]], dict[str, Any]]:
        use_caption = bool(caption)
        use_ref = ref_wav is not None
        # infer.py と同じく、使わない条件の CFG は切る（使う条件だけ推論が増える）
        cfg_text, cfg_caption, cfg_speaker, _ = self.ir["resolve_cfg_scales"](
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_scale=None,
            use_caption_condition=use_caption
            and bool(self.runtime.model_cfg.use_caption_condition),
            use_speaker_condition=use_ref,
        )
        req = self.ir["SamplingRequest"](
            text=text,
            caption=caption if use_caption else None,
            ref_wav=ref_wav,
            no_ref=not use_ref,
            seed=seed,
            num_steps=self.cfg.num_steps,
            t_schedule_mode=self.cfg.schedule,
            cfg_scale_text=cfg_text,
            cfg_scale_caption=cfg_caption,
            cfg_scale_speaker=cfg_speaker,
        )
        if self.length_check:
            result, info = self.ir["length_check"].synthesize(self.runtime, req)
        else:
            result, info = self.runtime.synthesize(req), {}
        return result.audio, result.sample_rate, result.stage_timings, info

    def unload(self) -> None:
        import torch  # pyright: ignore[reportMissingImports]

        self.runtime.unload()
        del self.runtime
        gc.collect()
        torch.cuda.empty_cache()


def _irodori_commit(irodori_dir: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(irodori_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "-"


def cmd_voices(args: argparse.Namespace, ir: dict[str, Any]) -> list[dict[str, Any]]:
    synth = Synth(ir, CONFIGS[args.config], length_check=False)
    rows = []
    for vid, caption in VOICE_CAPTIONS.items():
        if args.voices and vid not in args.voices:
            continue
        for seed in args.seeds:
            t = time.perf_counter()
            audio, sr, _, _ = synth(args.text, caption=caption, ref_wav=None, seed=seed)
            elapsed = time.perf_counter() - t
            name = f"{vid}_s{seed}.wav"
            ir["save_wav"](args.out / name, audio, sr)
            rows.append(
                {"voice": vid, "seed": seed, "file": name, "caption": caption, "elapsed_s": elapsed}
            )
            print(f"  {name}: {elapsed:.2f} s  {caption}", flush=True)
    synth.unload()
    lines = [
        "# 声の候補（Irodori-TTS v4.1-Small、キャプションのみ）",
        "",
        f"文: {args.text}",
        "",
        "| ファイル | キャプション |",
        "|---|---|",
        *(f"| {r['file']} | {r['caption']} |" for r in rows),
        "",
    ]
    (args.out / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def cmd_speed(args: argparse.Namespace, ir: dict[str, Any]) -> list[dict[str, Any]]:
    import torch  # pyright: ignore[reportMissingImports]

    ref = str(args.ref_wav.resolve()) if args.ref_wav else None
    modes = {
        "caption": {"caption": args.caption, "ref_wav": None},
        "ref": {"caption": None, "ref_wav": ref},
        "ref+caption": {"caption": args.caption, "ref_wav": ref},
    }
    rows: list[dict[str, Any]] = []
    manifest = (args.out / "manifest.jsonl").open("a", encoding="utf-8")
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"== {cfg_name}: {cfg}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        with VramPeak() as peak:
            t = time.perf_counter()
            synth = Synth(ir, cfg, length_check=args.length_check)
            load_s = time.perf_counter() - t
            for mode in args.modes:
                kw = modes[mode]
                if "ref" in mode and ref is None:
                    raise SystemExit(f"--modes {mode} には --ref-wav が要る")
                for w in range(args.warmup):
                    synth(SENTENCES[5], seed=1000 + w, **kw)
                lat: list[float] = []
                audio_s = 0.0
                stages: dict[str, float] = {}
                per_sentence = []
                for rep in range(args.repeats):
                    for i, text in enumerate(SENTENCES):
                        torch.cuda.synchronize()
                        t = time.perf_counter()
                        audio, sr, timings, info = synth(text, seed=rep * 100 + i, **kw)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t
                        secs = audio.shape[-1] / sr
                        lat.append(elapsed)
                        audio_s += secs
                        for stage, sec in timings:
                            stages[stage] = stages.get(stage, 0.0) + sec
                        per_sentence.append(
                            {"i": i, "rep": rep, "elapsed_s": elapsed, "audio_s": secs, **info}
                        )
                        wav = Path(cfg_name) / mode / f"{i:02d}_r{rep}.wav"
                        ir["save_wav"](args.out / wav, audio, sr)
                        item = {"id": f"{cfg_name}/{mode}/{i:02d}_r{rep}", "wav": str(wav)}
                        manifest.write(
                            json.dumps({**item, "text": text}, ensure_ascii=False) + "\n"
                        )
                n = len(lat)
                row = {
                    "config": cfg_name,
                    **cfg.__dict__,
                    "mode": mode,
                    "length_check": args.length_check,
                    "sdpa_cudnn": os.environ.get("IRODORI_SDPA_CUDNN", "1"),
                    "n": n,
                    "latency_s": summarize_seconds(lat),
                    "audio_s_mean": audio_s / n,
                    "rtf": sum(lat) / audio_s,
                    "stage_s_mean": {k: v / n for k, v in stages.items()},
                    "load_s": load_s,
                    "per_sentence": per_sentence,
                }
                rows.append(row)
                ls = row["latency_s"]
                print(
                    f"  {mode}: p50 {ls['p50'] * 1000:.0f} ms  p90 {ls['p90'] * 1000:.0f} ms"
                    f"  max {ls['max'] * 1000:.0f} ms  RTF {row['rtf']:.3f}"
                    f"  音声 {row['audio_s_mean']:.2f} s/文",
                    flush=True,
                )
            synth.unload()
        torch_peak = torch.cuda.max_memory_allocated() / 2**20
        for row in rows:
            if row["config"] == cfg_name:
                row["vram_peak_mib"] = peak.peak_mib
                row["torch_peak_allocated_mib"] = torch_peak
        print(f"  VRAM ピーク {peak.peak_mib} MiB（torch の確保 {torch_peak:.0f} MiB）", flush=True)
    manifest.close()
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("voices", help="キャプションから声の候補を作る")
    pv.add_argument("--config", default="small-rf40-fp32", choices=list(CONFIGS))
    pv.add_argument("--voices", nargs="*", help="作る候補（既定はすべて）")
    pv.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    pv.add_argument("--text", default=VOICE_TEXT)
    ps = sub.add_parser("speed", help="設定ごとの合成時間と VRAM を測る")
    ps.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ps.add_argument(
        "--modes", nargs="+", default=["caption", "ref"], choices=["caption", "ref", "ref+caption"]
    )
    ps.add_argument("--ref-wav", type=Path, help="話者の参照音声（声を固定する）")
    ps.add_argument("--caption", default=VOICE_CAPTIONS["v1"])
    ps.add_argument("--repeats", type=int, default=2)
    ps.add_argument("--warmup", type=int, default=2)
    ps.add_argument("--length-check", action="store_true", help="短い文で長さを測り直す（2 パス）")
    for sp in (pv, ps):
        sp.add_argument("--out", type=Path, required=True)
        sp.add_argument("--min-free-mib", type=int, default=12000)
        sp.add_argument(
            "--sdpa-cudnn",
            choices=["0", "1"],
            default=None,
            help="IRODORI_SDPA_CUDNN（0 にすると長さが毎回変わる入力で速くなることがある）",
        )
    args = p.parse_args()

    if args.sdpa_cudnn is not None:
        os.environ["IRODORI_SDPA_CUDNN"] = args.sdpa_cudnn
    chosen, gpus = pin_free_gpu(min_free_mib=args.min_free_mib)
    print("GPU の状況:", *(g.describe() for g in gpus), sep="\n  ")
    print("使う GPU:", chosen.describe() if chosen else os.environ.get("CUDA_VISIBLE_DEVICES"))

    irodori_dir = Path(os.environ.get("IRODORI_DIR", Path.home() / "Irodori-TTS"))
    ir = _import_irodori(irodori_dir)
    import torch  # pyright: ignore[reportMissingImports]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = cmd_voices(args, ir) if args.cmd == "voices" else cmd_speed(args, ir)
    result = {
        "cmd": args.cmd,
        "started_gpus": [gpu_dict(g) for g in gpus],
        "used_gpu": gpu_dict(chosen) if chosen else None,
        "versions": {"irodori_commit": _irodori_commit(irodori_dir), "torch": torch.__version__},
        "args": {k: str(v) for k, v in vars(args).items()},
        "rows": rows,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = args.out / f"tts-{args.cmd}-{stamp}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
