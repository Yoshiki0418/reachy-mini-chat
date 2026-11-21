"""scripts/bench/ の判断ロジック: 空いている GPU の選び方と、CER の計算。"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"bench_{name}", _BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass は定義中に sys.modules から自分のモジュールを引くので、先に登録する
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gpu = _load("gpu")
metrics = _load("metrics")

# 2026-11-20 に実際に見た状況: GPU0 は他の人のジョブで埋まり、GPU1 が空いていた
SMI_BUSY0 = "0, NVIDIA RTX A6000, 49140, 46070, 100\n1, NVIDIA RTX A6000, 49140, 3466, 0\n"


def test_parse_gpu_query() -> None:
    gpus = gpu.parse_gpu_query(SMI_BUSY0)
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].memory_free_mib == 49140 - 46070
    assert gpus[1].utilization_pct == 0


def test_parse_gpu_query_unreadable_values() -> None:
    gpus = gpu.parse_gpu_query("0, A, [N/A], 10, 5\n1, B, 1000, 10, [N/A]\n\nbroken line\n")
    # メモリが読めない GPU は外す。使用率が読めない GPU は埋まっているとみなす
    assert [(g.index, g.utilization_pct) for g in gpus] == [(1, 100)]


def test_choose_free_gpu_avoids_busy_gpu() -> None:
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(SMI_BUSY0), min_free_mib=12000)
    assert chosen is not None and chosen.index == 1


def test_choose_free_gpu_follows_whichever_is_free() -> None:
    # 状況が逆になったら GPU0 を選ぶ（番号を決め打ちしない）
    smi = "0, A6000, 49140, 500, 0\n1, A6000, 49140, 47000, 98\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi), min_free_mib=12000)
    assert chosen is not None and chosen.index == 0


def test_choose_free_gpu_memory_is_a_hard_requirement() -> None:
    # 使用率が低くても、メモリが足りない GPU は選ばない
    smi = "0, A6000, 49140, 45000, 0\n1, A6000, 49140, 20000, 60\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi), min_free_mib=12000)
    assert chosen is not None and chosen.index == 1


def test_choose_free_gpu_prefers_more_free_memory_on_tie() -> None:
    smi = "0, A6000, 49140, 20000, 0\n1, A6000, 49140, 5000, 0\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi))
    assert chosen is not None and chosen.index == 1


def test_choose_free_gpu_none_when_nothing_fits() -> None:
    assert gpu.choose_free_gpu(gpu.parse_gpu_query(SMI_BUSY0), min_free_mib=48000) is None
    assert gpu.choose_free_gpu([]) is None


def test_pin_free_gpu_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "query_gpus", lambda: gpu.parse_gpu_query(SMI_BUSY0))
    env: dict[str, str] = {}
    chosen, _ = gpu.pin_free_gpu(min_free_mib=12000, env=env)
    assert chosen is not None and chosen.index == 1
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}


def test_pin_free_gpu_respects_manual_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "query_gpus", lambda: gpu.parse_gpu_query(SMI_BUSY0))
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    chosen, _ = gpu.pin_free_gpu(min_free_mib=12000, env=env)
    assert chosen is not None and chosen.index == 0
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}


def test_pin_free_gpu_exits_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "query_gpus", lambda: gpu.parse_gpu_query(SMI_BUSY0))
    with pytest.raises(SystemExit):
        gpu.pin_free_gpu(min_free_mib=48000, env={})


def test_parse_process_vram() -> None:
    smi = "1234, 3000\n999, 45720\n1234, 500\n[N/A], [N/A]\n"
    assert gpu.parse_process_vram(smi, 1234) == 3500
    assert gpu.parse_process_vram(smi, 42) == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("おはよう、リーチ！", "おはようリーチ"),
        ("ＡＢＣ　１２３。", "abc123"),
        ("「今日は」 晴れ?", "今日は晴れ"),
    ],
)
def test_normalize_ja(text: str, expected: str) -> None:
    assert metrics.normalize_ja(text) == expected


@pytest.mark.parametrize(
    ("ref", "hyp", "dist"),
    [
        ("", "", 0),
        ("あいう", "あいう", 0),
        ("あいう", "あう", 1),
        ("あいう", "かいうえ", 2),
        ("", "あ", 1),
    ],
)
def test_edit_distance(ref: str, hyp: str, dist: int) -> None:
    assert metrics.edit_distance(ref, hyp) == dist
    assert metrics.edit_distance(hyp, ref) == dist


def test_corpus_cer_weights_by_reference_length() -> None:
    # 誤り 1 文字 / 正解 10 文字（句読点は数えない）
    pairs = [("あいうえお。", "あいうえこ"), ("かきくけこ", "かきくけこ")]
    assert metrics.corpus_cer(pairs) == pytest.approx(0.1)
    assert metrics.corpus_cer([]) == 0.0


def test_summarize_seconds() -> None:
    s = metrics.summarize_seconds([0.1, 0.2, 0.3, 0.4, 1.0])
    assert s["mean"] == pytest.approx(0.4)
    assert s["p50"] == pytest.approx(0.3)
    assert s["p90"] == pytest.approx(0.76)
    assert s["max"] == 1.0
    assert math.isnan(metrics.summarize_seconds([])["p50"])
