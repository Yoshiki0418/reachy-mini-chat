"""reachy_server.gpu: 共用の GPU マシンで、そのとき空いている GPU を選ぶ。"""

from __future__ import annotations

import subprocess

import pytest

from reachy_server import gpu

# 2026-11-20 に実際に見た状況: GPU0 は他の人のジョブで埋まり、GPU1 が空いていた
SMI_BUSY0 = "0, NVIDIA RTX A6000, 49140, 46070, 100\n1, NVIDIA RTX A6000, 49140, 3466, 0\n"


def _patch_smi(monkeypatch: pytest.MonkeyPatch, text: str = SMI_BUSY0) -> None:
    monkeypatch.setattr(gpu, "query_gpus", lambda: gpu.parse_gpu_query(text))


def test_parse_gpu_query() -> None:
    gpus = gpu.parse_gpu_query(SMI_BUSY0)
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].memory_free_mib == 49140 - 46070
    assert gpus[1].utilization_pct == 0


def test_parse_gpu_query_unreadable_values() -> None:
    gpus = gpu.parse_gpu_query("0, A, [N/A], 10, 5\n1, B, 1000, 10, [N/A]\n\nbroken line\n")
    # メモリが読めない GPU は外す。使用率が読めない GPU は埋まっているとみなす
    assert [(g.index, g.utilization_pct) for g in gpus] == [(1, 100)]


def test_choose_avoids_busy_gpu() -> None:
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(SMI_BUSY0), min_free_mib=12000)
    assert chosen is not None and chosen.index == 1


def test_choose_follows_whichever_is_free() -> None:
    # 状況が逆になったら GPU0 を選ぶ（番号を決め打ちしない）
    smi = "0, A6000, 49140, 500, 0\n1, A6000, 49140, 47000, 98\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi), min_free_mib=12000)
    assert chosen is not None and chosen.index == 0


def test_choose_memory_is_a_hard_requirement() -> None:
    # 使用率が低くても、メモリが足りない GPU は選ばない
    smi = "0, A6000, 49140, 45000, 0\n1, A6000, 49140, 20000, 60\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi), min_free_mib=12000)
    assert chosen is not None and chosen.index == 1


def test_choose_prefers_more_free_memory_on_tie() -> None:
    smi = "0, A6000, 49140, 20000, 0\n1, A6000, 49140, 5000, 0\n"
    chosen = gpu.choose_free_gpu(gpu.parse_gpu_query(smi))
    assert chosen is not None and chosen.index == 1


def test_choose_none_when_nothing_fits() -> None:
    assert gpu.choose_free_gpu(gpu.parse_gpu_query(SMI_BUSY0), min_free_mib=48000) is None
    assert gpu.choose_free_gpu([]) is None


def test_pin_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_smi(monkeypatch)
    env: dict[str, str] = {}
    chosen = gpu.pin_free_gpu(min_free_mib=12000, env=env)
    assert chosen is not None and chosen.index == 1
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}


def test_pin_respects_manual_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_smi(monkeypatch)
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    chosen = gpu.pin_free_gpu(min_free_mib=12000, env=env)
    assert chosen is not None and chosen.index == 0
    # 手で選んだ番号も nvidia-smi の番号として効くよう、並び順をそろえる
    assert env == {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_pin_manual_choice_with_other_order_is_not_identified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_smi(monkeypatch)
    env = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    assert gpu.pin_free_gpu(min_free_mib=12000, env=env) is None
    assert env == {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "FASTEST_FIRST"}


def test_pin_raises_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_smi(monkeypatch)
    with pytest.raises(RuntimeError, match="空きメモリ"):
        gpu.pin_free_gpu(min_free_mib=48000, env={})


def test_pin_without_nvidia_smi_leaves_env_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[gpu.GpuStatus]:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu, "query_gpus", boom)
    env: dict[str, str] = {}
    assert gpu.pin_free_gpu(min_free_mib=1, env=env) is None
    assert env == {}


def test_pin_with_failing_nvidia_smi_leaves_env_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> list[gpu.GpuStatus]:
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    monkeypatch.setattr(gpu, "query_gpus", boom)
    env: dict[str, str] = {}
    assert gpu.pin_free_gpu(min_free_mib=1, env=env) is None
    assert env == {}
