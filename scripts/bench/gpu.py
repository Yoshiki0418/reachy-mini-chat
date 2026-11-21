"""共用の GPU マシンで、そのとき空いている GPU を選ぶ（ベンチ用。標準ライブラリだけで動く）。

研究室の GPU は他の人のジョブと共用なので、番号を決め打ちしない。起動時に nvidia-smi で
空きメモリと使用率を見て、条件を満たす中で一番空いている GPU を選ぶ。

torch / ctranslate2 を import する **前に** `pin_free_gpu()` を呼ぶこと。
`CUDA_VISIBLE_DEVICES` は CUDA の初期化より前にしか効かない。
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import MutableMapping
from dataclasses import asdict, dataclass
from typing import Any

GPU_QUERY = "index,name,memory.total,memory.used,utilization.gpu"


@dataclass(frozen=True)
class GpuStatus:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int

    @property
    def memory_free_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib

    def describe(self) -> str:
        return (
            f"GPU{self.index} {self.name}: 使用中 {self.memory_used_mib}/{self.memory_total_mib} MiB,"
            f" 使用率 {self.utilization_pct}%"
        )


def _to_int(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:  # "[N/A]" など
        return None


def parse_gpu_query(csv_text: str) -> list[GpuStatus]:
    """`nvidia-smi --query-gpu=GPU_QUERY --format=csv,noheader,nounits` の出力を読む。

    メモリが読めない GPU は候補から外す。使用率が読めないときは 100%（埋まっている）とみなす。
    """
    gpus: list[GpuStatus] = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        # GPU 名にカンマは入らない前提（nvidia-smi の CSV も引用符を付けない）
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 5:
            continue
        index, total, used = _to_int(cols[0]), _to_int(cols[2]), _to_int(cols[3])
        if index is None or total is None or used is None:
            continue
        util = _to_int(cols[4])
        gpus.append(GpuStatus(index, cols[1], total, used, 100 if util is None else util))
    return gpus


def choose_free_gpu(gpus: list[GpuStatus], *, min_free_mib: int = 0) -> GpuStatus | None:
    """空きメモリが `min_free_mib` 以上の GPU のうち、使用率が低く、空きが多いものを選ぶ。

    メモリは足りないと落ちる（必須条件）、使用率は遅くなるだけ（優先順位）、として扱う。
    """
    candidates = [g for g in gpus if g.memory_free_mib >= min_free_mib]
    if not candidates:
        return None
    return min(candidates, key=lambda g: (g.utilization_pct, -g.memory_free_mib, g.index))


def query_gpus() -> list[GpuStatus]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_gpu_query(out)


def pin_free_gpu(
    *, min_free_mib: int, env: MutableMapping[str, str] | None = None
) -> tuple[GpuStatus | None, list[GpuStatus]]:
    """空いている GPU を選んで `CUDA_VISIBLE_DEVICES` に入れる。

    すでに `CUDA_VISIBLE_DEVICES` が設定されていれば、それに従う（手で選びたいとき用）。
    手で選んだ番号も nvidia-smi の番号として扱うので、`CUDA_DEVICE_ORDER` が未設定なら
    PCI_BUS_ID にそろえる。別の並び順が指定されていれば、どの GPU か特定できないので None を返す。
    戻り値は（選んだ GPU、計測時の全 GPU の状況）。条件を満たす GPU がなければ SystemExit。
    """
    env = os.environ if env is None else env
    gpus = query_gpus()
    preset = env.get("CUDA_VISIBLE_DEVICES")
    if preset is not None:
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if env["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
            return None, gpus
        chosen = next((g for g in gpus if str(g.index) == preset.strip()), None)
        return chosen, gpus
    chosen = choose_free_gpu(gpus, min_free_mib=min_free_mib)
    if chosen is None:
        lines = "\n".join(g.describe() for g in gpus)
        raise SystemExit(f"空きメモリが {min_free_mib} MiB 以上の GPU がありません:\n{lines}")
    # nvidia-smi の番号と CUDA の番号をそろえる（既定の FASTEST_FIRST だとずれることがある）
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(chosen.index)
    return chosen, gpus


def parse_process_vram(csv_text: str, pid: int) -> int:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits` から、
    指定プロセスが使っている VRAM（MiB）を合計する。"""
    total = 0
    for line in csv_text.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 2:
            continue
        p, used = _to_int(cols[0]), _to_int(cols[1])
        if p == pid and used is not None:
            total += used
    return total


def process_vram_mib(pid: int | None = None) -> int:
    """このプロセス（または指定 PID）が使っている VRAM。CUDA コンテキスト分も含む。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_process_vram(out, os.getpid() if pid is None else pid)


def gpu_dict(g: GpuStatus) -> dict[str, Any]:
    """結果の JSON に書く形。"""
    return asdict(g)


class VramPeak:
    """別スレッドで nvidia-smi を見て、このプロセスの VRAM の最大値を記録する。

    torch のアロケータの統計と違い、CUDA コンテキストやライブラリの作業領域も含む
    （同じ GPU に何を載せられるかを決めるのは、こちらの値）。
    """

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, process_vram_mib())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> VramPeak:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, process_vram_mib())
