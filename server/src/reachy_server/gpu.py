"""共用の GPU マシンで、そのとき空いている GPU を選ぶ（標準ライブラリだけ。設計書 §4.4）。

研究室の GPU は他の人のジョブと共用なので、番号を決め打ちしない。起動時に nvidia-smi で
空きメモリと使用率を見て、条件を満たす中で一番空いているものを選ぶ。選んだ番号は
``CUDA_VISIBLE_DEVICES`` に入れるので、以後このプロセスで読み込むモデルはすべて同じ GPU に載る。

CUDA を初期化する **前に** 呼ぶこと（環境変数は初期化の後では効かない）。
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import MutableMapping
from dataclasses import dataclass

log = logging.getLogger(__name__)

_QUERY = "index,name,memory.total,memory.used,utilization.gpu"


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
    """``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` の出力を読む。

    メモリが読めない GPU は候補から外す。使用率が読めないときは 100%（埋まっている）とみなす。
    """
    gpus: list[GpuStatus] = []
    for line in csv_text.splitlines():
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
    """空きメモリが ``min_free_mib`` 以上の GPU のうち、使用率が低く、空きが多いものを選ぶ。

    メモリは足りないと落ちる（必須条件）、使用率は遅くなるだけ（優先順位）として扱う。
    """
    candidates = [g for g in gpus if g.memory_free_mib >= min_free_mib]
    if not candidates:
        return None
    return min(candidates, key=lambda g: (g.utilization_pct, -g.memory_free_mib, g.index))


def query_gpus() -> list[GpuStatus]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    return parse_gpu_query(out)


def pin_free_gpu(
    *,
    min_free_mib: int,
    env: MutableMapping[str, str] | None = None,
) -> GpuStatus | None:
    """空いている GPU を選び、``CUDA_VISIBLE_DEVICES`` を設定する。

    - すでに ``CUDA_VISIBLE_DEVICES`` があれば、手で選んだものとしてそれに従う。
    - nvidia-smi の番号と CUDA の番号をそろえるため、``CUDA_DEVICE_ORDER`` が未設定なら
      ``PCI_BUS_ID`` にする。別の並び順が指定されていれば、番号から GPU を特定できないので None。
    - nvidia-smi が使えない（GPU なし・CPU 実行）ときも None を返し、環境は変えない。

    条件を満たす GPU がなければ ``RuntimeError``（足りないまま載せて落ちるより、起動で止める）。
    """
    env = os.environ if env is None else env
    try:
        gpus = query_gpus()
    except (OSError, subprocess.SubprocessError):
        log.warning("nvidia-smi が使えない。GPU の選択は行わない")
        return None
    for g in gpus:
        log.info("%s", g.describe())

    preset = env.get("CUDA_VISIBLE_DEVICES")
    if preset is not None:
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if env["CUDA_DEVICE_ORDER"] != "PCI_BUS_ID":
            return None
        return next((g for g in gpus if str(g.index) == preset.strip()), None)

    chosen = choose_free_gpu(gpus, min_free_mib=min_free_mib)
    if chosen is None:
        raise RuntimeError(
            f"空きメモリが {min_free_mib} MiB 以上の GPU がない: "
            + "; ".join(g.describe() for g in gpus)
        )
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(chosen.index)
    log.info("使う GPU: %s", chosen.describe())
    return chosen
