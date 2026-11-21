"""ベンチの指標（標準ライブラリだけで動く）: 日本語の CER と、処理時間の要約。"""

from __future__ import annotations

import math
import unicodedata

# 句読点・記号・空白は評価から外す（STT の表記ゆれで差が出やすく、読み上げには影響しない）
_DROP_CATEGORIES = ("P", "S", "Z", "C")


def normalize_ja(text: str) -> str:
    """CER 用の正規化: NFKC（全角英数→半角など）、小文字化、句読点・記号・空白の除去。

    漢数字と算用数字（「五日」と「5日」）の違いはそのまま残る。モデル間の比較には影響が小さい。
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith(_DROP_CATEGORIES))


def edit_distance(ref: str, hyp: str) -> int:
    """文字単位のレーベンシュタイン距離（置換・挿入・削除がそれぞれ 1）。"""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    """（正解, 認識結果）の組から、コーパス全体の CER（誤り文字数の合計 / 正解文字数の合計）。"""
    errors = 0
    chars = 0
    for ref, hyp in pairs:
        r, h = normalize_ja(ref), normalize_ja(hyp)
        errors += edit_distance(r, h)
        chars += len(r)
    if chars == 0:
        return 0.0
    return errors / chars


def percentile(values: list[float], q: float) -> float:
    """線形補間の分位点（q は 0〜100）。空なら NaN。"""
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_seconds(values: list[float]) -> dict[str, float]:
    """処理時間の要約（平均・中央値・p90・最大）。"""
    if not values:
        return {"mean": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "max": max(values),
        "n": len(values),
    }
