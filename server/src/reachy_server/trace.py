"""ターン単位のトレース（区間時間）。設計書 §1.3・§4.4。

どこが遅いかを追えるよう、1 ターンの区間時間を JSON の 1 行にしてログへ出す。
人物の情報（名前・発言の内容）は出さない（設計書 §3.7）。ref・turn_id と時間だけ。

LangSmith によるトレースは、環境変数（``LANGSMITH_TRACING=true`` と ``LANGSMITH_API_KEY``）で
LangChain / LangGraph が自動で有効にする。ここでは何もしない。会話の内容がクラウドへ送られる
ので、研究室の方針を確認してから有効にすること。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

trace_log = logging.getLogger("reachy_server.trace")

Clock = Callable[[], float]


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


@dataclass
class TurnTrace:
    """1 ターンの時刻の記録。時刻は ``clock()``（単調増加）。"""

    turn_id: str
    ref: str
    kind: str  # "utterance"（発話への応答）か "proactive"（自発発話）
    clock: Clock = time.perf_counter
    t_start: float = 0.0
    stt_s: float | None = None
    """発話の終わり → 文字起こしを送るまで（utterance のみ）。"""
    decision_wait_s: float | None = None
    """文字起こしを送る → ロボットの判定が届くまで（utterance のみ。ロボット側の判断の時間）。"""
    marks: dict[str, float] = field(default_factory=dict[str, float])
    tts_total_s: float = 0.0
    sentences: int = 0

    def __post_init__(self) -> None:
        self.t_start = self.clock()

    def mark(self, name: str) -> None:
        """最初の 1 回だけ記録する（first_sentence・first_audio など）。"""
        self.marks.setdefault(name, self.clock())

    def since_start(self, name: str) -> float | None:
        t = self.marks.get(name)
        return None if t is None else t - self.t_start

    def summary(self, status: str) -> dict[str, Any]:
        first_sentence = self.since_start("first_sentence")
        first_audio = self.since_start("first_audio")
        total = self.clock() - self.t_start
        tts_first = (
            first_audio - first_sentence
            if first_audio is not None and first_sentence is not None
            else None
        )
        # 話し終えてから第一声まで（体感の遅れ）。ロボット側の判定の時間を含む
        felt: float | None = None
        if self.stt_s is not None and self.decision_wait_s is not None and first_audio is not None:
            felt = self.stt_s + self.decision_wait_s + first_audio
        return {
            "event": "turn_trace",
            "turn_id": self.turn_id,
            "ref": self.ref,
            "kind": self.kind,
            "status": status,
            "stt_ms": ms(self.stt_s),
            "decision_wait_ms": ms(self.decision_wait_s),
            "first_sentence_ms": ms(first_sentence),
            "tts_first_ms": ms(tts_first),
            "first_audio_ms": ms(first_audio),
            "felt_ms": ms(felt),
            "tts_total_ms": ms(self.tts_total_s),
            "total_ms": ms(total),
            "sentences": self.sentences,
        }

    def emit(self, status: str) -> dict[str, Any]:
        summary = self.summary(status)
        trace_log.info(json.dumps(summary, ensure_ascii=False))
        return summary
