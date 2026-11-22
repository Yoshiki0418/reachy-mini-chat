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
    t_speech_end: float | None = None
    """発話の終わり（utterance.end を受けた時刻。utterance のみ）。体感の遅れの起点。"""
    stt_s: float | None = None
    """発話の終わり → 文字起こしができるまで（STT の時間。utterance のみ）。"""
    decision_wait_s: float | None = None
    """文字起こしを送り始める → ロボットの判定が届くまで（utterance のみ。ロボット側の判断の時間）。"""
    marks: dict[str, float] = field(default_factory=dict[str, float])
    tts_first_s: float | None = None
    """最初の文の合成にかかった時間（送信待ちは含まない）。"""
    tts_total_s: float = 0.0
    """合成にかかった時間の合計（送信待ちは含まない）。"""
    sentences: int = 0
    emitted: bool = False

    def __post_init__(self) -> None:
        self.t_start = self.clock()

    def mark(self, name: str) -> None:
        """最初の 1 回だけ記録する（first_sentence・first_audio など）。"""
        self.marks.setdefault(name, self.clock())

    def synthesized(self, seconds: float) -> None:
        """1 文の合成が終わった。``seconds`` は合成だけの時間（音声を送る時間は含めない）。"""
        if self.tts_first_s is None:
            self.tts_first_s = seconds
        self.tts_total_s += seconds
        self.sentences += 1

    def since_start(self, name: str) -> float | None:
        t = self.marks.get(name)
        return None if t is None else t - self.t_start

    def summary(self, status: str) -> dict[str, Any]:
        first_sentence = self.since_start("first_sentence")
        first_audio = self.since_start("first_audio")
        total = self.clock() - self.t_start
        # 話し終えてから第一声まで（体感の遅れ）。ロボット側の判定の時間を含む
        # 区間の足し算ではなく、時刻の差で出す（文字起こしの送信待ちなど、どの区間にも入らない時間も含む）
        t_first_audio = self.marks.get("first_audio")
        felt = (
            t_first_audio - self.t_speech_end
            if t_first_audio is not None and self.t_speech_end is not None
            else None
        )
        return {
            "event": "turn_trace",
            "turn_id": self.turn_id,
            "ref": self.ref,
            "kind": self.kind,
            "status": status,
            "stt_ms": ms(self.stt_s),
            "decision_wait_ms": ms(self.decision_wait_s),
            "first_sentence_ms": ms(first_sentence),
            "tts_first_ms": ms(self.tts_first_s),
            "first_audio_ms": ms(first_audio),
            "felt_ms": ms(felt),
            "tts_total_ms": ms(self.tts_total_s),
            "total_ms": ms(total),
            "sentences": self.sentences,
        }

    def emit(self, status: str) -> dict[str, Any] | None:
        """ログへ 1 行出す。出すのは 1 回だけ（2 回目以降は None）。"""
        if self.emitted:
            return None
        self.emitted = True
        summary = self.summary(status)
        trace_log.info(json.dumps(summary, ensure_ascii=False))
        return summary
