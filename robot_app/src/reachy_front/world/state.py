"""World Model の状態（設計書 §3.1）。判断モジュール（宛先判定・自発発話・割り込み）が見る「いまの状況」。

すべて不変（frozen）で、更新は :mod:`.reducer` が新しい状態を作って返す。読む側は、取り出した
スナップショットが後から変わらないので、判断の途中で状況が動いても一貫して読める。
**個人の顔データや発話の内容は持たない**（``person_id`` / ``track_id`` / 時刻・真偽だけ）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TrackState:
    """カメラに映っている 1 人（追跡の単位）。"""

    track_id: int
    status: Literal["tracking", "present"] = "tracking"
    """tracking=映っただけ / present=入室と判定された。"""
    person_id: str | None = None
    id_score: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    width_frac: float = 0.0
    last_seen: float = 0.0
    entered_at: float | None = None
    eye_contact: bool = False
    eye_contact_since: float | None = None

    def eye_contact_duration(self, now: float) -> float:
        """いまのアイコンタクトの継続秒数（していなければ 0）。"""
        if not self.eye_contact or self.eye_contact_since is None:
            return 0.0
        return max(0.0, now - self.eye_contact_since)


@dataclass(frozen=True)
class RobotState:
    speaking: bool = False
    speaking_turn_id: str | None = None
    speaking_since: float | None = None
    last_spoke_end: float | None = None
    last_barge_in: float | None = None
    server_connected: bool = False


@dataclass(frozen=True)
class ConversationState:
    user_speaking: bool = False
    utt_id: str | None = None
    """いま・直前の発話（SpeechStarted の ID）。"""
    speech_started_at: float | None = None
    last_speech_ended: float | None = None
    doa_deg: float | None = None
    pending_utt_id: str | None = None
    """話し終えたが、まだ宛先の判定が出ていない発話。"""
    last_decision: Literal["respond", "backchannel", "ignore"] | None = None
    active_turn_id: str | None = None
    """サーバーが応答を返している最中のターン。"""
    last_turn_ended: float | None = None
    last_robot_question: float | None = None
    """ロボットが最後に質問した時刻（会話の継続の判定に使う）。"""
    last_proactive: float | None = None
    last_proactive_person_id: str | None = None


@dataclass(frozen=True)
class WorldState:
    ts: float = 0.0
    """最後にイベントを畳み込んだ時刻。"""
    tracks: tuple[TrackState, ...] = field(default_factory=tuple[TrackState, ...])
    robot: RobotState = field(default_factory=RobotState)
    conversation: ConversationState = field(default_factory=ConversationState)

    def track(self, track_id: int) -> TrackState | None:
        return next((t for t in self.tracks if t.track_id == track_id), None)

    @property
    def present(self) -> tuple[TrackState, ...]:
        """入室と判定された人。"""
        return tuple(t for t in self.tracks if t.status == "present")

    @property
    def people_count(self) -> int:
        return len(self.present)

    def gazing(self, now: float | None = None, *, min_s: float = 0.0) -> tuple[TrackState, ...]:
        """こちらを見ている人（``min_s`` 秒以上、見続けている人）。"""
        t = self.ts if now is None else now
        return tuple(x for x in self.tracks if x.eye_contact and x.eye_contact_duration(t) >= min_s)


__all__ = ["ConversationState", "RobotState", "TrackState", "WorldState"]
