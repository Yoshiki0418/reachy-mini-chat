"""ロボット内部のイベント（設計書 §3.2）。モジュール間はこの型だけで話す。

- すべて不変（frozen）。``ts`` はバスが発行時に時計で押す。``trace_id`` は、ある出来事から
  連鎖して起きたイベントに同じ値が付く（:mod:`.bus`）
- 記録と再生（#37）のため、JSON にできて、``type`` から元の型に戻せる（:data:`AnyEvent`）
- **個人の顔データや音声の中身は載せない**。人物は ``person_id``（ID）、画面上の追跡は ``track_id``
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from reachy_protocol.messages import Emotion


class Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: float | None = Field(default=None, description="発行時刻（秒、バスの時計）。バスが押す")
    trace_id: str | None = Field(default=None, description="連鎖して起きたイベントに共通のID")


# ---- perception ----------------------------------------------------------------------------


class FaceTracked(Event):
    type: Literal["face_tracked"] = "face_tracked"
    track_id: int
    cx: float = Field(ge=0, le=1, description="顔の中心（画面の幅に対する割合）")
    cy: float = Field(ge=0, le=1)
    width_frac: float = Field(ge=0, le=1, description="顔幅/画面幅")


class PersonIdentified(Event):
    type: Literal["person_identified"] = "person_identified"
    track_id: int
    person_id: str | None = Field(description="None=知らない人")
    score: float = Field(ge=0, le=1)


class PersonEntered(Event):
    type: Literal["person_entered"] = "person_entered"
    track_id: int
    person_id: str | None = None


class PersonPassedBy(Event):
    type: Literal["person_passed_by"] = "person_passed_by"
    track_id: int


class PersonLeft(Event):
    type: Literal["person_left"] = "person_left"
    track_id: int
    person_id: str | None = None


class EyeContactChanged(Event):
    type: Literal["eye_contact_changed"] = "eye_contact_changed"
    track_id: int
    active: bool


class SpeechStarted(Event):
    type: Literal["speech_started"] = "speech_started"
    utt_id: str
    doa_deg: float | None = Field(default=None, description="音の到来方向（度）")


class SpeechEnded(Event):
    type: Literal["speech_ended"] = "speech_ended"
    utt_id: str
    duration_s: float = Field(ge=0)
    doa_deg: float | None = None


# ---- client（サーバーから）-----------------------------------------------------------------


class TranscriptReceived(Event):
    type: Literal["transcript_received"] = "transcript_received"
    utt_id: str
    text: str


class SentenceReceived(Event):
    type: Literal["sentence_received"] = "sentence_received"
    turn_id: str
    idx: int = Field(ge=0)
    text: str
    emotion: Emotion = "neutral"


class AudioChunkReceived(Event):
    type: Literal["audio_chunk_received"] = "audio_chunk_received"
    turn_id: str
    seq: int = Field(ge=0)
    nbytes: int = Field(ge=0)


class TurnEnded(Event):
    type: Literal["turn_ended"] = "turn_ended"
    turn_id: str
    cancelled: bool = False


class ServerConnectionChanged(Event):
    type: Literal["server_connection_changed"] = "server_connection_changed"
    connected: bool


# ---- social --------------------------------------------------------------------------------


class FloorDecided(Event):
    type: Literal["floor_decided"] = "floor_decided"
    utt_id: str
    decision: Literal["respond", "backchannel", "ignore"]
    reasons: list[str] = Field(default_factory=list)


class ProactiveDecided(Event):
    type: Literal["proactive_decided"] = "proactive_decided"
    intent: str
    person_id: str | None = None
    reasons: list[str] = Field(default_factory=list)


# ---- expression ----------------------------------------------------------------------------


class RobotSpeechStarted(Event):
    type: Literal["robot_speech_started"] = "robot_speech_started"
    turn_id: str | None = None


class RobotSpeechEnded(Event):
    type: Literal["robot_speech_ended"] = "robot_speech_ended"
    turn_id: str | None = None


class BargeIn(Event):
    type: Literal["barge_in"] = "barge_in"
    turn_id: str | None = Field(default=None, description="中断された、ロボットの発話のターン")


AnyEvent = Annotated[
    FaceTracked
    | PersonIdentified
    | PersonEntered
    | PersonPassedBy
    | PersonLeft
    | EyeContactChanged
    | SpeechStarted
    | SpeechEnded
    | TranscriptReceived
    | SentenceReceived
    | AudioChunkReceived
    | TurnEnded
    | ServerConnectionChanged
    | FloorDecided
    | ProactiveDecided
    | RobotSpeechStarted
    | RobotSpeechEnded
    | BargeIn,
    Field(discriminator="type"),
]

_ADAPTER: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def dump_event(event: Event) -> str:
    """記録用の 1 行の JSON。"""
    return event.model_dump_json(exclude_none=False)


def load_event(line: str) -> Event:
    """``dump_event`` の逆。未知の ``type`` は ValidationError。"""
    return _ADAPTER.validate_json(line)


__all__ = [
    "AnyEvent",
    "AudioChunkReceived",
    "BargeIn",
    "Event",
    "EyeContactChanged",
    "FaceTracked",
    "FloorDecided",
    "PersonEntered",
    "PersonIdentified",
    "PersonLeft",
    "PersonPassedBy",
    "ProactiveDecided",
    "RobotSpeechEnded",
    "RobotSpeechStarted",
    "SentenceReceived",
    "ServerConnectionChanged",
    "SpeechEnded",
    "SpeechStarted",
    "TranscriptReceived",
    "TurnEnded",
    "dump_event",
    "load_event",
]
