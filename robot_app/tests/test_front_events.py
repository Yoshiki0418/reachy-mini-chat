"""型付きイベントと、記録のための JSON の往復。"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from reachy_front.core import dump_event, load_event
from reachy_front.core import events as ev
from reachy_front.core.events import AnyEvent, Event

SAMPLES: list[Event] = [
    ev.FaceTracked(track_id=1, cx=0.5, cy=0.4, width_frac=0.2),
    ev.PersonIdentified(track_id=1, person_id="p1", score=0.8),
    ev.PersonIdentified(track_id=2, person_id=None, score=0.1),
    ev.PersonEntered(track_id=1),
    ev.PersonPassedBy(track_id=3),
    ev.PersonLeft(track_id=1, person_id="p1"),
    ev.EyeContactChanged(track_id=1, active=True),
    ev.SpeechStarted(utt_id="u1", doa_deg=12.5),
    ev.SpeechEnded(utt_id="u1", duration_s=1.2),
    ev.TranscriptReceived(utt_id="u1", text="こんにちは"),
    ev.SentenceReceived(turn_id="t1", idx=0, text="やあ", emotion="happy"),
    ev.AudioChunkReceived(turn_id="t1", seq=0, nbytes=960),
    ev.TurnEnded(turn_id="t1"),
    ev.ServerConnectionChanged(connected=False),
    ev.FloorDecided(utt_id="u1", decision="respond", reasons=["name_called"]),
    ev.ProactiveDecided(intent="greet", person_id="p1"),
    ev.RobotSpeechStarted(turn_id="t1"),
    ev.RobotSpeechEnded(),
    ev.BargeIn(turn_id="t1"),
]


def test_every_event_class_has_a_sample() -> None:
    union = get_args(AnyEvent)[0]  # Annotated[Union[...], Field] の Union
    classes = set(get_args(union))
    assert {type(e) for e in SAMPLES} == classes


@pytest.mark.parametrize("event", SAMPLES, ids=lambda e: e.type)  # type: ignore[attr-defined]
def test_json_roundtrip_restores_the_same_type(event: Event) -> None:
    stamped = event.model_copy(update={"ts": 12.5, "trace_id": "abc"})
    line = dump_event(stamped)
    assert "\n" not in line
    assert load_event(line) == stamped


def test_events_are_frozen_and_reject_unknown_fields() -> None:
    e = ev.TurnEnded(turn_id="t1")
    with pytest.raises(ValidationError):
        e.turn_id = "t2"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        ev.TurnEnded.model_validate({"turn_id": "t1", "face_image": "..."})


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_event('{"type": "nope"}')


def test_field_ranges_are_checked() -> None:
    with pytest.raises(ValidationError):
        ev.FaceTracked(track_id=1, cx=1.5, cy=0.5, width_frac=0.1)
    with pytest.raises(ValidationError):
        ev.PersonIdentified(track_id=1, person_id=None, score=2)
