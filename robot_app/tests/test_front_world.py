"""World Model の reducer と World。"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from reachy_front.core import EventBus, ManualClock
from reachy_front.core import events as ev
from reachy_front.core.events import Event
from reachy_front.world import World, WorldState, reduce


def fold(events: Sequence[Event], state: WorldState | None = None) -> WorldState:
    s = state or WorldState()
    for e in events:
        s = reduce(s, e)
    return s


def at(ts: float, e: Event) -> Event:
    return e.model_copy(update={"ts": ts})


# ---------------------------------------------------------------- 在室者


def test_face_tracking_creates_and_updates_a_track() -> None:
    s = fold(
        [
            at(1, ev.FaceTracked(track_id=3, cx=0.4, cy=0.5, width_frac=0.1)),
            at(2, ev.FaceTracked(track_id=3, cx=0.6, cy=0.5, width_frac=0.2)),
        ]
    )
    t = s.track(3)
    assert t is not None
    assert (t.cx, t.width_frac, t.last_seen, t.status) == (0.6, 0.2, 2, "tracking")
    assert s.present == () and s.people_count == 0  # 映っただけでは「在室」ではない
    assert s.ts == 2


def test_entering_makes_a_person_present_and_keeps_the_first_entry_time() -> None:
    s = fold(
        [
            at(1, ev.PersonEntered(track_id=1)),
            at(5, ev.PersonEntered(track_id=1, person_id="p1")),  # 重複の通知
        ]
    )
    t = s.track(1)
    assert t is not None and t.status == "present"
    assert (t.entered_at, t.person_id) == (1, "p1")
    assert s.people_count == 1


def test_identification_sets_the_person() -> None:
    s = fold([at(1, ev.PersonIdentified(track_id=2, person_id="p9", score=0.7))])
    t = s.track(2)
    assert t is not None and (t.person_id, t.id_score) == ("p9", 0.7)
    s = reduce(s, ev.PersonIdentified(track_id=2, person_id=None, score=0.1))
    t2 = s.track(2)
    assert t2 is not None and t2.person_id is None  # 同定が外れた


@pytest.mark.parametrize("leave", [ev.PersonLeft(track_id=1), ev.PersonPassedBy(track_id=1)])
def test_leaving_or_passing_by_removes_the_track(leave: Event) -> None:
    s = fold([ev.PersonEntered(track_id=1), ev.PersonEntered(track_id=2), leave])
    assert [t.track_id for t in s.tracks] == [2]
    assert fold([leave], s).track(1) is None  # 知らない人が去っても壊れない


def test_tracks_stay_sorted_by_id() -> None:
    s = fold(
        [ev.PersonEntered(track_id=5), ev.PersonEntered(track_id=2), ev.PersonEntered(track_id=9)]
    )
    assert [t.track_id for t in s.tracks] == [2, 5, 9]


# ---------------------------------------------------------------- アイコンタクト


def test_eye_contact_duration_and_idempotent_start() -> None:
    s = fold(
        [
            at(10, ev.EyeContactChanged(track_id=1, active=True)),
            at(11, ev.EyeContactChanged(track_id=1, active=True)),  # 重複でも見始めは 10 のまま
        ]
    )
    t = s.track(1)
    assert t is not None and t.eye_contact_since == 10
    assert t.eye_contact_duration(12.5) == 2.5
    assert [x.track_id for x in s.gazing(12.5, min_s=2.0)] == [1]
    assert s.gazing(12.5, min_s=3.0) == ()
    s = reduce(s, at(13, ev.EyeContactChanged(track_id=1, active=False)))
    t = s.track(1)
    assert t is not None and t.eye_contact is False and t.eye_contact_duration(14) == 0.0
    assert s.gazing() == ()


# ---------------------------------------------------------------- 会話


def test_user_speech_then_floor_decision() -> None:
    s = fold([at(1, ev.SpeechStarted(utt_id="u1", doa_deg=80.0))])
    c = s.conversation
    assert (c.user_speaking, c.utt_id, c.speech_started_at, c.doa_deg) == (True, "u1", 1, 80.0)
    s = reduce(s, at(3, ev.SpeechEnded(utt_id="u1", duration_s=2.0)))
    c = s.conversation
    assert (c.user_speaking, c.pending_utt_id, c.last_speech_ended) == (False, "u1", 3)
    assert c.doa_deg == 80.0  # 終わりに方向が無ければ、直前の値を残す
    s = reduce(s, ev.FloorDecided(utt_id="u1", decision="respond"))
    assert s.conversation.pending_utt_id is None
    assert s.conversation.last_decision == "respond"


def test_a_decision_for_an_older_utterance_does_not_clear_the_pending_one() -> None:
    s = fold(
        [
            ev.SpeechEnded(utt_id="u1", duration_s=1),
            ev.SpeechEnded(utt_id="u2", duration_s=1),
            ev.FloorDecided(utt_id="u1", decision="ignore"),
        ]
    )
    assert s.conversation.pending_utt_id == "u2"


def test_turn_lifecycle_and_robot_questions() -> None:
    s = fold(
        [
            at(1, ev.SentenceReceived(turn_id="t1", idx=0, text="こんにちは。")),
            at(2, ev.SentenceReceived(turn_id="t1", idx=1, text="お名前は？")),
        ]
    )
    assert s.conversation.active_turn_id == "t1"
    assert s.conversation.last_robot_question == 2
    s = reduce(s, at(3, ev.SentenceReceived(turn_id="t1", idx=2, text="よろしくね。")))
    assert s.conversation.last_robot_question == 2  # 質問でない文では更新しない
    s = reduce(s, at(4, ev.TurnEnded(turn_id="other")))
    assert s.conversation.active_turn_id == "t1"  # 別のターンの終わりは無視
    s = reduce(s, at(5, ev.TurnEnded(turn_id="t1")))
    assert (s.conversation.active_turn_id, s.conversation.last_turn_ended) == (None, 5)


def test_half_width_question_mark_counts() -> None:
    s = fold([at(1, ev.SentenceReceived(turn_id="t", idx=0, text="いい？ "))])
    assert s.conversation.last_robot_question == 1


def test_proactive_records_time_and_person() -> None:
    s = fold([at(7, ev.ProactiveDecided(intent="greet", person_id="p1"))])
    assert (s.conversation.last_proactive, s.conversation.last_proactive_person_id) == (7, "p1")


# ---------------------------------------------------------------- ロボット


def test_robot_speech_barge_in_and_connection() -> None:
    s = fold([at(1, ev.RobotSpeechStarted(turn_id="t1"))])
    r = s.robot
    assert (r.speaking, r.speaking_turn_id, r.speaking_since) == (True, "t1", 1)
    s = reduce(s, at(2, ev.BargeIn(turn_id="t1")))
    assert s.robot.last_barge_in == 2
    s = reduce(s, at(3, ev.RobotSpeechEnded(turn_id="t1")))
    r = s.robot
    assert (r.speaking, r.speaking_turn_id, r.last_spoke_end) == (False, None, 3)
    assert fold([ev.ServerConnectionChanged(connected=True)]).robot.server_connected is True


# ---------------------------------------------------------------- 性質


def test_reduce_is_pure_and_snapshots_do_not_change() -> None:
    before = fold([at(1, ev.PersonEntered(track_id=1))])
    snapshot_tracks = before.tracks
    after = reduce(before, at(2, ev.EyeContactChanged(track_id=1, active=True)))
    assert before.tracks is snapshot_tracks
    t_before = before.track(1)
    assert (
        t_before is not None and t_before.eye_contact is False
    )  # 古いスナップショットは変わらない
    assert after is not before
    with pytest.raises(dataclasses.FrozenInstanceError):
        before.ts = 9.0  # pyright: ignore[reportAttributeAccessIssue]


def test_unknown_events_only_advance_the_time_and_time_never_goes_back() -> None:
    s = fold([at(5, ev.AudioChunkReceived(turn_id="t", seq=0, nbytes=1))])
    assert s.ts == 5 and s.tracks == ()
    assert reduce(s, at(3, ev.AudioChunkReceived(turn_id="t", seq=1, nbytes=1))).ts == 5
    assert reduce(s, ev.AudioChunkReceived(turn_id="t", seq=2, nbytes=1)).ts == 5  # ts なし


def test_same_events_give_the_same_state() -> None:
    events = [
        at(1, ev.PersonEntered(track_id=1)),
        at(2, ev.EyeContactChanged(track_id=1, active=True)),
        at(3, ev.SpeechStarted(utt_id="u", doa_deg=1.0)),
    ]
    assert fold(events) == fold(events)


# ---------------------------------------------------------------- World（バス結線）


async def test_world_follows_the_bus() -> None:
    clock = ManualClock(100.0)
    bus = EventBus(clock)
    world = World()
    world.attach(bus)
    bus.publish(ev.PersonEntered(track_id=1))
    clock.advance(2.0)
    bus.publish(ev.EyeContactChanged(track_id=1, active=True))
    clock.advance(1.0)
    bus.publish(ev.SpeechStarted(utt_id="u1"))
    await bus.drain(1)
    snap = world.snapshot()
    assert snap.ts == 103.0
    assert snap.people_count == 1
    t = snap.track(1)
    assert t is not None and t.eye_contact_since == 102.0
    assert snap.conversation.user_speaking
    old = snap
    bus.publish(ev.SpeechEnded(utt_id="u1", duration_s=1))
    await bus.drain(1)
    assert old.conversation.user_speaking  # 取り出したスナップショットは変わらない
    assert not world.snapshot().conversation.user_speaking
    await bus.close()
