"""イベントを状態に畳み込む純粋関数。副作用も時計の参照もなく、同じ列なら同じ結果になる。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from ..core import events as ev
from ..core.events import Event
from .state import TrackState, WorldState

_QUESTION_MARKS = ("？", "?")


def reduce(state: WorldState, event: Event) -> WorldState:
    """``event`` を ``state`` に畳み込んで、新しい状態を返す（知らないイベントは時刻だけ進める）。"""
    ts = event.ts if event.ts is not None else state.ts
    state = replace(state, ts=max(state.ts, ts))

    match event:
        case ev.FaceTracked():
            return _track(
                state,
                event.track_id,
                lambda t: replace(
                    t,
                    cx=event.cx,
                    cy=event.cy,
                    width_frac=event.width_frac,
                    last_seen=ts,
                ),
            )
        case ev.PersonIdentified():
            return _track(
                state,
                event.track_id,
                lambda t: replace(t, person_id=event.person_id, id_score=event.score),
            )
        case ev.PersonEntered():
            return _track(
                state,
                event.track_id,
                lambda t: replace(
                    t,
                    status="present",
                    entered_at=t.entered_at if t.entered_at is not None else ts,
                    person_id=event.person_id if event.person_id is not None else t.person_id,
                ),
            )
        case ev.PersonPassedBy() | ev.PersonLeft():
            return _remove_track(state, event.track_id)
        case ev.EyeContactChanged():
            return _track(state, event.track_id, lambda t: _eye_contact(t, event.active, ts))
        case ev.SpeechStarted():
            return _conv(
                state,
                user_speaking=True,
                utt_id=event.utt_id,
                speech_started_at=ts,
                doa_deg=event.doa_deg,
            )
        case ev.SpeechEnded():
            return _conv(
                state,
                user_speaking=False,
                utt_id=event.utt_id,
                last_speech_ended=ts,
                pending_utt_id=event.utt_id,
                doa_deg=event.doa_deg if event.doa_deg is not None else state.conversation.doa_deg,
            )
        case ev.FloorDecided():
            pending = state.conversation.pending_utt_id
            return _conv(
                state,
                last_decision=event.decision,
                pending_utt_id=None if pending == event.utt_id else pending,
            )
        case ev.ProactiveDecided():
            return _conv(state, last_proactive=ts, last_proactive_person_id=event.person_id)
        case ev.SentenceReceived():
            return _conv(
                state,
                active_turn_id=event.turn_id,
                last_robot_question=(
                    ts if event.text.rstrip().endswith(_QUESTION_MARKS) else _KEEP
                ),
            )
        case ev.TurnEnded():
            if state.conversation.active_turn_id != event.turn_id:
                return state
            return _conv(state, active_turn_id=None, last_turn_ended=ts)
        case ev.RobotSpeechStarted():
            return _robot(state, speaking=True, speaking_turn_id=event.turn_id, speaking_since=ts)
        case ev.RobotSpeechEnded():
            return _robot(
                state,
                speaking=False,
                speaking_turn_id=None,
                speaking_since=None,
                last_spoke_end=ts,
            )
        case ev.BargeIn():
            return _robot(state, last_barge_in=ts)
        case ev.ServerConnectionChanged():
            return _robot(state, server_connected=event.connected)
        case _:
            return state


class _Keep:
    """``_conv`` に「この項目は変えない」と伝える印（None に戻す指定と区別するため）。"""


_KEEP = _Keep()


def _eye_contact(t: TrackState, active: bool, ts: float) -> TrackState:
    if active:
        # すでに見ている最中なら、見始めた時刻は変えない（通知の重複で継続時間が縮まらないように）
        since = t.eye_contact_since if t.eye_contact and t.eye_contact_since is not None else ts
        return replace(t, eye_contact=True, eye_contact_since=since)
    return replace(t, eye_contact=False, eye_contact_since=None)


def _track(
    state: WorldState, track_id: int, update: Callable[[TrackState], TrackState]
) -> WorldState:
    current = state.track(track_id)
    if current is None:
        current = TrackState(track_id=track_id, last_seen=state.ts)
        rest = state.tracks
    else:
        rest = tuple(t for t in state.tracks if t.track_id != track_id)
    updated = update(current)
    tracks = tuple(sorted((*rest, updated), key=lambda t: t.track_id))
    return replace(state, tracks=tracks)


def _remove_track(state: WorldState, track_id: int) -> WorldState:
    return replace(state, tracks=tuple(t for t in state.tracks if t.track_id != track_id))


def _conv(state: WorldState, **changes: object) -> WorldState:
    clean = {k: v for k, v in changes.items() if v is not _KEEP}
    return replace(state, conversation=replace(state.conversation, **clean))  # pyright: ignore[reportArgumentType]


def _robot(state: WorldState, **changes: object) -> WorldState:
    return replace(state, robot=replace(state.robot, **changes))  # pyright: ignore[reportArgumentType]


__all__ = ["reduce"]
