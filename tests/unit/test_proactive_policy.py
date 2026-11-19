"""ロボット側の自発発話ポリシー。タイマー式ではなく「持続アイコンタクト」でのみ声をかける。"""

from __future__ import annotations

import conversation_ws as cw  # robot/ は conftest で sys.path に入っている
import pytest


class FakeGaze:
    def __init__(self) -> None:
        self.contact_since: float | None = None  # アイコンタクト開始時刻（None = 目が合っていない）

    def eye_contact(self, min_duration: float = 0.0) -> bool:
        if self.contact_since is None:
            return False
        return (cw.time.monotonic() - self.contact_since) >= min_duration


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(cw.time, "monotonic", lambda: t["now"])
    return t


def test_no_gaze_never_speaks(clock) -> None:
    assert cw.ProactivePolicy(None).decide() is None


def test_no_eye_contact_never_speaks(clock) -> None:
    """人がいても目が合っていなければ、どれだけ時間が経っても話しかけない（タイマー禁止）。"""
    policy = cw.ProactivePolicy(FakeGaze())
    for _ in range(100):
        clock["now"] += 60
        assert policy.decide() is None


def test_sustained_eye_contact_engages_once(clock) -> None:
    gaze = FakeGaze()
    policy = cw.ProactivePolicy(gaze)
    gaze.contact_since = clock["now"]
    clock["now"] += cw.ENGAGE_EYE_SEC - 0.1
    assert policy.decide() is None  # まだ短い
    clock["now"] += 0.2
    assert policy.decide() == "engage"
    policy.note_proactive("engage")
    clock["now"] += cw.PROACTIVE_MIN_GAP + 10
    assert policy.decide() is None  # 同じアイコンタクトの間は再発火しない


def test_reengages_after_looking_away_and_gap(clock) -> None:
    gaze = FakeGaze()
    policy = cw.ProactivePolicy(gaze)
    gaze.contact_since = clock["now"]
    clock["now"] += cw.ENGAGE_EYE_SEC
    assert policy.decide() == "engage"
    policy.note_proactive("engage")

    gaze.contact_since = None  # 目を逸らす → リセット
    assert policy.decide() is None

    clock["now"] += 1
    gaze.contact_since = clock["now"]
    clock["now"] += cw.ENGAGE_EYE_SEC
    assert policy.decide() is None  # MIN_GAP 内なのでまだ話さない

    clock["now"] += cw.PROACTIVE_MIN_GAP
    assert policy.decide() == "engage"


def test_recent_turn_suppresses_engage(clock) -> None:
    gaze = FakeGaze()
    policy = cw.ProactivePolicy(gaze)
    policy.note_turn()  # 直前に会話した
    gaze.contact_since = None
    policy.decide()  # 目を逸らして engaged リセット
    gaze.contact_since = clock["now"]
    clock["now"] += cw.ENGAGE_EYE_SEC
    assert policy.decide() is None
