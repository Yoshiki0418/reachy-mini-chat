"""ロボット側の設定。既存の環境変数名・初期値との互換を確かめる。"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from reachy_front.core import FrontSettings

REPO = Path(__file__).resolve().parents[2]
LEGACY = [REPO / "robot" / "gaze.py", REPO / "robot" / "conversation_ws.py"]

# 旧実装の（環境変数名, 初期値）→ 新しい設定での場所。値の意味（数・真偽）で比べる
MAPPING: dict[str, tuple[str, str]] = {
    "EC_FRONTAL_MAX": ("eye_contact", "frontal_max"),
    "EC_LEVEL_MAX": ("eye_contact", "level_max"),
    "EC_FACE_MIN_FRAC": ("eye_contact", "face_min_frac"),
    "EC_CENTER_MAX": ("eye_contact", "center_max"),
    "EC_VERT_MIN": ("eye_contact", "vert_min"),
    "EC_VERT_MAX": ("eye_contact", "vert_max"),
    "EC_GRACE": ("eye_contact", "grace_s"),
    "PROACTIVE_ENABLED": ("proactive", "enabled"),
    "PROACTIVE_ENGAGE_SEC": ("proactive", "engage_s"),
    "PROACTIVE_MIN_GAP": ("proactive", "min_gap_s"),
    "PROACTIVE_POLL_SEC": ("proactive", "poll_s"),
    "GATE_EYE_SEC": ("gate", "eye_s"),
    "GAZE_ENABLED": ("gaze", "enabled"),
    "GAZE_KP_YAW": ("gaze", "kp_yaw"),
    "GAZE_KP_PITCH": ("gaze", "kp_pitch"),
    "GAZE_YAW_SIGN": ("gaze", "yaw_sign"),
    "GAZE_PITCH_SIGN": ("gaze", "pitch_sign"),
    "GAZE_DEBUG": ("gaze", "debug"),
    "PC_HOST": ("server", "host"),
    "PC_PORT": ("server", "port"),
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [*MAPPING, "REACHY_SERVER_TOKEN"]:
        monkeypatch.delenv(name, raising=False)


def field(s: FrontSettings, where: tuple[str, str]) -> object:
    return getattr(getattr(s, where[0]), where[1])


def test_defaults_match_the_legacy_robot_code() -> None:
    """旧実装（robot/）のソースから `os.environ.get("名前", "初期値")` を拾い、初期値が同じか確かめる。"""
    legacy_files = [p for p in LEGACY if p.exists()]
    if not legacy_files:
        pytest.skip("旧実装を削除済み。この比較は役目を終えた")
    pairs: dict[str, str] = {}
    for path in legacy_files:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r'os\.environ\.get\(\s*"([A-Z_]+)",\s*"([^"]*)"\s*\)', text):
            pairs[m[1]] = m[2]
    s = FrontSettings()
    checked = 0
    for name, default in pairs.items():
        if name not in MAPPING:
            continue
        value = field(s, MAPPING[name])
        if isinstance(value, bool):
            assert value == (default == "1"), name
        elif isinstance(value, int | float):
            assert value == pytest.approx(float(default)), name
        else:
            assert value == default, name
        checked += 1
    assert checked >= 15  # 旧実装を読めていること


def test_every_mapped_name_is_an_env_name_of_a_field() -> None:
    s = FrontSettings()
    for name, where in MAPPING.items():
        model = getattr(s, where[0])
        assert type(model).model_fields[where[1]].alias == name


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EC_GRACE", "0.9")
    monkeypatch.setenv("PROACTIVE_ENABLED", "0")
    monkeypatch.setenv("GATE_EYE_SEC", "1.25")
    monkeypatch.setenv("GAZE_YAW_SIGN", "1")
    monkeypatch.setenv("PC_HOST", "pc.local")
    monkeypatch.setenv("PC_PORT", "9000")
    s = FrontSettings()
    assert s.eye_contact.grace_s == 0.9
    assert s.proactive.enabled is False
    assert s.gate.eye_s == 1.25
    assert s.gaze.yaw_sign == 1.0
    assert s.server.session_url == "ws://pc.local:9000/v1/session"
    assert s.server.http_url == "http://pc.local:9000"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PC_PORT", "0"),
        ("PC_PORT", "70000"),
        ("EC_FACE_MIN_FRAC", "2"),
        ("PROACTIVE_ENGAGE_SEC", "0"),
        ("PROACTIVE_MIN_GAP", "-1"),
        ("REACHY_SERVER_TOKEN", "short"),
        ("GAZE_ENABLED", "maybe"),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        FrontSettings()


def test_token_is_optional_and_never_dumped(monkeypatch: pytest.MonkeyPatch) -> None:
    assert FrontSettings().dump()["server"]["token"] is None
    secret = "s3cret-token-0123456789"
    monkeypatch.setenv("REACHY_SERVER_TOKEN", secret)
    s = FrontSettings()
    assert s.server.token is not None and s.server.token.get_secret_value() == secret
    dumped = json.dumps(s.dump())
    assert secret not in dumped and "**********" in dumped
    assert secret not in repr(s)
