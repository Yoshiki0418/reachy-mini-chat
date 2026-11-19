import pytest

from reachy_protocol import PROTOCOL_VERSION, is_compatible, parse_version
from reachy_protocol.version import Version


def test_current_version_is_semver() -> None:
    assert str(parse_version(PROTOCOL_VERSION)) == PROTOCOL_VERSION


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1.0.0", Version(1, 0, 0)), ("12.3.45", Version(12, 3, 45)), (" 2.0.1 ", Version(2, 0, 1))],
)
def test_parse_version(text: str, expected: Version) -> None:
    assert parse_version(text) == expected


@pytest.mark.parametrize(
    "text", ["", "1", "1.0", "1.0.0.0", "v1.0.0", "01.0.0", "1.0.x", "1.0.0-rc1"]
)
def test_parse_version_rejects_invalid(text: str) -> None:
    with pytest.raises(ValueError):
        parse_version(text)


@pytest.mark.parametrize(
    ("peer", "local", "ok"),
    [
        ("1.0.0", "1.0.0", True),
        ("1.4.2", "1.0.0", True),  # マイナー違いは互換（未知フィールドは無視される）
        ("1.0.0", "1.9.9", True),
        ("2.0.0", "1.0.0", False),  # メジャー違いは非互換
        ("0.9.0", "1.0.0", False),
        ("garbage", "1.0.0", False),  # 不正な版は接続させない
    ],
)
def test_is_compatible(peer: str, local: str, ok: bool) -> None:
    assert is_compatible(peer, local) is ok


def test_is_compatible_defaults_to_current_version() -> None:
    assert is_compatible(PROTOCOL_VERSION)
