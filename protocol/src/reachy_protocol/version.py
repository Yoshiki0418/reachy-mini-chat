"""プロトコルの版と互換判定。

版は semver（MAJOR.MINOR.PATCH）。
- MAJOR: 互換性のない変更（必須フィールドの追加・意味の変更・メッセージの削除）
- MINOR: 後方互換な追加（任意フィールドやメッセージの追加）。受信側は未知のフィールドを無視する
- PATCH: ドキュメントや検証の修正

メジャー版が一致すれば接続してよい。
"""

from __future__ import annotations

import re
from typing import NamedTuple

PROTOCOL_VERSION = "1.0.0"

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class Version(NamedTuple):
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def parse_version(text: str) -> Version:
    """版の文字列（例: 1.2.3）を Version にする。形式が不正なら ValueError。"""
    m = _SEMVER.match(text.strip())
    if m is None:
        raise ValueError(f"invalid protocol version: {text!r}")
    return Version(int(m[1]), int(m[2]), int(m[3]))


def is_compatible(peer: str, local: str = PROTOCOL_VERSION) -> bool:
    """相手の版と接続してよいか（メジャー版が同じなら True）。不正な版は False。"""
    try:
        return parse_version(peer).major == parse_version(local).major
    except ValueError:
        return False
