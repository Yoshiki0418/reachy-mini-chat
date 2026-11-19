"""応答テキストの文分割。"""

from __future__ import annotations

import pytest

from reachy_server.text import split_completed


@pytest.mark.parametrize(
    ("buffer", "sentences", "rest"),
    [
        ("", [], ""),
        ("こんにちは", [], "こんにちは"),
        ("おはよう。元気？今日は", ["おはよう。", "元気？"], "今日は"),
        ("えっ！？本当に", ["えっ！？"], "本当に"),  # 連続した区切りは 1 文
        ("一行目\n二行目\n", ["一行目", "二行目"], ""),
        ("。 。", ["。", "。"], ""),
        ("Hi! OK.", ["Hi!", "OK."], ""),
    ],
)
def test_split_completed(buffer: str, sentences: list[str], rest: str) -> None:
    assert split_completed(buffer) == (sentences, rest)
