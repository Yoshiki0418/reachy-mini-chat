"""人物メモリ用ツール。

express_motion と同じ ContextVar 方式で「今話している相手」をツールへ渡す。
pipeline がターンごとに current_person_var を設定し、ツールはそれを見て
名前や事実を PersonStore に保存する。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from pc_server.person.store import PersonStore


@dataclass
class PersonContext:
    store: PersonStore
    person_id: str


# ターン中の「今話している相手」。認識できない時は None。
current_person_var: ContextVar[PersonContext | None] = ContextVar("current_person", default=None)


# 名前として妥当でない（文っぽい・長すぎる）文字列を弾く。STT 幻聴でのDB汚染防止。
_NAME_STOP_CHARS = "。、．，!！?？…\n　 「」『』"


def _looks_like_name(name: str) -> bool:
    name = name.strip()
    if not (1 <= len(name) <= 8):
        return False
    if any(c in name for c in _NAME_STOP_CHARS):
        return False
    # 「〜です」「〜ます」など文末表現を含むものは名前ではないとみなす
    return not any(
        name.endswith(s) for s in ("です", "ます", "ました", "だよ", "でしょう", "ください")
    )


@tool
def set_person_name(name: str) -> str:
    """今話している相手の名前を覚える。相手が自分の名前をはっきり教えてくれたときだけ使う。"""
    ctx = current_person_var.get()
    if ctx is None:
        return "（今は相手を認識できていないので名前を保存できませんでした）"
    if not _looks_like_name(name):
        return "（名前として確からしくないので保存を見送りました）"
    ctx.store.set_name(ctx.person_id, name.strip())
    return f"名前「{name.strip()}」を覚えました。"


@tool
def remember_fact(fact: str) -> str:
    """今話している相手に関する情報（好み・出来事・所属など）を1つ覚える。

    短い平叙文で渡すこと（例: 「コーヒーが好き」「猫を飼っている」）。
    """
    ctx = current_person_var.get()
    if ctx is None:
        return "（今は相手を認識できていないので保存できませんでした）"
    ctx.store.add_fact(ctx.person_id, fact)
    return "覚えました。"
