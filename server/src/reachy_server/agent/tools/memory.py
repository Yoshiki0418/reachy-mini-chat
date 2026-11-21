"""記憶ツール。人物の記憶の正はロボット側にあり、ここは操作の発行だけ（設計書 §2.4）。

ツールは memory.op をロボットへ送り、ロボットの検証結果（memory.ack）を待って、LLM に
「覚えた / 覚えられなかった」を返す。**覚えられなかったこと（rejected・timeout）は、
覚えた体で話さない**よう、結果の文面で LLM に伝える。履歴（チェックポイント）にもこの結果が
残るので、以後のターンでも同じ前提になる。名前や事実の正本は、ロボットが毎ターン送る
``context.person``（拒否されたものはそこに入らない）。
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

from langchain_core.tools import tool

from ...responder import MemoryOpName, MemoryPort, MemoryResult

MAX_NAME_CHARS = 32  # PersonProfile.name の上限
MAX_FACT_CHARS = 200  # PersonProfile.facts の 1 件の上限
MAX_OPS_PER_TURN = 3  # 1 ターンに発行できる記憶操作の数（ツールの暴走・乱発の安全弁）


@dataclass
class MemoryContext:
    """ターンごとの記憶操作の口。AgentResponder がターンの始めに置く。"""

    person_id: str | None
    port: MemoryPort | None
    issued: int = 0


memory_ctx_var: contextvars.ContextVar[MemoryContext | None] = contextvars.ContextVar(
    "memory_ctx_var", default=None
)

_NOT_REMEMBERED = "覚えていない。「覚えたよ」と言わず、覚えたことにして話さない。"


def describe(result: MemoryResult, done: str) -> str:
    """LLM に返す結果の文面。"""
    if result.result == "applied":
        return f"{done}（記録は済んだ。この操作そのものには触れず、自然に会話を続ける）"
    if result.result == "rejected":
        why = f"（理由: {result.reason}）" if result.reason else ""
        return f"記録できなかった{why}。{_NOT_REMEMBERED}別の言い方で聞き直してもよい。"
    return f"記録できたか確認できなかった。{_NOT_REMEMBERED}"


async def _issue(op: MemoryOpName, value: str | None, done: str) -> str:
    ctx = memory_ctx_var.get()
    if ctx is None or ctx.port is None:
        return f"この場では記憶の機能が使えない。{_NOT_REMEMBERED}"
    if ctx.person_id is None:
        return f"相手が誰か分からないので記録できない。{_NOT_REMEMBERED}"
    if ctx.issued >= MAX_OPS_PER_TURN:
        return f"一度に記録できる数を超えた。{_NOT_REMEMBERED}"
    ctx.issued += 1
    return describe(await ctx.port(ctx.person_id, op, value), done)


@tool
async def set_person_name(name: str) -> str:
    """相手が自分の名前を教えてくれたとき（または名前を訂正したとき）に、その名前を覚えます。

    相手が自分で名乗った名前だけを、呼び名として渡すこと。推測や、他の人の名前は渡さない。

    Args:
        name: 相手が呼んでほしい名前（例: 山田、ゆうき）。敬称は付けない
    """
    name = name.strip()
    if not name or len(name) > MAX_NAME_CHARS:
        return f"名前は 1〜{MAX_NAME_CHARS} 文字で。{_NOT_REMEMBERED}聞き直してよい。"
    return await _issue("set_name", name, f"名前を「{name}」と記録した。")


@tool
async def remember_fact(fact: str) -> str:
    """相手について、次に会ったときの話題になる事柄を 1 つ覚えます（好きなもの、取り組んでいることなど）。

    本人が自分から話したことだけを、短い 1 文で。住所・連絡先・健康・家族など、
    機微な個人情報は覚えない。些細な相づちや、その場限りのことも覚えない。

    Args:
        fact: 覚える内容（例: 「コーヒーが好き」「卒論で画像認識を扱っている」）
    """
    fact = fact.strip()
    if not fact or len(fact) > MAX_FACT_CHARS:
        return f"内容は 1〜{MAX_FACT_CHARS} 文字で。{_NOT_REMEMBERED}"
    return await _issue("add_fact", fact, "記録した。")


@tool
async def forget_me() -> str:
    """相手が「忘れて」「覚えないで」と**はっきり頼んだとき**に、その人の記憶をすべて消します。

    元に戻せない。曖昧な場合は使わず、本当に忘れてよいか確かめてから使うこと。
    """
    return await _issue("forget", None, "その人の記憶をすべて消した。")
