"""ロボット側で実行するツール（RPC）。設計書 §4.2。

データ（今日の来訪状況・覚えている人）はロボットにあるので、サーバーのツールは ``tool.call`` で
ロボットに問い合わせ、``tool.result`` を待つ（待ち時間は 1 秒）。答えが得られなかったとき
（エラー・時間切れ）は、推測で答えず「分からない」と正直に伝えるよう、結果の文面で LLM に伝える。
ロボットが何を返すか（誰の名前まで出すか）はロボット側が決める。ここは受け取った結果を渡すだけ。
"""

from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass

from langchain_core.tools import tool
from pydantic import JsonValue

from ...responder import ToolOutcome, ToolPort

MAX_RESULT_CHARS = 2000  # LLM に渡す結果の上限（長すぎる結果で文脈を埋めない）
MAX_CALLS_PER_TURN = 3  # 1 ターンに引ける回数（暴走・乱発の安全弁）

_DONT_GUESS = "推測で答えず、分からないと正直に伝える。"


@dataclass
class RobotToolContext:
    """ターンごとのロボット側ツールの口。AgentResponder がターンの始めに置く。"""

    port: ToolPort | None
    issued: int = 0


robot_tool_ctx_var: contextvars.ContextVar[RobotToolContext | None] = contextvars.ContextVar(
    "robot_tool_ctx_var", default=None
)


def describe(outcome: ToolOutcome) -> str:
    """LLM に返す結果の文面。"""
    if outcome.status == "ok":
        text = json.dumps(outcome.result, ensure_ascii=False)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "…（以下省略）"
        return f"{text}\n（ロボットが持っているデータ。そのまま読み上げず、会話に合わせて短くまとめる）"
    if outcome.status == "error":
        return f"取得できなかった（{outcome.error}）。{_DONT_GUESS}"
    return f"ロボットから時間内に答えが得られなかった。{_DONT_GUESS}"


async def call_robot_tool(name: str, args: dict[str, JsonValue]) -> str:
    ctx = robot_tool_ctx_var.get()
    if ctx is None or ctx.port is None:
        return f"この場では使えない。{_DONT_GUESS}"
    if ctx.issued >= MAX_CALLS_PER_TURN:
        return f"一度に調べられる回数を超えた。{_DONT_GUESS}"
    ctx.issued += 1
    return describe(await ctx.port(name, args))


@tool
async def lab_presence() -> str:
    """いま・今日、研究室に誰が来ているか（来訪状況）を調べます。

    「今日誰が来てる？」「〇〇さんは来てる？」のように聞かれたときに使います。
    """
    return await call_robot_tool("lab_presence", {})


@tool
async def recall_person(name: str) -> str:
    """覚えている人について、ロボットの記憶から情報を引きます（最後に来た日など）。

    相手以外の人について聞かれたときに使います。分からなければ、分からないと答えます。

    Args:
        name: 調べたい人の名前
    """
    name = name.strip()
    if not name:
        return f"名前が空。{_DONT_GUESS}"
    return await call_robot_tool("recall_person", {"name": name})
