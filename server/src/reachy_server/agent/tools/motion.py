"""身体表現ツール。

LLM にはステータス文字列だけを返し（本文で動作に言及させない）、実際の指示は
ターンごとのシンク（ContextVar）に積む。AgentResponder がそれを ReplyMotion として送る。
"""

from __future__ import annotations

import contextvars
from typing import Literal

from langchain_core.tools import tool

motion_sink_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "motion_sink_var", default=None
)

# ロボット側 Motion Executor が解釈するプリセット名（robot/motion.py）
MotionName = Literal["nod", "shake", "tilt", "look_away", "happy", "greet", "think"]


@tool
def express_motion(motion: MotionName) -> str:
    """会話の流れに合わせてロボットの身体表現（首やアンテナの動き）を再生します。

    自然なタイミングでのみ使ってください（毎回は不要）。動作後の返答で、
    この動作自体には言及しないこと。

    - nod: 同意・相槌 / shake: 否定 / tilt: 疑問や興味 / look_away: 照れたとき
    - happy: 嬉しいとき / greet: 挨拶 / think: 考えているとき

    Args:
        motion: 再生するモーションのプリセット名
    """
    sink = motion_sink_var.get()
    if sink is not None:
        sink.append(motion)
    return f"(motion '{motion}' queued; do not mention this action in your reply)"
