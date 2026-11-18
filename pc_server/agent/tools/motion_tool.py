"""モーション/表情ツール (reachy 用)。

gakucho-ai-motion の motion_tool.py の「contextvar サイドチャネル」方式を踏襲する:
- LLM にはステータス文字列だけを返す（トークン節約・本文で動作に言及させない）
- 実際のモーション指示は per-session の sink (list) に積み、WebSocket 経由でロボットへ送る

gakucho は外部 T2M サーバー→BVH だが、reachy は名前付きプリセットを送り、
ロボット側 Motion Executor が goto_target(head/antennas/body_yaw) に変換する。
"""

from __future__ import annotations

import contextvars
from typing import Literal

from langchain_core.tools import tool

# ws ハンドラが各ターンの先頭で「空 list」をセットし、ツールがここに append する。
motion_sink_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "motion_sink_var", default=None
)

# ロボット側 Motion Executor が解釈するプリセット名。
MotionName = Literal[
    "nod",  # うなずく（同意・相槌）
    "shake",  # 首を横に振る（否定）
    "tilt",  # 首をかしげる（疑問・興味）
    "look_away",  # 照れて顔を逸らす
    "happy",  # 嬉しそうにアンテナを動かす
    "greet",  # 挨拶（軽く頭を動かす）
    "think",  # 考えるしぐさ
]


@tool
def express_motion(motion: MotionName) -> str:
    """会話の流れに合わせてロボットの身体表現（首やアンテナの動き）を再生します。

    自然なタイミングでのみ使ってください（毎回は不要）。動作後の返答で、
    この動作自体には言及しないこと。

    使いどころの例:
    - nod: 相手に同意・相槌するとき
    - shake: 否定するとき
    - tilt: 疑問や興味を示すとき
    - look_away: 照れたとき・恥ずかしいとき（自分から顔を逸らす）
    - happy: 嬉しい・楽しい気持ちを表すとき
    - greet: 挨拶するとき
    - think: 考えているとき

    Args:
        motion: 再生するモーションのプリセット名
    """
    sink = motion_sink_var.get()
    if sink is not None:
        sink.append({"type": "motion", "name": motion})
    return f"(motion '{motion}' queued; do not mention this action in your reply)"
