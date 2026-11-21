"""エージェントのツール。サーバーで実行するもの、ロボットへ操作を発行するもの（記憶）、ロボットへ RPC するもの。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .memory import forget_me, memory_ctx_var, remember_fact, set_person_name
from .motion import express_motion, motion_sink_var
from .robot import lab_presence, recall_person, robot_tool_ctx_var
from .time import get_current_time
from .weather import get_weather

ALL_TOOLS: list[BaseTool] = [
    get_current_time,
    get_weather,
    express_motion,
    set_person_name,
    remember_fact,
    forget_me,
    lab_presence,
    recall_person,
]

__all__ = [
    "ALL_TOOLS",
    "express_motion",
    "forget_me",
    "get_current_time",
    "get_weather",
    "lab_presence",
    "memory_ctx_var",
    "motion_sink_var",
    "recall_person",
    "remember_fact",
    "robot_tool_ctx_var",
    "set_person_name",
]
