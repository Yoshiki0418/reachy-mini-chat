"""エージェントのツール。ロボット側ツールの RPC は #23。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .memory import forget_me, memory_ctx_var, remember_fact, set_person_name
from .motion import express_motion, motion_sink_var
from .time import get_current_time
from .weather import get_weather

ALL_TOOLS: list[BaseTool] = [
    get_current_time,
    get_weather,
    express_motion,
    set_person_name,
    remember_fact,
    forget_me,
]

__all__ = [
    "ALL_TOOLS",
    "express_motion",
    "forget_me",
    "get_current_time",
    "get_weather",
    "memory_ctx_var",
    "motion_sink_var",
    "remember_fact",
    "set_person_name",
]
