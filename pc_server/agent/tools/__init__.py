"""エージェントのツール群。ここに @tool を追加していくのが機能拡張の中心。"""

from __future__ import annotations

from .memory_tool import PersonContext, current_person_var, remember_fact, set_person_name
from .motion_tool import express_motion, motion_sink_var
from .time_tool import get_current_time
from .weather_tool import get_weather

# エージェントにバインドするツールの一覧。
ALL_TOOLS = [
    get_current_time,
    get_weather,
    express_motion,
    set_person_name,
    remember_fact,
]

__all__ = [
    "ALL_TOOLS",
    "PersonContext",
    "current_person_var",
    "express_motion",
    "get_current_time",
    "get_weather",
    "motion_sink_var",
    "remember_fact",
    "set_person_name",
]
