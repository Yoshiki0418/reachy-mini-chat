"""エージェントのツール。記憶ツール（memory.op の発行）は #22、ロボット側ツールの RPC は #23。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .motion import express_motion, motion_sink_var
from .time import get_current_time
from .weather import get_weather

ALL_TOOLS: list[BaseTool] = [get_current_time, get_weather, express_motion]

__all__ = ["ALL_TOOLS", "express_motion", "get_current_time", "get_weather", "motion_sink_var"]
