"""現在時刻ツール (LangChain @tool)。"""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """現在の日時を「YYYY年MM月DD日 HH時MM分SS秒」形式で返します。
    「今何時？」「今日は何日？」など時刻・日付を尋ねられたときに使います。
    """
    return datetime.now().strftime("%Y年%m月%d日 %H時%M分%S秒")
