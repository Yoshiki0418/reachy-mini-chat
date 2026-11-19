"""LangGraph エージェント (ReAct 型ツールループ)。

`create_react_agent` は「agent ノード + ToolNode + tools_condition」の標準 ReAct
パターンをそのまま組んだ prebuilt。会話履歴は MemorySaver チェックポインタに
thread_id 単位で保持する（セッション = 1 スレッド）。
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from .prompt import SYSTEM_PROMPT
from .tools import ALL_TOOLS

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "200"))
DEFAULT_TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.7"))


def build_agent(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    checkpointer: MemorySaver | None = None,
) -> CompiledStateGraph:
    """ツール付き会話エージェントをビルドして返す。"""
    llm = ChatOpenAI(model=model, temperature=temperature, max_completion_tokens=max_tokens)
    return create_react_agent(
        llm,
        ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or MemorySaver(),
    )
