"""LangGraph エージェント（ReAct 型のツールループ）を明示的な StateGraph で組む。

    START → agent ⇄ tools → END

ターンごとの文脈は状態の ``turn_context`` に入れ、agent ノードがシステムメッセージの末尾に付ける。
履歴（messages）にはユーザー発話・ツール・応答だけが残り、文脈は毎ターン最新のものに差し替わる。
コンテキスト構築・ガードのノードの追加は #63。

（旧実装の ``create_react_agent`` は LangGraph v1.0 で非推奨になったため使わない）
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .prompt import PERSONA
from .tools import ALL_TOOLS


class ReachyAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    turn_context: str


def build_prompt(state: ReachyAgentState) -> list[BaseMessage]:
    system = PERSONA
    if state.get("turn_context"):
        system = f"{PERSONA}\n{state['turn_context']}"
    return [SystemMessage(system), *state["messages"]]


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool] = ALL_TOOLS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any]:
    bound = model.bind_tools(list(tools))

    async def agent(state: ReachyAgentState) -> dict[str, list[BaseMessage]]:
        # ainvoke でも、stream_mode="messages" で呼ばれればトークンはストリーミングされる
        return {"messages": [await bound.ainvoke(build_prompt(state))]}

    graph = StateGraph(ReachyAgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
