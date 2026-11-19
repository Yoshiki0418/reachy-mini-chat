"""LangGraph エージェント（ReAct 型のツールループ）を明示的な StateGraph で組む。

    START → agent ⇄ tools → END

ターンごとの文脈（相手・状況）は実行時の設定 ``configurable.turn_context`` で渡し、agent ノードが
システムメッセージの末尾に付ける。状態には入れないので、**履歴（チェックポイント）には保存されない**。
履歴に残るのはユーザー発話・ツール・応答だけ（人物情報の正本はロボット側。設計書 §2.4）。
コンテキスト構築・ガードのノードの追加は #63。

（旧実装の ``create_react_agent`` は LangGraph v1.0 で非推奨になったため使わない）
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
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


def build_prompt(state: ReachyAgentState, turn_context: str = "") -> list[BaseMessage]:
    system = f"{PERSONA}\n{turn_context}" if turn_context else PERSONA
    return [SystemMessage(system), *state["messages"]]


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool] = ALL_TOOLS,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any]:
    bound = model.bind_tools(list(tools))

    async def agent(
        state: ReachyAgentState, config: RunnableConfig
    ) -> dict[str, list[BaseMessage]]:
        turn_context = str(config.get("configurable", {}).get("turn_context", ""))
        # ainvoke でも、stream_mode="messages" で呼ばれればトークンはストリーミングされる
        return {"messages": [await bound.ainvoke(build_prompt(state, turn_context))]}

    graph = StateGraph(ReachyAgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
