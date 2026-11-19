"""LangGraph エージェントを Responder として使うアダプタ。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ..providers import LLMProvider
from ..responder import ReplyEvent, ReplyMotion, ReplySentence, TurnRequest
from ..text import split_completed
from .graph import build_agent
from .prompt import render_context, user_message
from .tools import motion_sink_var

RECURSION_LIMIT = 10


def thread_id(request: TurnRequest) -> str:
    """会話スレッド = 相手 × 日付。相手が分からないときは 'anon'。永続化は #21。"""
    person = request.context.person
    day = request.context.situation.local_time.date().isoformat()
    return f"{person.id if person else 'anon'}:{day}"


class AgentResponder:
    def __init__(self, llm: LLMProvider, agent: CompiledStateGraph[Any] | None = None) -> None:
        self.agent = agent or build_agent(llm.chat_model())

    async def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        sink: list[str] = []
        token = motion_sink_var.set(sink)
        try:
            config: RunnableConfig = {
                "configurable": {"thread_id": thread_id(request)},
                # ツール呼び出しが止まらない場合の安全弁（agent / tools の往復で約 5 回まで）
                "recursion_limit": RECURSION_LIMIT,
            }
            inputs = {
                "messages": [HumanMessage(user_message(request))],
                "turn_context": render_context(request),
            }
            buffer = ""
            async for item in self.agent.astream(inputs, config, stream_mode="messages"):
                chunk, meta = cast(tuple[Any, dict[str, Any]], item)
                # ユーザーへの返答（agent ノードのテキスト）だけを読む。ツールの結果は読まない
                if not isinstance(chunk, AIMessageChunk) or meta.get("langgraph_node") != "agent":
                    continue
                buffer += chunk.text
                sentences, buffer = split_completed(buffer)
                for sentence in sentences:
                    for motion in _drain(sink):
                        yield motion
                    yield ReplySentence(sentence)
            tail = buffer.strip()
            for motion in _drain(sink):
                yield motion
            if tail:
                yield ReplySentence(tail)
        finally:
            # 別タスクから閉じられた場合は reset できないが、シンクはこのターン限りなので問題ない
            with contextlib.suppress(ValueError):
                motion_sink_var.reset(token)


def _drain(sink: list[str]) -> list[ReplyMotion]:
    out = [ReplyMotion(name) for name in sink]
    sink.clear()
    return out
