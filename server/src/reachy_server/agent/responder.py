"""LangGraph エージェントを Responder として使うアダプタ。"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ..history import anon_thread, person_thread
from ..providers import LLMProvider
from ..responder import ReplyEvent, ReplyMotion, ReplySentence, TurnRequest
from ..text import split_completed
from .graph import build_agent
from .prompt import render_context, user_message
from .tools import memory_ctx_var, motion_sink_var, robot_tool_ctx_var
from .tools.memory import MemoryContext
from .tools.robot import RobotToolContext

RECURSION_LIMIT = 10


def thread_id(request: TurnRequest) -> str:
    """会話スレッド = ロボット × 相手 × 日付（:mod:`reachy_server.history`）。

    相手を認識できないときは、そのターン限りのスレッドにして終了時に消す。
    見知らぬ来訪者どうしで会話が混ざらないように（同じ人でも文脈は途切れるが、安全側に倒す）。
    """
    person = request.context.person
    if person is None:
        return anon_thread(request.robot_id, request.ref)
    return person_thread(request.robot_id, person.id, request.context.situation.local_time.date())


class AgentResponder:
    def __init__(
        self, llm: LLMProvider, checkpointer: BaseCheckpointSaver[Any] | None = None
    ) -> None:
        self.agent: CompiledStateGraph[Any] = build_agent(
            llm.chat_model(), checkpointer=checkpointer
        )

    async def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        sink: list[str] = []
        token = motion_sink_var.set(sink)
        person = request.context.person
        memory_token = memory_ctx_var.set(
            MemoryContext(person_id=person.id if person else None, port=request.memory)
        )
        robot_token = robot_tool_ctx_var.set(RobotToolContext(port=request.robot_tool))
        tid = thread_id(request)
        try:
            config: RunnableConfig = {
                # turn_context は状態ではなく設定で渡す（履歴に保存しない）
                "configurable": {"thread_id": tid, "turn_context": render_context(request)},
                # ツール呼び出しが止まらない場合の安全弁（agent / tools の往復で約 5 回まで）
                "recursion_limit": RECURSION_LIMIT,
            }
            inputs = {"messages": [HumanMessage(user_message(request))]}
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
            saver = self.agent.checkpointer
            if request.context.person is None and isinstance(saver, BaseCheckpointSaver):
                # ターン限りのスレッドは残さない（取消・失敗のときも）
                await saver.adelete_thread(tid)
            # 別タスクから閉じられた場合は reset できないが、シンクはこのターン限りなので問題ない
            with contextlib.suppress(ValueError):
                motion_sink_var.reset(token)
            with contextlib.suppress(ValueError):
                memory_ctx_var.reset(memory_token)
            with contextlib.suppress(ValueError):
                robot_tool_ctx_var.reset(robot_token)


def _drain(sink: list[str]) -> list[ReplyMotion]:
    out = [ReplyMotion(name) for name in sink]
    sink.clear()
    return out
