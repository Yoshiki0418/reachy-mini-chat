"""応答生成の口（Responder）。

セッションは「何を話すか」をこの口から受け取り、合成と送信だけを担う。
実装は LangGraph のエージェント（:class:`reachy_server.agent.AgentResponder`）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from reachy_protocol.messages import Emotion, TurnContext


@dataclass(frozen=True)
class TurnRequest:
    ref: str
    """utt_id か intent_id。"""
    context: TurnContext
    user_text: str | None = None
    """ユーザー発話への応答なら文字起こし。自発発話なら None。"""
    proactive_kind: Literal["greet", "engage", "see_off"] | None = None
    strategy: str | None = None


@dataclass(frozen=True)
class ReplySentence:
    text: str
    emotion: Emotion = "neutral"


@dataclass(frozen=True)
class ReplyMotion:
    name: str


ReplyEvent: TypeAlias = ReplySentence | ReplyMotion


class Responder(Protocol):
    def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        """応答を文単位で順に返す。取消はタスクのキャンセルで行う。"""
        ...
