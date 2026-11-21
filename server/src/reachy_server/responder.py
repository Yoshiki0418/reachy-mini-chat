"""応答生成の口（Responder）。

セッションは「何を話すか」をこの口から受け取り、合成と送信だけを担う。
実装は LangGraph のエージェント（:class:`reachy_server.agent.AgentResponder`）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from reachy_protocol.messages import Emotion, TurnContext


@dataclass(frozen=True)
class MemoryResult:
    """memory.op に対するロボットの返事。``timeout`` はロボットが時間内に答えなかったこと。"""

    result: Literal["applied", "rejected", "timeout"]
    reason: str | None = None


MemoryOpName: TypeAlias = Literal["set_name", "add_fact", "forget"]

MemoryPort: TypeAlias = Callable[[str, MemoryOpName, str | None], Awaitable[MemoryResult]]
"""(person_id, op, value) を memory.op としてロボットへ送り、memory.ack を待つ。"""


@dataclass(frozen=True)
class TurnRequest:
    ref: str
    """utt_id か intent_id。"""
    context: TurnContext
    user_text: str | None = None
    """ユーザー発話への応答なら文字起こし。自発発話なら None。"""
    proactive_kind: Literal["greet", "engage", "see_off"] | None = None
    strategy: str | None = None
    robot_id: str = "robot"
    """会話スレッドをロボットごとに分けるため。"""
    overheard: tuple[str, ...] = ()
    """最近まわりで聞こえた、自分宛てではない発話（場の文脈）。"""
    memory: MemoryPort | None = None
    """記憶操作の口。None なら（テストなど）記憶ツールは使えない。"""


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
