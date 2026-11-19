"""応答生成の口（Responder）と、LLM をストリーミングして文に区切る最小の実装。

セッションは「何を話すか」をこの口から受け取り、合成と送信だけを担う。
LangGraph のエージェント（記憶・ツール・ガード）への置き換えは #20 以降。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from reachy_protocol.messages import Emotion, TurnContext

from .providers import LLMProvider
from .text import split_completed


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


_SYSTEM = (
    "あなたは研究室の入口にいる小型ロボット「リーチ」です。来た人に親しみやすく短く日本語で話します。"
    "返答は 1〜2 文。記号は使わず、読み上げやすい文にしてください。"
)

_PROACTIVE = {
    "greet": "（目の前に人が来ました。あなたから明るく短く挨拶してください。）",
    "engage": "（相手がこちらを見ています。あなたから短く声をかけてください。）",
    "see_off": "（相手が帰ろうとしています。短く見送ってください。）",
}


def build_messages(request: TurnRequest) -> list[BaseMessage]:
    """最小のプロンプト。構造化コンテキストとペルソナは #64 / #65 で作り込む。"""
    person = request.context.person
    who = f"相手: {person.name or '名前未登録'}（{person.familiarity}）" if person else "相手: 不明"
    system = f"{_SYSTEM}\n{who}"
    if request.user_text is not None:
        user = request.user_text
    else:
        user = _PROACTIVE.get(request.proactive_kind or "greet", _PROACTIVE["greet"])
    return [SystemMessage(system), HumanMessage(user)]


class LLMResponder:
    """LLM の出力をストリーミングし、完結した文から順に返す。"""

    def __init__(self, llm: LLMProvider) -> None:
        self.model = llm.chat_model()

    async def respond(self, request: TurnRequest) -> AsyncIterator[ReplyEvent]:
        buffer = ""
        async for chunk in self.model.astream(build_messages(request)):
            buffer += chunk.text
            sentences, buffer = split_completed(buffer)
            for sentence in sentences:
                yield ReplySentence(sentence)
        tail = buffer.strip()
        if tail:
            yield ReplySentence(tail)
