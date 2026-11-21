"""会話の要約。相手が退出したら、その日の会話を短くまとめてロボットに保存させる（設計書 §2.2・§3.6）。

要約は ``memory.op(add_episode)`` でロボットに返す。人物の記憶の正はロボットなので、サーバーは
要約を保存せず、ロボットが検証して保存する。次に会ったとき ``context.person.last_summary`` として
戻ってくる。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver

from .agent.prompt import PROACTIVE_DIRECTIVES
from .providers import LLMProvider

log = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 300
"""ロボットに送る要約の上限。次回の会話の話題にする一言なので短くてよい（プロトコルの上限は 1000）。"""

SUMMARY_PROMPT = """あなたは、研究室の受付ロボット「リーチ」の記憶係です。
次の会話を、次にその人に会ったときの話題にするための短いメモに要約してください。

# ルール
- 日本語で 1〜3 文、全体で 150 文字以内。「〜と話していた」のように事実だけを書く。
- 相手が自分から話した、次回の話題になることだけを残す（取り組んでいること、好きなもの、予定など）。
- 住所・電話・メールアドレス・健康・家族・恋愛・他人の私的な情報は書かない。
- 評価や推測、説教、「来ていない」などの責める表現は書かない。
- 話題にできることが何もなければ、「なし」とだけ書く。
- 要約の本文だけを出力する（前置き・箇条書き・引用符は不要）。
"""

_NONE_MARKERS = {"なし", "無し", "特になし", "なし。"}
_DIRECTIVES = set(PROACTIVE_DIRECTIVES.values())


def dialogue(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    """履歴から、（話者, 発言）だけを取り出す。ツールの呼び出し・結果と、自発発話の合図は除く。"""
    out: list[tuple[str, str]] = []
    for m in messages:
        text = m.text.strip() if isinstance(m, BaseMessage) else ""
        if not text:
            continue
        if isinstance(m, HumanMessage):
            if text not in _DIRECTIVES:
                out.append(("相手", text))
        elif isinstance(m, AIMessage):
            out.append(("リーチ", text))
    return out


def render(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in turns)


def clean_summary(text: str) -> str | None:
    """LLM の出力を整える。話題がない・空なら None。"""
    text = text.strip().strip("「」\"'").strip()
    if not text or text in _NONE_MARKERS:
        return None
    return text[:MAX_SUMMARY_CHARS]


class Summarizer:
    """スレッド（ロボット × 相手 × 日付）の会話を要約する。"""

    def __init__(self, llm: LLMProvider, checkpointer: BaseCheckpointSaver) -> None:  # pyright: ignore[reportMissingTypeArgument,reportUnknownParameterType]
        self._llm = llm
        self._saver = checkpointer
        self._done: dict[str, str] = {}
        """要約済みのスレッドと、そのときの会話の指紋。同じ内容を二重に要約しない。

        発言数ではなく内容で見る。忘れた後に同じ長さの別の会話をしても、要約が飛ばされないように。
        """
        self._locks: dict[str, asyncio.Lock] = {}
        """スレッドごとの鍵。同じスレッドの要約が同時に走っても、LLM を二重に呼ばない。"""

    async def messages(self, thread_id: str) -> list[AnyMessage]:
        tup = await self._saver.aget_tuple({"configurable": {"thread_id": thread_id}})
        if tup is None:
            return []
        return list(tup.checkpoint["channel_values"].get("messages", []))

    async def summarize_thread(self, thread_id: str) -> str | None:
        async with self._locks.setdefault(thread_id, asyncio.Lock()):
            turns = dialogue(await self.messages(thread_id))
            # 相手が一言も話していない（こちらの挨拶だけ）なら、覚えておくことはない
            if not any(who == "相手" for who, _ in turns):
                return None
            transcript = render(turns)
            fingerprint = hashlib.sha256(transcript.encode()).hexdigest()
            if self._done.get(thread_id) == fingerprint:
                return None
            reply = await self._llm.chat_model().ainvoke(
                [SystemMessage(SUMMARY_PROMPT), HumanMessage(transcript)]
            )
            self._done[thread_id] = fingerprint
            return clean_summary(reply.text)
