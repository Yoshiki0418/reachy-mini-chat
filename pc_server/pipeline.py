"""会話パイプライン: STT → LangGraph エージェント(stream) → 文分割 → SBV2 TTS。

1ターンを、順序付きイベントの async ジェネレータとして返す:
  {"type":"stt",      "text": <認識文>}
  {"type":"motion",   "name": <motion名>}
  {"type":"audio",    "sr": int, "audio": np.ndarray(float32), "text": <合成した文>}
  {"type":"turn_end"}

文単位で TTS するので、最初の文が読み上がるまでのレイテンシを短縮できる。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

from pc_server.agent import (
    PersonContext,
    build_agent,
    current_person_var,
    motion_sink_var,
)
from pc_server.models import WhisperSTT, get_tts
from pc_server.person import FaceIdentifier, PersonStore

# 文の区切りとみなす文字。
_SENTENCE_END = "。.!?！？\n"

# 自発発話（proactive）の合図。STT の代わりにこの指示をエージェントへ渡し、ロボットの側から
# 話し始めさせる。同一 thread_id で渡すので直前の会話と重複しにくい。
_PROACTIVE_DIRECTIVES = {
    "idle": (
        "（合図：会話が少し途切れています。目の前の相手に、あなたの方から自然に短く"
        "話題を振ってください。直前の会話と重複しない話題で。この合図自体には触れないこと。）"
    ),
    "greet": (
        "（合図：目の前に人が来ました。あなたの方から明るく短く声をかけて挨拶してください。"
        "この合図自体には触れないこと。）"
    ),
    "engage": (
        "（合図：相手がこちらをじっと見ています。あなたに関心がありそうなので、"
        "あなたの方から明るく短く声をかけてください。この合図自体には触れないこと。）"
    ),
}

# ロボットが自分の名前で呼ばれたと判断する語（STT 表記ゆれを含む）。
# 広すぎる "reach" は英語幻聴に誤マッチするので入れない。
_NAME_CALLS = ("リーチ", "りーち", "レイチ", "れいち", "reachy")


def _is_name_call(text: str) -> bool:
    low = text.lower()
    return any(n in low for n in _NAME_CALLS)


# 相手の名前をまだ知らないときに、自発発話へ足す「名前を尋ねる」指示。
_ASK_NAME_DIRECTIVE = (
    "（補足：あなたはまだこの相手の名前を知りません。挨拶や会話の流れの中で、"
    "初めましての人には自分から明るく名前を尋ねてください。教えてもらったら set_person_name "
    "で覚えること。しつこく問い詰めない。）"
)


def _split_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """buffer から「区切り文字で終わる完結文」を切り出す。

    戻り値: (完結した文のリスト, 未完の残り)
    """
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_END:
            piece = buffer[start : i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    return sentences, buffer[start:]


class ConversationPipeline:
    def __init__(self) -> None:
        self.stt = WhisperSTT()
        self.tts = get_tts()
        self.agent = build_agent()
        self.persons = PersonStore()
        # 顔認識モデルの初期化・DL に失敗しても会話は続けられるよう握りつぶす。
        try:
            self.face: FaceIdentifier | None = FaceIdentifier()
        except Exception as e:
            print(f"[person] FaceIdentifier 無効化: {e}", flush=True)
            self.face = None

    async def _tts(self, text: str) -> tuple[int, np.ndarray]:
        # SBV2 の合成は CPU バウンドなのでスレッドに逃がしてイベントループを塞がない。
        return await asyncio.to_thread(self.tts.synthesize, text)

    async def _identify_person(
        self, frame_b64: str | None
    ) -> tuple[str | None, PersonContext | None, bool]:
        """フレームから相手を同定し、(文脈メモ, ツール用コンテキスト, 名前未登録か) を返す。

        文脈メモは LLM に前置きする「（相手メモ：…）」。認識できなければ (None, None, False)。
        3 要素目 needs_name は「相手はいるが名前をまだ知らない」＝新規 or 未登録の返し。
        """
        if not frame_b64 or self.face is None:
            return None, None, False
        emb = await asyncio.to_thread(self.face.embed_jpeg_b64, frame_b64)
        if emb is None:
            return None, None, False
        person, is_new = self.persons.get_or_create(emb)
        self.persons.touch(person.id)
        if not is_new:
            # 既知でも埋め込みを少しずつ蓄えて同定を安定化（上限内でのみ）。
            self.persons.add_embedding(person.id, emb)

        if is_new:
            note = "（相手メモ：初めて会う人のようです。名前はまだ未登録。）"
        elif person.name:
            facts = "／".join(person.facts) if person.facts else "特になし"
            note = f"（相手メモ：名前={person.name}。覚えていること：{facts}。）"
        else:
            facts = "／".join(person.facts) if person.facts else "特になし"
            note = f"（相手メモ：以前会ったことのある人。名前は未登録。覚えていること：{facts}。）"
        needs_name = not person.name
        return note, PersonContext(self.persons, person.id), needs_name

    @staticmethod
    def _content_with_optional_image(text: str, frame_b64: str | None) -> Any:
        """テキスト（＋あればカメラ画像）を LLM メッセージ content に組み立てる。"""
        if not frame_b64:
            return text
        return [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                # detail=low で画像トークンを一定に抑える（履歴に積んでも安価）
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low",
                },
            },
        ]

    async def _stream_agent(
        self,
        user_content: Any,
        session_id: str,
        person_ctx: PersonContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """エージェントをトークンストリームで実行し、文単位で TTS・モーションを yield する。

        run_turn（ユーザー発話）と run_proactive（自発発話）で共有する本体。
        末尾に turn_end を必ず出す。
        """
        # --- モーション・人物メモリの side-channel をこのターン用に用意 ---
        sink: list[dict] = []
        motion_sink_var.set(sink)
        current_person_var.set(person_ctx)

        def drain_motions() -> list[dict]:
            out = list(sink)
            sink.clear()
            return out

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        buffer = ""
        async for chunk, _meta in self.agent.astream(
            {"messages": [HumanMessage(content=user_content)]},
            config=config,
            stream_mode="messages",
        ):
            # content は str か content block の list。.text は常にテキスト部分の str を返す。
            if not isinstance(chunk, AIMessageChunk):
                continue
            text = chunk.text
            if not text:
                continue
            buffer += text
            sentences, buffer = _split_completed_sentences(buffer)
            for sentence in sentences:
                for m in drain_motions():
                    yield m
                sr, wav = await self._tts(sentence)
                yield {"type": "audio", "sr": sr, "audio": wav, "text": sentence}

        # 残りの未完文を flush
        tail = buffer.strip()
        if tail:
            for m in drain_motions():
                yield m
            sr, wav = await self._tts(tail)
            yield {"type": "audio", "sr": sr, "audio": wav, "text": tail}

        # ツールだけ呼ばれて本文が無い等でも、残ったモーションを送る
        for m in drain_motions():
            yield m

        yield {"type": "turn_end"}

    async def run_turn(
        self,
        audio: np.ndarray,
        session_id: str,
        frame_b64: str | None = None,
        eye_contact: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        # --- STT ---
        user_text = await asyncio.to_thread(self.stt.transcribe, audio, "ja")
        yield {"type": "stt", "text": user_text}
        if not user_text:
            yield {"type": "turn_end"}
            return

        # --- 応答ゲート ---
        # 目が合っていた(eye_contact) か 名前で呼ばれた 時だけ応答する。
        # eye_contact が None（robot が情報を送らない＝gaze 無効）なら従来通り常に応答。
        addressed = (eye_contact is None) or eye_contact or _is_name_call(user_text)
        if not addressed:
            print(f"[gate] 傍聴（割り込まない）: {user_text!r}", flush=True)
            yield {"type": "turn_end"}
            return

        note, person_ctx, _needs_name = await self._identify_person(frame_b64)
        text = f"{note}\n{user_text}" if note else user_text
        user_content = self._content_with_optional_image(text, frame_b64)
        async for ev in self._stream_agent(user_content, session_id, person_ctx):
            yield ev

    async def run_proactive(
        self, session_id: str, frame_b64: str | None = None, reason: str = "idle"
    ) -> AsyncIterator[dict[str, Any]]:
        """ロボットの側から話し始める自発発話ターン。STT を介さない。"""
        directive = _PROACTIVE_DIRECTIVES.get(reason, _PROACTIVE_DIRECTIVES["idle"])
        note, person_ctx, needs_name = await self._identify_person(frame_b64)
        # 相手の名前をまだ知らなければ、自発発話のついでに自分から名前を尋ねる。
        if needs_name:
            directive = f"{directive}{_ASK_NAME_DIRECTIVE}"
        text = f"{note}\n{directive}" if note else directive
        content = self._content_with_optional_image(text, frame_b64)
        async for ev in self._stream_agent(content, session_id, person_ctx):
            yield ev
