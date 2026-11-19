"""会話履歴（LangGraph のチェックポイント）の保存先と、削除・保持期限。設計書 §4.1。

スレッド ID は ``<robot_id>:<person_id>:<YYYY-MM-DD>``。履歴に残るのは発話・ツール・応答だけで、
相手のプロフィールや状況（turn_context）は保存しない（人物情報の正本はロボット側）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

log = logging.getLogger(__name__)

ANON = "anon"


def person_thread(robot_id: str, person_id: str, day: date) -> str:
    return f"{robot_id}:{person_id}:{day.isoformat()}"


def anon_thread(robot_id: str, ref: str) -> str:
    """相手が不明なターン限りのスレッド。"""
    return f"{robot_id}:{ANON}:{ref}"


def _parse(thread_id: str) -> tuple[str, str, str] | None:
    parts = thread_id.split(":")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


class HistoryStore:
    """チェックポインタの持ち主。``path`` が None ならメモリ上（再起動で消える）。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._saver: BaseCheckpointSaver[Any] | None = None
        self._conn: Any = None
        self._lock = asyncio.Lock()

    async def checkpointer(self) -> BaseCheckpointSaver[Any]:
        """初回に開く（何度呼んでもよい）。"""
        async with self._lock:
            if self._saver is None:
                self._saver = await self._open()
            return self._saver

    async def _open(self) -> BaseCheckpointSaver[Any]:
        if self.path is None:
            return InMemorySaver()
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        saver = AsyncSqliteSaver(self._conn)
        await saver.setup()
        log.info("history store: %s", self.path)
        return saver

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
            self._conn = None
            self._saver = None

    async def thread_ids(self) -> set[str]:
        saver = await self.checkpointer()
        return {
            str(c.config.get("configurable", {}).get("thread_id")) async for c in saver.alist(None)
        }

    async def delete_thread(self, thread_id: str) -> None:
        saver = await self.checkpointer()
        await saver.adelete_thread(thread_id)

    async def forget_person(self, robot_id: str, person_id: str) -> int:
        """その人のスレッドをすべて削除し、削除した数を返す。"""
        doomed = [
            t
            for t in await self.thread_ids()
            if (p := _parse(t)) is not None and p[0] == robot_id and p[1] == person_id
        ]
        for t in doomed:
            await self.delete_thread(t)
        log.info("forgot person robot=%s person=%s threads=%d", robot_id, person_id, len(doomed))
        return len(doomed)

    async def prune(self, retention_days: int, today: date | None = None) -> int:
        """保持期限を過ぎた日付のスレッドを削除し、削除した数を返す。

        日付の読めないスレッド（ターン限りの anon など、本来は残っていないもの）も削除する。
        """
        cutoff = (today or date.today()) - timedelta(days=retention_days)
        doomed: list[str] = []
        for t in await self.thread_ids():
            parsed = _parse(t)
            try:
                day = date.fromisoformat(parsed[2]) if parsed and parsed[1] != ANON else None
            except ValueError:
                day = None
            if day is None or day < cutoff:
                doomed.append(t)
        for t in doomed:
            await self.delete_thread(t)
        if doomed:
            log.info("pruned %d history threads older than %s", len(doomed), cutoff)
        return len(doomed)

    async def prune_forever(self, retention_days: int, interval_s: float = 24 * 3600) -> None:
        """起動時と、以後 interval_s ごとに prune する（アプリの lifespan で動かす）。"""
        while True:
            with contextlib.suppress(Exception):
                await self.prune(retention_days)
            await asyncio.sleep(interval_s)
