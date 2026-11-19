"""会話履歴の保存・忘却・保持期限と、場の文脈バッファ。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from reachy_protocol.messages import (
    FloorDecision,
    PersonForget,
    PersonProfile,
    Situation,
    TurnContext,
)
from reachy_protocol.testing import FakeRobotClient, default_context
from reachy_protocol.transport import memory_pair
from reachy_server.agent import AgentResponder
from reachy_server.app import create_app
from reachy_server.config import ServerSettings
from reachy_server.history import HistoryStore, anon_thread, person_thread
from reachy_server.overheard import OverheardBuffer
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers
from reachy_server.responder import TurnRequest
from reachy_server.session import Session

JST = timezone(timedelta(hours=9))
DAY = date(2026, 11, 19)
SECRET_FACT = "猫を三匹飼っている"


def ctx(person_id: str | None = "p1") -> TurnContext:
    person = (
        PersonProfile(
            id=person_id, name="山田", familiarity="daily", visit_count=9, facts=[SECRET_FACT]
        )
        if person_id
        else None
    )
    return TurnContext(
        person=person,
        situation=Situation(people_count=1, local_time=datetime(2026, 11, 19, 9, tzinfo=JST)),
    )


async def turn(
    responder: AgentResponder, text: str, person_id: str | None = "p1", **kw: Any
) -> None:
    req = TurnRequest(ref=f"u-{text}", context=ctx(person_id), user_text=text, robot_id="r1", **kw)
    _ = [e async for e in responder.respond(req)]


async def seed(store: HistoryStore, *thread_ids: str) -> None:
    saver = await store.checkpointer()
    from langgraph.checkpoint.base import empty_checkpoint

    for t in thread_ids:
        await saver.aput(
            {"configurable": {"thread_id": t, "checkpoint_ns": ""}}, empty_checkpoint(), {}, {}
        )


# ---------------------------------------------------------------- 永続化


async def test_history_survives_restart_with_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "history.sqlite"
    store1 = HistoryStore(db)
    await turn(
        AgentResponder(FakeLLM(["一回目の返事。"]), await store1.checkpointer()), "昨日の続きだけど"
    )
    await store1.close()

    store2 = HistoryStore(db)  # 再起動
    llm = FakeLLM(["二回目の返事。"])
    await turn(AgentResponder(llm, await store2.checkpointer()), "覚えてる？")
    prompt = llm.models[0].inputs[0]
    assert [m.text for m in prompt[1:]] == ["昨日の続きだけど", "一回目の返事。", "覚えてる？"]
    assert await store2.thread_ids() == {person_thread("r1", "p1", DAY)}
    await store2.close()


async def test_turn_context_and_overheard_are_not_persisted(tmp_path: Path) -> None:
    """相手の事実や、周りで聞こえた会話は履歴 DB に残さない（人物情報の正本はロボット側）。"""
    store = HistoryStore(tmp_path / "h.sqlite")
    llm = FakeLLM(["そうなんだ。"])
    await turn(
        AgentResponder(llm, await store.checkpointer()), "やあ", overheard=("来週の締切やばい",)
    )
    # プロンプトには入っている
    system = llm.models[0].inputs[0][0].text
    assert SECRET_FACT in system and "来週の締切やばい" in system
    # 履歴には入っていない
    saver = await store.checkpointer()
    stored = [repr(c.checkpoint) async for c in saver.alist(None)]
    assert stored and any("やあ" in s for s in stored)
    assert not any(SECRET_FACT in s or "締切" in s for s in stored)
    await store.close()


async def test_anonymous_turns_leave_nothing_in_sqlite(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "h.sqlite")
    await turn(
        AgentResponder(FakeLLM(["どうも。"]), await store.checkpointer()),
        "こんにちは",
        person_id=None,
    )
    assert await store.thread_ids() == set()
    await store.close()


# ---------------------------------------------------------------- 忘却と保持期限


async def test_forget_person_deletes_only_that_person_on_that_robot() -> None:
    store = HistoryStore()
    keep = {person_thread("r1", "p2", DAY), person_thread("r2", "p1", DAY)}
    await seed(
        store,
        person_thread("r1", "p1", DAY),
        person_thread("r1", "p1", DAY - timedelta(days=1)),
        *keep,
    )
    assert await store.forget_person("r1", "p1") == 2
    assert await store.thread_ids() == keep


async def test_prune_by_retention() -> None:
    store = HistoryStore()
    today = DAY
    keep = {person_thread("r1", "p1", today), person_thread("r1", "p1", today - timedelta(days=14))}
    await seed(
        store,
        *keep,
        person_thread("r1", "p1", today - timedelta(days=15)),
        anon_thread("r1", "u9"),  # 本来残らないもの
        "garbage-thread",
        "r1:p1:not-a-date",
    )
    assert await store.prune(14, today=today) == 4
    assert await store.thread_ids() == keep


async def test_prune_forever_prunes_at_startup() -> None:
    store = HistoryStore()
    await seed(store, person_thread("r1", "p1", date(2000, 1, 1)))
    task = asyncio.create_task(store.prune_forever(14, interval_s=3600))
    await asyncio.sleep(0.05)
    task.cancel()
    assert await store.thread_ids() == set()


async def test_person_forget_message_clears_server_history(
    settings: ServerSettings, token: str
) -> None:
    store = HistoryStore()
    providers = Providers(stt=FakeSTT(), tts=FakeTTS(), llm=FakeLLM(["うん。"]))
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, providers, store).run())
    async with FakeRobotClient(robot_end, robot_id="reachy-1", token=token) as robot:
        await robot.connect()
        t = await robot.say()
        await robot.decide(t.utt_id, "respond", default_context(person_id="p-yamada"))
        await robot.next_turn()
        assert {
            tid.split(":")[:2] == ["reachy-1", "p-yamada"] for tid in await store.thread_ids()
        } == {True}
        await robot.ep.send(PersonForget(person_id="p-yamada"))
        await robot.ping()  # 処理済みを待つ
        assert await store.thread_ids() == set()
    await asyncio.wait_for(task, 1.0)


def test_app_lifespan_opens_and_closes_store(settings: ServerSettings) -> None:
    store = HistoryStore()
    app = create_app(settings, history=store)
    with TestClient(app):
        assert store._saver is not None  # pyright: ignore[reportPrivateUsage]
    assert store._saver is None  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------- 場の文脈バッファ


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_overheard_expires_and_is_bounded() -> None:
    clock = Clock()
    buf = OverheardBuffer(ttl_s=600, max_items=3, clock=clock)
    for i in range(4):
        buf.add(f"話{i}")
        clock.now += 100
    assert buf.recent() == ("話1", "話2", "話3")  # 上限 3 件
    clock.now = 100 + 600 + 1  # 話1 の期限切れ
    assert buf.recent() == ("話2", "話3")
    buf.add("   ")
    assert buf.recent() == ("話2", "話3")  # 空は足さない
    buf.clear()
    assert buf.recent() == ()


def test_overheard_disabled() -> None:
    buf = OverheardBuffer(ttl_s=0)
    buf.add("聞こえた")
    assert not buf.enabled and buf.recent() == ()


async def test_ignored_utterances_become_overheard_context(
    settings: ServerSettings, token: str
) -> None:
    llm = FakeLLM(["そうなんだね。"])
    providers = Providers(
        stt=FakeSTT(script=["来週の締切やばいね", "リーチ、聞いてた？"]), tts=FakeTTS(), llm=llm
    )
    robot_end, server_end = memory_pair()
    task = asyncio.create_task(Session(server_end, settings, providers, HistoryStore()).run())
    async with FakeRobotClient(robot_end, token=token) as robot:
        await robot.connect()
        first = await robot.say()
        await robot.ep.send(FloorDecision(utt_id=first.utt_id, decision="ignore"))
        second = await robot.say()
        await robot.decide(second.utt_id, "respond", default_context())
        await robot.next_turn()
    await asyncio.wait_for(task, 1.0)
    system = llm.models[0].inputs[0][0].text
    assert "最近まわりで聞こえた会話" in system and "「来週の締切やばいね」" in system
    assert "自分から話題にしない" in system
    assert [m.text for m in llm.models[0].inputs[0][1:]] == ["リーチ、聞いてた？"]


async def test_ids_containing_colons_are_handled() -> None:
    """プロトコルの ID は ':' を含みうる。忘却も保持期限も正しく効く（Codex 指摘 #95）。"""
    store = HistoryStore()
    tricky = person_thread("lab:entrance", "p:1", DAY)
    other = person_thread("lab:entrance", "p", DAY)
    await seed(store, tricky, other)
    assert await store.prune(14, today=DAY) == 0  # 壊れたスレッド扱いで消されない
    assert await store.forget_person("lab:entrance", "p:1") == 1
    assert await store.thread_ids() == {other}


def test_thread_id_components_are_reversible() -> None:
    from reachy_server.history import _parse  # pyright: ignore[reportPrivateUsage]

    assert _parse(person_thread("a:b", "c%3Ad", DAY)) == ("a:b", "c%3Ad", DAY.isoformat())
    assert _parse(anon_thread("r:1", "u:9")) == ("r:1", None, "u:9")
    # 実在の person_id が "anon" や "*" でも、相手不明のスレッドと区別できる
    assert _parse(person_thread("r", "anon", DAY)) == ("r", "anon", DAY.isoformat())
    assert _parse(person_thread("r", "*", DAY)) == ("r", "*", DAY.isoformat())
