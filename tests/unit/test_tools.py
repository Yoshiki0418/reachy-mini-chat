import numpy as np
import pytest

from pc_server.agent.tools.memory_tool import (
    PersonContext,
    _looks_like_name,
    current_person_var,
    remember_fact,
    set_person_name,
)
from pc_server.agent.tools.motion_tool import express_motion, motion_sink_var
from pc_server.person.store import PersonStore


@pytest.mark.parametrize("name", ["山田", "たろう", "Alice"])
def test_looks_like_name_accepts(name: str) -> None:
    assert _looks_like_name(name)


@pytest.mark.parametrize(
    "name",
    ["", "とても長すぎる名前ですよね", "山田です", "はい、そうです", "よろしくお願いします"],
)
def test_looks_like_name_rejects(name: str) -> None:
    assert not _looks_like_name(name)


@pytest.fixture
def person_ctx(tmp_path):
    store = PersonStore(path=str(tmp_path / "people.json"))
    p, _ = store.get_or_create(np.asarray([1.0, 0.0], dtype=np.float32))
    token = current_person_var.set(PersonContext(store, p.id))
    yield store, p.id
    current_person_var.reset(token)


def test_set_person_name_saves(person_ctx) -> None:
    store, pid = person_ctx
    set_person_name.invoke({"name": "山田"})
    assert store.people[pid].name == "山田"


def test_set_person_name_rejects_sentence(person_ctx) -> None:
    store, pid = person_ctx
    out = set_person_name.invoke({"name": "山田です"})
    assert store.people[pid].name is None
    assert "見送り" in out


def test_memory_tools_without_person_do_nothing() -> None:
    assert "認識できていない" in set_person_name.invoke({"name": "山田"})
    assert "認識できていない" in remember_fact.invoke({"fact": "猫好き"})


def test_remember_fact(person_ctx) -> None:
    store, pid = person_ctx
    remember_fact.invoke({"fact": "猫を飼っている"})
    assert store.people[pid].facts == ["猫を飼っている"]


def test_express_motion_appends_to_sink() -> None:
    sink: list[dict] = []
    token = motion_sink_var.set(sink)
    try:
        out = express_motion.invoke({"motion": "nod"})
    finally:
        motion_sink_var.reset(token)
    assert sink == [{"type": "motion", "name": "nod"}]
    assert "do not mention" in out


def test_express_motion_without_sink_is_safe() -> None:
    express_motion.invoke({"motion": "happy"})
