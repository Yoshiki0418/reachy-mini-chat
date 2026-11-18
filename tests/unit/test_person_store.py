import numpy as np
import pytest

from pc_server.person.store import MAX_EMB_PER_PERSON, PersonStore


def unit(*xs: float) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path) -> PersonStore:
    return PersonStore(path=str(tmp_path / "people.json"))


def test_new_person_is_created(store: PersonStore) -> None:
    p, is_new = store.get_or_create(unit(1, 0, 0))
    assert is_new
    assert p.name is None
    assert len(store.people) == 1


def test_same_face_is_identified(store: PersonStore) -> None:
    p1, _ = store.get_or_create(unit(1, 0, 0))
    p2, is_new = store.get_or_create(unit(1, 0.05, 0))  # ほぼ同じ向き
    assert not is_new
    assert p2.id == p1.id


def test_dissimilar_face_is_new_person(store: PersonStore) -> None:
    p1, _ = store.get_or_create(unit(1, 0, 0))
    p2, is_new = store.get_or_create(unit(0, 1, 0))  # 直交 = 類似度 0
    assert is_new
    assert p2.id != p1.id


def test_add_embedding_is_capped(store: PersonStore) -> None:
    p, _ = store.get_or_create(unit(1, 0, 0))
    for _ in range(MAX_EMB_PER_PERSON + 5):
        store.add_embedding(p.id, unit(1, 0, 0))
    assert len(store.people[p.id].embeddings) == MAX_EMB_PER_PERSON


def test_name_and_facts_persist(tmp_path) -> None:
    path = str(tmp_path / "people.json")
    s1 = PersonStore(path=path)
    p, _ = s1.get_or_create(unit(1, 0, 0))
    s1.set_name(p.id, "  山田 ")
    s1.add_fact(p.id, "コーヒーが好き")
    s1.add_fact(p.id, "コーヒーが好き")  # 重複は追加しない

    s2 = PersonStore(path=path)
    loaded = s2.people[p.id]
    assert loaded.name == "山田"
    assert loaded.facts == ["コーヒーが好き"]


def test_unknown_person_updates_are_noops(store: PersonStore) -> None:
    store.set_name("missing", "x")
    store.add_fact("missing", "y")
    store.touch("missing")
    assert store.people == {}


def test_corrupt_file_starts_empty(tmp_path) -> None:
    path = tmp_path / "people.json"
    path.write_text("{not json", encoding="utf-8")
    assert PersonStore(path=str(path)).people == {}
