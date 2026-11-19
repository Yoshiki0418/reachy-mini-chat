"""人物ストア (JSON 永続化)。

顔特徴のコサイン類似で人物を同定し、名前・事実・埋め込み・最終対面時刻を保持する。
MVP なのでグラフDBではなく単一 JSON。スレッド安全のため mutation は lock で保護する。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from uuid import uuid4

import numpy as np

# SFace 推奨のコサイン閾値（これ以上で同一人物とみなす）。
COSINE_THRESHOLD = float(os.environ.get("PERSON_MATCH_THRESHOLD", "0.363"))
# 1 人あたり保持する埋め込みの上限（多様な角度・照明を少しだけ蓄える）。
MAX_EMB_PER_PERSON = int(os.environ.get("PERSON_MAX_EMB", "8"))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_DEFAULT_PATH = os.path.join(_REPO_ROOT, "data", "people.json")


@dataclass
class Person:
    id: str
    name: str | None = None
    facts: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    created: float = 0.0
    last_seen: float = 0.0


class PersonStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("PERSON_DB_PATH", _DEFAULT_PATH)
        self._lock = threading.Lock()
        self.people: dict[str, Person] = {}
        self._load()

    # ---- 永続化 ----
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data.get("people", []):
                p = Person(**d)
                self.people[p.id] = p
        except Exception as e:  # noqa: BLE001
            print(f"[person] ストア読込失敗（新規で開始）: {e}", flush=True)

    def _save_locked(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"people": [asdict(p) for p in self.people.values()]},
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    # ---- 同定 ----
    def identify(self, emb: np.ndarray) -> Person | None:
        """最も似た既知人物を返す（コサインが閾値未満なら None）。"""
        best: Person | None = None
        best_sim = COSINE_THRESHOLD
        for p in self.people.values():
            for e in p.embeddings:
                sim = float(np.dot(emb, np.asarray(e, dtype=np.float32)))
                if sim > best_sim:
                    best_sim = sim
                    best = p
        return best

    def get_or_create(self, emb: np.ndarray) -> tuple[Person, bool]:
        """同定できればその人物、できなければ新規登録。戻り値: (person, is_new)。"""
        with self._lock:
            p = self.identify(emb)
            if p is not None:
                return p, False
            now = time.time()
            pid = uuid4().hex[:12]
            p = Person(id=pid, embeddings=[emb.tolist()], created=now, last_seen=now)
            self.people[pid] = p
            self._save_locked()
            return p, True

    # ---- 更新 ----
    def add_embedding(self, person_id: str, emb: np.ndarray) -> None:
        with self._lock:
            p = self.people.get(person_id)
            if p is None or len(p.embeddings) >= MAX_EMB_PER_PERSON:
                return
            p.embeddings.append(emb.tolist())
            self._save_locked()

    def set_name(self, person_id: str, name: str) -> None:
        with self._lock:
            p = self.people.get(person_id)
            if p is None:
                return
            p.name = name.strip()
            self._save_locked()

    def add_fact(self, person_id: str, fact: str) -> None:
        with self._lock:
            p = self.people.get(person_id)
            if p is None:
                return
            fact = fact.strip()
            if fact and fact not in p.facts:
                p.facts.append(fact)
                self._save_locked()

    def touch(self, person_id: str) -> None:
        with self._lock:
            p = self.people.get(person_id)
            if p is None:
                return
            p.last_seen = time.time()
            self._save_locked()
