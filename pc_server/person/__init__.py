"""人物メモリ (Phase 5)。

顔の埋め込み (SFace) で人物を同定し、名前・覚えた事実を JSON に永続化する。
声ベースの同定は将来拡張（roadmap 参照）。
"""

from __future__ import annotations

from .face_id import FaceIdentifier
from .store import Person, PersonStore

__all__ = ["FaceIdentifier", "Person", "PersonStore"]
