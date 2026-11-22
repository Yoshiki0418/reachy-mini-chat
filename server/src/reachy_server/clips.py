"""フィラーと定型挨拶の事前合成。設計書 §4.3。

起動時に TTS で合成してメモリに持ち、ロボットの要求で配る（:mod:`reachy_server.app`）。
文言は個人情報を含まない固定のもの。**使うタイミングはロボットが決める**（反射的な間つなぎと、
サーバー断のときの入室反応だけで、タイマーで話し始めるためには使わない）。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from reachy_protocol.clips import ClipInfo, ClipKind, ClipManifest
from reachy_protocol.messages import Emotion

from .providers import TTSProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipSpec:
    id: str
    kind: ClipKind
    text: str
    emotion: Emotion = "neutral"


DEFAULT_CLIPS: tuple[ClipSpec, ...] = (
    # 間つなぎ。LLM の最初の文が遅れそうなときに、ロボットが先に流す
    ClipSpec("filler-etto", "filler", "えっとね。", "thinking"),
    ClipSpec("filler-unto", "filler", "うーんとね。", "thinking"),
    ClipSpec("filler-sou", "filler", "そうだなあ。", "thinking"),
    ClipSpec("filler-matte", "filler", "ちょっと待ってね。", "thinking"),
    ClipSpec("filler-hai", "filler", "はーい。", "happy"),
    # サーバー断のときの入室反応（会話はできない）
    ClipSpec("greeting-hello", "greeting", "こんにちは！", "happy"),
    ClipSpec("greeting-morning", "greeting", "おはようございます！", "happy"),
    ClipSpec("greeting-evening", "greeting", "こんばんは！", "happy"),
    ClipSpec("greeting-welcome", "greeting", "いらっしゃいませ！", "happy"),
    ClipSpec("notice-offline", "notice", "ごめんね、いまはお話しできないんだ。", "sad"),
)


@dataclass(frozen=True)
class Clip:
    info: ClipInfo
    wav: bytes


class ClipStore:
    def __init__(self, specs: tuple[ClipSpec, ...] = DEFAULT_CLIPS) -> None:
        ids = [s.id for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError("音声の id が重複している")
        self._specs = specs
        self._clips: dict[str, Clip] = {}
        self._manifest: ClipManifest | None = None

    @property
    def manifest(self) -> ClipManifest | None:
        """合成が終わるまでは None。"""
        return self._manifest

    def get(self, clip_id: str) -> Clip | None:
        return self._clips.get(clip_id)

    async def build(self, tts: TTSProvider) -> None:
        """全部を合成する。1 つ失敗しても、ほかは配る（失敗した分はログに残し、一覧に載せない）。"""
        clips: dict[str, Clip] = {}
        for spec in self._specs:
            try:
                wav = await tts.synthesize(spec.text, spec.emotion)
            except Exception:
                log.exception("事前合成に失敗: %s", spec.id)
                continue
            info = ClipInfo(
                id=spec.id,
                kind=spec.kind,
                text=spec.text,
                emotion=spec.emotion,
                sha256=hashlib.sha256(wav).hexdigest(),
                nbytes=len(wav),
            )
            clips[spec.id] = Clip(info, wav)
        self._clips = clips
        self._manifest = ClipManifest(
            version=_version(clips), clips=[c.info for c in clips.values()]
        )
        log.info("事前合成の音声を用意: %d / %d 件", len(clips), len(self._specs))


def _version(clips: dict[str, Clip]) -> str:
    """声・文言・感情のどれかが変わると変わる。内容（音声のハッシュ）から決める。"""
    h = hashlib.sha256()
    for clip in clips.values():
        i = clip.info
        h.update(f"{i.id}\0{i.kind}\0{i.text}\0{i.emotion}\0{i.sha256}\n".encode())
    return h.hexdigest()[:16]


__all__ = ["DEFAULT_CLIPS", "Clip", "ClipSpec", "ClipStore"]
