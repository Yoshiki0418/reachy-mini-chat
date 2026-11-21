"""Irodori-TTS のプロバイダ。偽のワーカー（本物の子プロセス）でプロトコルと復旧を検証する。"""

from __future__ import annotations

import asyncio
import io
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from reachy_protocol.messages import Emotion
from reachy_server.config import ServerSettings
from reachy_server.providers import TTSProvider, build_providers
from reachy_server.providers.irodori import (
    EMOTION_EMOJI,
    IrodoriTTS,
    WorkerError,
    prepare_text,
    worker_command,
)
from reachy_server.providers.irodori_worker import CancelSet

FAKE_WORKER = Path(__file__).with_name("irodori_fake_worker.py")


def fake_command(log: Path, *extra: str) -> list[str]:
    return [sys.executable, str(FAKE_WORKER), "--log", str(log), *extra]


def starts(log: Path) -> int:
    return log.read_text(encoding="utf-8").splitlines().count("start")


def texts(log: Path) -> list[str]:
    return [x[5:] for x in log.read_text(encoding="utf-8").splitlines() if x.startswith("text:")]


@pytest.fixture
async def tts(tmp_path: Path) -> AsyncIterator[IrodoriTTS]:
    t = IrodoriTTS(fake_command(tmp_path / "w.log"), timeout_s=2.0, startup_timeout_s=10.0)
    yield t
    await t.close()


# ---------------------------------------------------------------- 文の準備


def test_prepare_text_prefixes_the_emotion_emoji() -> None:
    assert prepare_text(" こんにちは。 ") == "こんにちは。"
    assert prepare_text("やった！", "happy") == "😊やった！"
    assert prepare_text("えっ", "surprised") == "😲えっ"


def test_every_emotion_has_a_mapping_and_sad_is_not_crying() -> None:
    emotions: tuple[Emotion, ...] = (
        "neutral",
        "happy",
        "surprised",
        "thinking",
        "sad",
        "shy",
        "proud",
    )
    assert set(EMOTION_EMOJI) == set(emotions)
    assert EMOTION_EMOJI["neutral"] == ""
    assert EMOTION_EMOJI["sad"] != "😭"  # 泣き声になる


def test_worker_command_runs_in_the_irodori_environment() -> None:
    cmd = worker_command(
        "~/Irodori-TTS",
        checkpoint="Aratako/Irodori-TTS-v4.1-Small-MF",
        precision="fp32",
        ref_wav=None,
        caption="明るい声",
        num_steps=None,
        sample_rate=24000,
    )
    assert cmd[0].endswith("/Irodori-TTS/.venv/bin/python")
    assert "~" not in cmd[0]
    assert cmd[1].endswith("irodori_worker.py")
    assert cmd[cmd.index("--caption") + 1] == "明るい声"
    assert "--ref-wav" not in cmd and "--num-steps" not in cmd


# ---------------------------------------------------------------- プロトコル


async def test_synthesize_returns_wav_bytes(tts: IrodoriTTS) -> None:
    assert isinstance(tts, TTSProvider)
    wav = await tts.synthesize("こんにちは", "happy")
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.getnframes() == int(len("😊こんにちは") * 0.01 * 24000)  # 絵文字つきの文が届いた


async def test_sequential_requests_get_their_own_answers(tts: IrodoriTTS) -> None:
    def frames(wav: bytes) -> int:
        with wave.open(io.BytesIO(wav)) as w:
            return w.getnframes()

    results = await asyncio.gather(tts.synthesize("あ"), tts.synthesize("ああああ"))
    assert [frames(r) for r in results] == [int(1 * 0.01 * 24000), int(4 * 0.01 * 24000)]


async def test_warm_starts_once_and_is_shared(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    t = IrodoriTTS(fake_command(log), timeout_s=2.0)
    try:
        await asyncio.gather(t.warm(), t.warm(), t.synthesize("あ"))
        await t.synthesize("い")
        assert starts(log) == 1
    finally:
        await t.close()


async def test_a_failed_sentence_does_not_kill_the_worker(tts: IrodoriTTS, tmp_path: Path) -> None:
    with pytest.raises(WorkerError, match="boom"):
        await tts.synthesize("ERROR")
    assert await tts.synthesize("つぎ")
    assert starts(tmp_path / "w.log") == 1


# ---------------------------------------------------------------- 取消と復旧


async def test_cancel_drops_the_queued_sentence_and_keeps_later_ones_correct(
    tts: IrodoriTTS, tmp_path: Path
) -> None:
    log = tmp_path / "w.log"
    slow = asyncio.create_task(tts.synthesize("SLOW"))
    await asyncio.sleep(0.1)  # ワーカーは SLOW を合成中
    queued = asyncio.create_task(tts.synthesize("取り消す文"))
    await asyncio.sleep(0.05)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await slow
    after = await tts.synthesize("あと")
    assert after
    assert "取り消す文" in texts(log)  # ワーカーは受け取ったが、取消で合成せず捨てた
    assert starts(log) == 1  # 取消ではワーカーを再起動しない


async def test_cancelled_answer_is_not_mixed_into_the_next_request(tts: IrodoriTTS) -> None:
    first = asyncio.create_task(tts.synthesize("SLOW"))
    await asyncio.sleep(0.1)
    first.cancel()  # 応答は後から届く
    with pytest.raises(asyncio.CancelledError):
        await first
    wav = await tts.synthesize("ああ")  # 先の応答を取り違えない
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnframes() == int(2 * 0.01 * 24000)


async def test_crashed_worker_is_restarted_on_the_next_request(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    t = IrodoriTTS(fake_command(log), timeout_s=2.0)
    try:
        with pytest.raises(WorkerError):
            await t.synthesize("CRASH")
        assert await t.synthesize("復旧")
        assert starts(log) == 2
    finally:
        await t.close()


async def test_hung_worker_times_out_and_is_replaced(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    t = IrodoriTTS(fake_command(log), timeout_s=0.3)
    try:
        with pytest.raises(TimeoutError):
            await t.synthesize("HANG")
        assert await t.synthesize("復旧")
        assert starts(log) == 2
    finally:
        await t.close()


async def test_worker_that_exits_before_ready_fails_the_startup(tmp_path: Path) -> None:
    t = IrodoriTTS(fake_command(tmp_path / "w.log", "--no-ready"), startup_timeout_s=5.0)
    with pytest.raises(WorkerError):
        await t.warm()
    await t.close()


async def test_startup_timeout_kills_the_worker(tmp_path: Path) -> None:
    t = IrodoriTTS(fake_command(tmp_path / "w.log", "--startup-delay", "30"), startup_timeout_s=0.3)
    with pytest.raises(TimeoutError):
        await t.warm()
    await t.close()


async def test_close_stops_the_worker(tmp_path: Path) -> None:
    t = IrodoriTTS(fake_command(tmp_path / "w.log"))
    await t.warm()
    proc = t._worker.proc  # pyright: ignore[reportOptionalMemberAccess]
    await t.close()
    assert proc.returncode is not None


# ---------------------------------------------------------------- ワーカーの取消の記録


def test_cancel_set_drops_a_queued_request_once() -> None:
    c = CancelSet()
    c.add(3)
    assert c.take(2) is False
    assert c.take(3) is True
    assert c.take(3) is False  # 一度効いたら消える


def test_cancel_set_does_not_grow_with_cancels_that_arrive_during_synthesis() -> None:
    # 合成が始まってから届いた取消は take されない。処理し終えた時点で捨てる
    c = CancelSet()
    for rid in range(1, 1001):
        c.add(rid)  # 合成中に取消が届く
        c.done(rid)  # 処理し終えた
    assert len(c) == 0


def test_cancel_set_keeps_cancels_for_later_requests() -> None:
    c = CancelSet()
    c.add(5)
    c.add(9)
    c.done(5)  # 5 までは処理した。先の 9 の取消は残す
    assert len(c) == 1
    assert c.take(9) is True


# ---------------------------------------------------------------- 登録簿


def settings(**kw: Any) -> ServerSettings:
    return ServerSettings(  # pyright: ignore[reportCallIssue]
        token=SecretStr("x" * 16), tts="irodori", **kw
    )


def test_registry_builds_without_loading_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reachy_server import gpu

    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    picked: list[int] = []
    monkeypatch.setattr(
        gpu, "pin_free_gpu", lambda *, min_free_mib, env=None: picked.append(min_free_mib)
    )
    tts = build_providers(settings(tts_ref_wav=str(ref), gpu_min_free_mib=7000)).tts
    assert tts.name == "irodori"
    assert picked == [7000]  # 起動する前に GPU を選ぶ（子プロセスが環境を引き継ぐ）
    assert isinstance(tts, IrodoriTTS)
    assert tts._worker is None  # pyright: ignore[reportPrivateUsage]  # 読み込みは warm で


def test_registry_requires_a_voice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from reachy_server import gpu

    monkeypatch.setattr(gpu, "pin_free_gpu", lambda *, min_free_mib, env=None: None)
    with pytest.raises(FileNotFoundError, match="参照音声"):
        build_providers(settings(tts_ref_wav=str(tmp_path / "missing.wav")))
    build_providers(settings(tts_ref_wav=str(tmp_path / "missing.wav"), tts_caption="明るい声"))
