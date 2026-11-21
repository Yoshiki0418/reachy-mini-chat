"""Irodori-TTS の TTS プロバイダ。設計書 §4.3・docs/benchmarks.md。

Irodori-TTS は依存の版がこのパッケージの lock と合わないので、専用の Python 環境でワーカー
（:mod:`.irodori_worker`）を子プロセスとして起動し、標準入出力でつなぐ。ここは標準ライブラリだけ。

- 合成は 1 文ずつ。ワーカーは 1 つのモデルを常駐させ、要求を順に処理する。
- 取消（タスクのキャンセル）は、ワーカーに伝えて待ち行列から外す。
- ワーカーが落ちた・止まったときは、殺して次の要求で起こし直す。
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reachy_protocol.messages import Emotion

log = logging.getLogger(__name__)

# 感情タグ → Irodori-TTS の絵文字（EMOJI_ANNOTATIONS.md）。文頭に付けて話し方を寄せる
EMOTION_EMOJI: dict[Emotion, str] = {
    "neutral": "",
    "happy": "😊",  # 楽しげに、嬉しそうに
    "surprised": "😲",  # 驚き、感嘆
    "thinking": "🤔",  # 疑問の声
    "sad": "😟",  # 心配そうに（😭 は泣き声になるので避ける）
    "shy": "🫣",  # 恥ずかしそうに、照れながら
    "proud": "😎",  # 得意げに、自信ありげに
}


def prepare_text(text: str, emotion: Emotion = "neutral") -> str:
    """感情に合う絵文字を文頭に付ける。"""
    return EMOTION_EMOJI.get(emotion, "") + text.strip()


def worker_command(
    irodori_dir: str,
    *,
    checkpoint: str,
    precision: str,
    ref_wav: str | None,
    caption: str | None,
    num_steps: int | None,
    sample_rate: int,
    python: str | None = None,
) -> list[str]:
    """Irodori-TTS の環境の Python でワーカーを起動するコマンド。"""
    root = Path(irodori_dir).expanduser()
    cmd = [
        python or str(root / ".venv" / "bin" / "python"),
        str(Path(__file__).with_name("irodori_worker.py")),
        "--irodori-dir",
        str(root),
        "--checkpoint",
        checkpoint,
        "--precision",
        precision,
        "--sample-rate",
        str(sample_rate),
    ]
    if ref_wav:
        cmd += ["--ref-wav", str(Path(ref_wav).expanduser().resolve())]
    if caption:
        cmd += ["--caption", caption]
    if num_steps is not None:
        cmd += ["--num-steps", str(num_steps)]
    return cmd


@dataclass
class _Worker:
    proc: asyncio.subprocess.Process
    pending: dict[int, asyncio.Future[tuple[dict[str, Any], bytes]]] = field(default_factory=dict)
    reader: asyncio.Task[None] | None = None


class WorkerError(RuntimeError):
    pass


class IrodoriTTS:
    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 10.0,
        startup_timeout_s: float = 300.0,
    ) -> None:
        self.name = "irodori"
        self._command = list(command)
        self._env = None if env is None else dict(env)
        self._timeout_s = timeout_s
        self._startup_timeout_s = startup_timeout_s
        self._worker: _Worker | None = None
        self._start_lock = asyncio.Lock()
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- 起動と終了

    async def warm(self) -> None:
        """ワーカーを起動し、モデルを読み込み終えるまで待つ（サーバー起動時に呼ぶ）。"""
        await self._ensure()

    async def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            await self._stop(worker)

    async def _ensure(self) -> _Worker:
        async with self._start_lock:
            worker = self._worker
            if worker is not None and worker.proc.returncode is None:
                return worker
            self._worker = await self._spawn()
            return self._worker

    async def _spawn(self) -> _Worker:
        log.info("Irodori-TTS のワーカーを起動: %s", self._command[1:3])
        proc = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,  # ワーカーのログはそのまま親の標準エラーへ
            env=None if self._env is None else {**os.environ, **self._env},
        )
        worker = _Worker(proc)
        try:
            assert proc.stdout is not None
            line = await asyncio.wait_for(proc.stdout.readline(), self._startup_timeout_s)
            if not line or not json.loads(line).get("ready"):
                raise WorkerError("ワーカーが起動を完了する前に終了した")
        except BaseException:
            await self._stop(worker)
            raise
        worker.reader = asyncio.create_task(self._read_loop(worker))
        log.info("Irodori-TTS のワーカーが準備できた")
        return worker

    async def _stop(self, worker: _Worker) -> None:
        if worker.proc.returncode is None:
            worker.proc.kill()
        await worker.proc.wait()
        if worker.reader is not None:
            worker.reader.cancel()
        self._fail_pending(worker, WorkerError("ワーカーを止めた"))

    @staticmethod
    def _fail_pending(worker: _Worker, exc: Exception) -> None:
        for fut in worker.pending.values():
            if not fut.done():
                fut.set_exception(exc)
        worker.pending.clear()

    async def _read_loop(self, worker: _Worker) -> None:
        stdout = worker.proc.stdout
        assert stdout is not None
        try:
            while line := await stdout.readline():
                header = json.loads(line)
                nbytes = int(header.get("nbytes", 0))
                body = await stdout.readexactly(nbytes) if nbytes else b""
                fut = worker.pending.pop(int(header["id"]), None)
                if fut is not None and not fut.done():  # 取消済みの応答は捨てる
                    fut.set_result((header, body))
        except (asyncio.IncompleteReadError, ValueError, KeyError):
            log.exception("ワーカーの応答を読めなかった")
        finally:
            self._fail_pending(worker, WorkerError("ワーカーが終了した"))

    # ---------------------------------------------------------------- 合成

    async def synthesize(self, text: str, emotion: Emotion = "neutral") -> bytes:
        worker = await self._ensure()
        rid = next(self._ids)
        fut: asyncio.Future[tuple[dict[str, Any], bytes]] = (
            asyncio.get_running_loop().create_future()
        )
        worker.pending[rid] = fut
        stdin = worker.proc.stdin
        assert stdin is not None
        try:
            request = {"id": rid, "text": prepare_text(text, emotion)}
            stdin.write(json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n")
            await stdin.drain()
            header, body = await asyncio.wait_for(fut, self._timeout_s)
        except asyncio.CancelledError:
            self._send_cancel(worker, rid)  # 始めていなければ、ワーカーが捨てる
            raise
        except TimeoutError:
            log.error("合成が %.1f 秒を超えた。ワーカーを再起動する", self._timeout_s)
            await self._discard(worker)
            raise
        except (BrokenPipeError, ConnectionResetError) as e:
            await self._discard(worker)
            raise WorkerError("ワーカーに送れなかった") from e
        finally:
            worker.pending.pop(rid, None)
        if not header.get("ok"):
            raise WorkerError(str(header.get("error", "unknown error")))
        return body

    def _send_cancel(self, worker: _Worker, rid: int) -> None:
        stdin = worker.proc.stdin
        if stdin is None or worker.proc.returncode is not None:
            return
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            stdin.write(json.dumps({"cancel": rid}).encode("utf-8") + b"\n")

    async def _discard(self, worker: _Worker) -> None:
        """止まった・壊れたワーカーを捨てる。次の要求で起こし直す。"""
        if self._worker is worker:
            self._worker = None
        await self._stop(worker)


__all__ = ["EMOTION_EMOJI", "IrodoriTTS", "WorkerError", "prepare_text", "worker_command"]
