"""``RobotIO`` の実機チェック。ロボット上で 1 コマンドで流し、確認の結果を表にして出す。

    /venvs/apps_venv/bin/python -m reachy_front check-io

サーバーには繋がない（ロボット単体で動く）。#35（実機結線）の前に、SDK との対応を確かめる。
動くのは小さな音（0.6 秒のビープ）と、首を左右に 10 度・アンテナを少し振るだけ。
結果の ``OK`` / ``NG`` は機械的な判定、``要確認`` は人が耳と目で見て判断する項目。
"""

from __future__ import annotations

import asyncio
import io
import math
import statistics
import struct
import time
import wave
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ..io.base import AudioFrame, HeadPose, RobotIO

Out = Callable[[str], None]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # "OK" / "NG" / "要確認"
    detail: str


def beep_wav(
    seconds: float = 0.6, freq: float = 440.0, sr: int = 24000, volume: float = 0.25
) -> bytes:
    n = int(seconds * sr)
    pcm = struct.pack(
        f"<{n}h", *(int(volume * 32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n))
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def _rms_peak(frames: list[AudioFrame]) -> tuple[float, float]:
    total, peak, n = 0.0, 0.0, 0
    for f in frames:
        samples = struct.unpack(f"<{f.n_samples}f", f.pcm_f32le)
        total += sum(x * x for x in samples)
        peak = max([peak, *(abs(x) for x in samples)])
        n += len(samples)
    return (math.sqrt(total / n) if n else 0.0), peak


async def _read_audio_for(
    rio: RobotIO, seconds: float, clock: Callable[[], float]
) -> list[AudioFrame]:
    frames: list[AudioFrame] = []
    end = clock() + seconds
    while clock() < end:
        f = await rio.read_audio()
        if f is None:
            await asyncio.sleep(0.01)
        else:
            frames.append(f)
    return frames


async def run_checks(
    rio: RobotIO,
    *,
    out: Out = print,
    listen_s: float = 2.0,
    doa_s: float = 3.0,
    doa_prep_s: float = 2.0,
    doa_directions: Sequence[str] = ("正面", "左", "右"),
    sleep: Sleep = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(name: str, status: str, detail: str) -> None:
        results.append(CheckResult(name, status, detail))
        out(f"[{status}] {name}: {detail}")

    async def guarded(name: str, fn: Callable[[], Awaitable[None]]) -> None:
        try:
            await fn()
        except Exception as e:
            add(name, "NG", f"例外 {type(e).__name__}: {e}")

    # 1. 開始
    async def start() -> None:
        await rio.start()
        sr = rio.input_sample_rate
        add("開始・マイクのサンプルレート", "OK" if sr > 0 else "NG", f"{sr} Hz")

    await guarded("開始", start)

    # 2. マイク
    async def mic() -> None:
        out(f"… マイクを {listen_s:.0f} 秒聞きます（静かにしてください）")
        frames = await _read_audio_for(rio, listen_s, clock)
        rms, peak = _rms_peak(frames)
        add(
            "マイクの読み取り",
            "OK" if frames else "NG",
            f"{len(frames)} チャンク、無音時の RMS={rms:.4f} / ピーク={peak:.3f}",
        )

    await guarded("マイクの読み取り", mic)

    # 3. DoA（方向ごとに話しかけて、角度と is_speech を方向の名前つきで記録する）
    async def doa() -> None:
        summary: list[str] = []
        for name in doa_directions:
            out(
                f"… 【{name}】から話しかけてください（{doa_prep_s:.0f} 秒後に {doa_s:.0f} 秒間、測ります）"
            )
            await sleep(doa_prep_s)
            angles: list[float] = []
            end = clock() + doa_s
            while clock() < end:
                await rio.read_audio()
                r = await rio.read_doa()
                if r is not None and r.speech and r.angle_deg is not None:
                    angles.append(r.angle_deg)
                await sleep(0.01)
            if angles:
                summary.append(
                    f"{name} {statistics.median(angles):.0f}°"
                    f"（{min(angles):.0f}〜{max(angles):.0f}、発話判定 {len(angles)} 回）"
                )
            else:
                summary.append(f"{name} 発話と判定されず")
        add(
            "DoA（方向ごとの角度）",
            "要確認",
            "、".join(summary)
            + "（方向と角度の対応が、SDK の座標系。DoA と顔の位置の対応付けに使う）",
        )

    await guarded("DoA", doa)

    # 4. カメラ
    async def camera() -> None:
        frame = await rio.read_frame()
        add(
            "カメラの読み取り",
            "OK" if frame is not None else "NG",
            f"{frame.width}x{frame.height}" if frame else "画像が取れない",
        )

    await guarded("カメラ", camera)

    # 5. スピーカー（再生が終わるまで待つ長さが、音の長さと合うか）
    async def speaker() -> None:
        wav = beep_wav(0.6)
        out("… ビープを 1 回鳴らします")
        await rio.set_mic_enabled(False)
        t0 = clock()
        try:
            await rio.play_wav(wav)
        finally:
            await rio.set_mic_enabled(True)  # 再生が失敗しても、あとの検査のためにマイクは戻す
        took = clock() - t0
        add(
            "再生（ビープ 0.6 秒）",
            "要確認",
            f"play_wav が戻るまで {took:.2f} 秒。音は聞こえましたか",
        )

    await guarded("スピーカー", speaker)

    # 6. 再生後にマイクが読めるか（再生の間は録音を止める運用のため）
    async def mic_after() -> None:
        frames = await _read_audio_for(rio, 1.0, clock)
        add(
            "再生後にマイクが戻る",
            "OK" if frames else "NG",
            f"{len(frames)} チャンク（0 なら、録音の止め方・再開の仕方を見直す）",
        )

    await guarded("再生後のマイク", mic_after)

    # 7. 再生の停止（割り込み）
    async def stop_playback() -> None:
        out("… 2 秒のビープを鳴らし、0.5 秒後に止めます")
        task = asyncio.ensure_future(rio.play_wav(beep_wav(2.0)))
        await sleep(0.5)
        await rio.stop_playback()
        add(
            "再生の停止",
            "要確認",
            "ビープは 0.5 秒付近で止まりましたか（止まらなければ stop_playback を見直す）",
        )
        await task
        frames = await _read_audio_for(rio, 0.5, clock)
        add("停止後もマイクと再生が使える", "OK" if frames else "NG", f"{len(frames)} チャンク")
        await rio.play_wav(beep_wav(0.3))  # 開け直した再生がまた鳴るか

    await guarded("再生の停止", stop_playback)

    # 8. 首・アンテナ
    async def head() -> None:
        out("… 首を左右に 10 度、アンテナを少し動かします")
        await rio.goto_pose(HeadPose(yaw=10.0), (10.0, -10.0), duration_s=0.5)
        await rio.goto_pose(HeadPose(yaw=-10.0), (-10.0, 10.0), duration_s=0.8)
        await rio.goto_pose(HeadPose(), (0.0, 0.0), duration_s=0.5)
        for i in range(10):
            await rio.set_pose(HeadPose(pitch=5.0 * math.sin(i / 2)), (0.0, 0.0))
            await sleep(0.05)
        await rio.goto_pose(HeadPose(), (0.0, 0.0), duration_s=0.3)
        add(
            "首・アンテナ",
            "要確認",
            "左右に動き、中立に戻りましたか（向きが逆なら yaw の符号を記録）",
        )

    await guarded("首・アンテナ", head)

    await guarded("終了", rio.stop)
    ng = [r for r in results if r.status == "NG"]
    out(f"\n結果: OK/要確認 {len(results) - len(ng)} 件、NG {len(ng)} 件")
    return results


__all__ = ["CheckResult", "beep_wav", "run_checks"]
