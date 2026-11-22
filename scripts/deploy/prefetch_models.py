"""モデルを事前にダウンロードする（初回起動で数分待たされないように）。デプロイの手順で 1 回流す。

    uv run python scripts/deploy/prefetch_models.py

- STT: 設定の ``stt_model`` を Hugging Face から取得（GPU は使わない）
- TTS: Irodori-TTS のワーカーを 1 回起動し、チェックポイントやコーデックの取得とカーネルの準備まで
  済ませる（GPU を使う。共用機なので、起動時に空いている GPU を自分で選ぶ）
設定は ``REACHY_SERVER_*``（.env）と同じ。使わない側（stt / tts が fake）は飛ばす。
"""

from __future__ import annotations

import asyncio
import sys

from reachy_server.config import ServerSettings
from reachy_server.providers import build_providers


async def main() -> int:
    settings = ServerSettings()  # pyright: ignore[reportCallIssue]
    if settings.stt == "faster-whisper":
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

        print(f"STT: {settings.stt_model}", flush=True)
        snapshot_download(repo_id=settings.stt_model)
    else:
        print(f"STT は {settings.stt}: 飛ばす", flush=True)

    providers = build_providers(settings)
    warm = getattr(providers.tts, "warm", None)
    if warm is None:
        print(f"TTS は {settings.tts}: 飛ばす", flush=True)
        return 0
    print(f"TTS: {settings.tts_checkpoint}（ワーカーを起動して読み込む）", flush=True)
    try:
        await warm()
    finally:
        close = getattr(providers.tts, "close", None)
        if close is not None:
            await close()
    print("完了", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
