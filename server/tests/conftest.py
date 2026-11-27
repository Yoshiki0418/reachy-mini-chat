from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from reachy_server.config import ServerSettings
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers

TOKEN = "test-token-0123456789"


@pytest.fixture(autouse=True)
def _isolate_from_local_config(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """開発機の ``.env`` や環境変数（本番の設定）を、テストが読まないようにする。

    ``ServerSettings`` は cwd の ``.env`` と ``REACHY_SERVER_*`` を読む。サーバーを動かしている
    マシンでテストを流すと、本物のプロバイダの設定を拾って失敗するので、毎回まっさらにする。
    """
    monkeypatch.setitem(ServerSettings.model_config, "env_file", None)
    for name in [k for k in os.environ if k.startswith("REACHY_SERVER_")]:
        monkeypatch.delenv(name)


@pytest.fixture
def token() -> str:
    return TOKEN


@pytest.fixture
def settings(token: str) -> ServerSettings:
    # history_db="" はメモリ上の履歴（テストがリポジトリ内に DB を作らないように）
    return ServerSettings(token=SecretStr(token), handshake_timeout_s=0.5, history_db="")


@pytest.fixture
def providers() -> Providers:
    return Providers(stt=FakeSTT(), tts=FakeTTS(), llm=FakeLLM())
