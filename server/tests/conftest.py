from __future__ import annotations

import pytest
from pydantic import SecretStr

from reachy_server.config import ServerSettings
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers

TOKEN = "test-token-0123456789"


@pytest.fixture
def token() -> str:
    return TOKEN


@pytest.fixture
def settings(token: str) -> ServerSettings:
    return ServerSettings(token=SecretStr(token), handshake_timeout_s=0.5)


@pytest.fixture
def providers() -> Providers:
    return Providers(stt=FakeSTT(), tts=FakeTTS(), llm=FakeLLM())
