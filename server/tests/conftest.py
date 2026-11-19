from __future__ import annotations

import pytest
from pydantic import SecretStr

from reachy_server.config import ServerSettings

TOKEN = "test-token-0123456789"


@pytest.fixture
def token() -> str:
    return TOKEN


@pytest.fixture
def settings(token: str) -> ServerSettings:
    return ServerSettings(token=SecretStr(token), handshake_timeout_s=0.5)
