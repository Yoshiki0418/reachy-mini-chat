"""FastAPI アプリ: /healthz と、本物の WebSocket 越しのハンドシェイク。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from reachy_protocol import PROTOCOL_VERSION
from reachy_protocol.messages import Hello, Ping, dump_message
from reachy_server import __version__
from reachy_server.app import create_app
from reachy_server.config import ServerSettings


@pytest.fixture
def client(settings: ServerSettings) -> TestClient:
    return TestClient(create_app(settings))


def test_healthz(client: TestClient) -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "server_version": __version__,
        "protocol_version": PROTOCOL_VERSION,
    }


def test_websocket_handshake_and_ping(client: TestClient, token: str) -> None:
    with client.websocket_connect("/v1/session") as ws:
        ws.send_text(dump_message(Hello(robot_id="reachy-1", token=SecretStr(token))))
        assert ws.receive_json()["type"] == "welcome"
        ws.send_text(dump_message(Ping(nonce="n1")))
        pong = ws.receive_json()
        assert (pong["type"], pong["nonce"]) == ("pong", "n1")


def test_websocket_rejects_bad_token(client: TestClient) -> None:
    with client.websocket_connect("/v1/session") as ws:
        ws.send_text(dump_message(Hello(robot_id="reachy-1", token=SecretStr("x" * 16))))
        assert ws.receive_json()["code"] == "unauthorized"


# ---------------------------------------------------------------- 設定


def test_token_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)  # 手元の .env を読まない
    monkeypatch.delenv("REACHY_SERVER_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        ServerSettings()  # pyright: ignore[reportCallIssue]


def test_short_token_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ServerSettings(token=SecretStr("short"))


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path, token: str) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REACHY_SERVER_TOKEN", token)
    monkeypatch.setenv("REACHY_SERVER_PORT", "9000")
    s = ServerSettings()  # pyright: ignore[reportCallIssue]
    assert s.port == 9000
    assert s.token.get_secret_value() == token
    assert s.masked()["token"] == "**********"
    assert token not in repr(s)
