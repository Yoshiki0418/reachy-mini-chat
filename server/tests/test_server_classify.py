"""宛先判定の補助 API（/v1/classify）。Fake LLM で検証する。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from reachy_protocol.classify import ClassifyRequest
from reachy_server.app import create_app
from reachy_server.classify import Classifier, build_input, parse_label
from reachy_server.config import ServerSettings
from reachy_server.providers import FakeLLM, FakeSTT, FakeTTS, Providers


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def client_with(settings: ServerSettings, llm: Any) -> TestClient:
    return TestClient(create_app(settings, Providers(stt=FakeSTT(), tts=FakeTTS(), llm=llm)))


@pytest.fixture
def client(settings: ServerSettings) -> Iterator[TestClient]:
    with client_with(settings, FakeLLM(replies=('{"label": "robot"}',))) as c:
        yield c


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"label": "robot"}', "robot"),
        ('{"label": "other"}', "other"),
        ('```json\n{"label": "unsure"}\n```', "unsure"),
        ('判定: {"label": "robot"} です', "robot"),
        ('{"label": "banana"}', "unsure"),  # 知らないラベル
        ("robot", "unsure"),  # JSON でない
        ("[1, 2]", "unsure"),
        ('{"label": 3}', "unsure"),
        ("", "unsure"),
    ],
)
def test_parse_label(raw: str, expected: str) -> None:
    assert parse_label(raw) == expected


def test_input_carries_context_but_prompt_does_not_leak_into_it() -> None:
    text = build_input(
        ClassifyRequest(text="お腹すいた", people_count=2, robot_asked_recently=True)
    )
    assert "お腹すいた" in text and "2" in text and "質問した" in text
    assert "質問した" not in build_input(ClassifyRequest(text="あ"))


def test_classify_returns_the_llm_label(client: TestClient, token: str) -> None:
    res = client.post("/v1/classify", json={"text": "リーチ、今日の天気は？"}, headers=auth(token))
    assert res.status_code == 200
    assert res.json() == {"label": "robot"}


def test_classify_requires_the_token(client: TestClient) -> None:
    assert client.post("/v1/classify", json={"text": "あ"}).status_code == 401


@pytest.mark.parametrize(
    "body", [{}, {"text": ""}, {"text": "あ" * 501}, {"text": "あ", "people_count": -1}]
)
def test_classify_rejects_bad_requests(
    client: TestClient, token: str, body: dict[str, Any]
) -> None:
    assert client.post("/v1/classify", json=body, headers=auth(token)).status_code == 422


class _Slow(BaseChatModel):
    delay: float = 5.0

    @property
    def _llm_type(self) -> str:
        return "slow"

    def _generate(self, messages: list[BaseMessage], *a: Any, **k: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(self, messages: list[BaseMessage], *a: Any, **k: Any) -> ChatResult:
        await asyncio.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage('{"label": "robot"}'))])


class _SlowLLM:
    name = "slow"

    def chat_model(self) -> BaseChatModel:
        return _Slow()


class _BrokenLLM:
    name = "broken"

    def chat_model(self) -> BaseChatModel:
        raise RuntimeError("boom")


async def test_timeout_gives_unsure_within_the_budget() -> None:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    res = await Classifier(_SlowLLM(), timeout_s=0.1).classify(ClassifyRequest(text="あ"))
    assert res.label == "unsure"
    assert loop.time() - t0 < 1.0  # LLM の 5 秒を待たない


async def test_llm_failure_gives_unsure() -> None:
    res = await Classifier(_BrokenLLM(), timeout_s=0.4).classify(ClassifyRequest(text="あ"))
    assert res.label == "unsure"


async def test_garbage_output_gives_unsure() -> None:
    res = await Classifier(FakeLLM(replies=("わかりません",))).classify(ClassifyRequest(text="あ"))
    assert res.label == "unsure"


async def test_utterance_text_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    secret = "ひみつの発話内容"
    with caplog.at_level(logging.DEBUG):
        await Classifier(FakeLLM(replies=('{"label": "other"}',))).classify(
            ClassifyRequest(text=secret)
        )
    assert secret not in caplog.text
    assert "other" in caplog.text
