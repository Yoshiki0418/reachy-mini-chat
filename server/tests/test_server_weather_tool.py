"""天気ツール（旧実装から移植）: Open-Meteo への通信は差し替えて検証する。"""

from __future__ import annotations

import io
import json
import urllib.parse
from typing import Any

import pytest

from reachy_server.agent.tools import weather

KANAZAWA = {"results": [{"latitude": 36.56, "longitude": 136.65, "name": "金沢市"}]}
CURRENT = {
    "current": {
        "temperature_2m": 12.3,
        "relative_humidity_2m": 70,
        "weather_code": 61,
        "wind_speed_10m": 8.1,
    }
}
DAILY = {
    "daily": {
        "weather_code": [0],
        "temperature_2m_max": [18.0],
        "temperature_2m_min": [7.5],
        "precipitation_probability_max": [10],
        "wind_speed_10m_max": [12.0],
    }
}


class FakeNet:
    """URL のホストと検索語で応答を選ぶ。None を返すと通信失敗にする。"""

    def __init__(self, geocode: dict[str, Any], forecast: dict[str, Any] | None) -> None:
        self.geocode = geocode
        self.forecast = forecast
        self.queries: list[str] = []

    def urlopen(self, req: Any, timeout: float = 0) -> io.BytesIO:
        url = req.full_url
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        if "geocoding" in url:
            self.queries.append(query["name"])
            body = self.geocode.get(query["name"], {"results": []})
        else:
            if self.forecast is None:
                raise OSError("network down")
            body = self.forecast
        return io.BytesIO(json.dumps(body).encode())


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def install(geocode: dict[str, Any], forecast: dict[str, Any] | None) -> FakeNet:
        fake = FakeNet(geocode, forecast)
        monkeypatch.setattr(weather.urllib.request, "urlopen", fake.urlopen)
        return fake

    return install


def ask(city: str, when: str = "今") -> str:
    return str(weather.get_weather.invoke({"city": city, "when": when}))


def test_current_weather_is_spoken_without_symbols(net) -> None:  # type: ignore[no-untyped-def]
    net({"金沢": KANAZAWA}, CURRENT)
    text = ask("金沢")
    assert text == "金沢市は今弱い雨で、気温は12.3度、湿度70パーセント、風は8.1キロくらいです。"
    assert "%" not in text and "°" not in text


def test_japanese_name_falls_back_to_romaji(net) -> None:  # type: ignore[no-untyped-def]
    fake = net({"Kanazawa": KANAZAWA}, CURRENT)
    assert ask("石川").startswith("石川は今")  # 表示名は元の日本語
    assert fake.queries == ["石川", "Kanazawa"]


def test_tomorrow_forecast(net) -> None:  # type: ignore[no-untyped-def]
    net({"金沢": KANAZAWA}, DAILY)
    text = ask("金沢", "明日")
    assert text.startswith("金沢市の明日の天気は快晴の予報で")
    assert "降水確率は10パーセント" in text


def test_unknown_city(net) -> None:  # type: ignore[no-untyped-def]
    net({}, CURRENT)
    assert ask("どこでもない町") == "「どこでもない町」の位置情報を取得できませんでした。"


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ("今", "の天気情報を取得できませんでした"),
        ("明日", "の明日の天気予報を取得できませんでした"),
    ],
)
def test_network_failure_is_reported(net, when: str, expected: str) -> None:  # type: ignore[no-untyped-def]
    net({"金沢": KANAZAWA}, None)
    assert expected in ask("金沢", when)


def test_empty_daily_is_reported(net) -> None:  # type: ignore[no-untyped-def]
    net({"金沢": KANAZAWA}, {"daily": {}})
    assert "天気予報を取得できませんでした" in ask("金沢", "明後日")


def test_geocode_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("dns")

    monkeypatch.setattr(weather.urllib.request, "urlopen", boom)
    assert "位置情報を取得できませんでした" in ask("金沢")
