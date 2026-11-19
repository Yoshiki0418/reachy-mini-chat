"""天気ツール — Open-Meteo API (無料・APIキー不要)。

pc_server/agent/tools/weather_tool.py から移植（元は gakucho-ai-motion）。
音声読み上げ前提のため、記号を使わず「度/パーセント/キロ」で表現する。
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from langchain_core.tools import tool

# WMO Weather Code → 日本語天気表現
_WMO_DESCRIPTIONS: dict[int, str] = {
    0: "快晴",
    1: "おおむね晴れ",
    2: "一部曇り",
    3: "曇り",
    45: "霧",
    48: "着氷性の霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "強い霧雨",
    56: "弱い着氷性の霧雨",
    57: "強い着氷性の霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    66: "弱い着氷性の雨",
    67: "強い着氷性の雨",
    71: "弱い雪",
    73: "雪",
    75: "強い雪",
    77: "霧雪",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "激しいにわか雨",
    85: "弱いにわか雪",
    86: "激しいにわか雪",
    95: "雷雨",
    96: "雹を伴う雷雨",
    99: "激しい雹を伴う雷雨",
}

# when 文字列 → 日数オフセット
_WHEN_OFFSETS: dict[str, int] = {
    "今日": 0,
    "今": 0,
    "現在": 0,
    "明日": 1,
    "あした": 1,
    "明後日": 2,
    "あさって": 2,
    "明々後日": 3,
    "しあさって": 3,
}


# Open-Meteo のジオコーディングは日本語名だと結果が空になりやすい。
# 主要な日本の地名を英語表記へフォールバックするための最小辞書。
_JP_CITY_ROMAJI: dict[str, str] = {
    "東京": "Tokyo",
    "大阪": "Osaka",
    "京都": "Kyoto",
    "名古屋": "Nagoya",
    "横浜": "Yokohama",
    "神戸": "Kobe",
    "札幌": "Sapporo",
    "福岡": "Fukuoka",
    "仙台": "Sendai",
    "広島": "Hiroshima",
    "金沢": "Kanazawa",
    "石川": "Kanazawa",
    "那覇": "Naha",
    "沖縄": "Naha",
    "北海道": "Sapporo",
    "奈良": "Nara",
    "神奈川": "Yokohama",
    "千葉": "Chiba",
    "埼玉": "Saitama",
    "新潟": "Niigata",
}


def _geocode_once(name: str) -> tuple[float, float, str] | None:
    params = urllib.parse.urlencode({"name": name, "count": 1, "language": "ja", "format": "json"})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReachyMini/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    results = data.get("results")
    if not results:
        return None
    r = results[0]
    return (r["latitude"], r["longitude"], r.get("name", name))


def _geocode(city: str) -> tuple[float, float, str] | None:
    # まず与えられた名前で検索し、ダメなら日本語→英語表記へフォールバック。
    hit = _geocode_once(city)
    if hit is not None:
        return hit
    key = city.strip().rstrip("市都府県")
    romaji = _JP_CITY_ROMAJI.get(city.strip()) or _JP_CITY_ROMAJI.get(key)
    if romaji is not None:
        alt = _geocode_once(romaji)
        if alt is not None:
            # 表示名は元の日本語名を優先
            return (alt[0], alt[1], city.strip())
    return None


def _fetch_current(lat: float, lon: float) -> dict | None:
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "Asia/Tokyo",
            "forecast_days": 1,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReachyMini/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return data.get("current")


def _fetch_daily(lat: float, lon: float, target_date: str) -> dict | None:
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
            "timezone": "Asia/Tokyo",
            "start_date": target_date,
            "end_date": target_date,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ReachyMini/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    daily = data.get("daily")
    if not daily:
        return None
    return {
        "weather_code": daily["weather_code"][0],
        "temp_max": daily["temperature_2m_max"][0],
        "temp_min": daily["temperature_2m_min"][0],
        "precip_prob": daily["precipitation_probability_max"][0],
        "wind_max": daily["wind_speed_10m_max"][0],
    }


@tool
def get_weather(city: str, when: str = "今") -> str:
    """指定した都市の天気情報を取得します。東京・大阪・金沢など日本各地や世界の都市に対応。

    Args:
        city: 都市名（例: 東京、大阪、金沢）
        when: いつの天気か（今、今日、明日、明後日）。デフォルトは「今」
    """
    geo = _geocode(city)
    if geo is None:
        return f"「{city}」の位置情報を取得できませんでした。"

    lat, lon, name = geo
    offset = _WHEN_OFFSETS.get(when.strip(), 0)

    if offset == 0:
        current = _fetch_current(lat, lon)
        if current is None:
            return f"「{name}」の天気情報を取得できませんでした。"
        desc = _WMO_DESCRIPTIONS.get(current.get("weather_code", -1), "不明")
        return (
            f"{name}は今{desc}で、気温は{current.get('temperature_2m', '不明')}度、"
            f"湿度{current.get('relative_humidity_2m', '不明')}パーセント、"
            f"風は{current.get('wind_speed_10m', '不明')}キロくらいです。"
        )

    target = datetime.now() + timedelta(days=offset)
    daily = _fetch_daily(lat, lon, target.strftime("%Y-%m-%d"))
    if daily is None:
        return f"「{name}」の{when}の天気予報を取得できませんでした。"
    desc = _WMO_DESCRIPTIONS.get(daily["weather_code"], "不明")
    when_label = (
        when if when in ("明日", "明後日", "明々後日") else f"{target.month}月{target.day}日"
    )
    return (
        f"{name}の{when_label}の天気は{desc}の予報で、"
        f"最高気温{daily['temp_max']}度、最低気温{daily['temp_min']}度、"
        f"降水確率は{daily['precip_prob']}パーセント、"
        f"風は最大{daily['wind_max']}キロくらいの見込みです。"
    )
