"""scripts/dev/check_secrets.py: Public リポジトリへ秘密情報・個人データを入れないためのガード。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_secrets.py"
_spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_secrets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_secrets)
forbidden_path_reason = check_secrets.forbidden_path_reason
scan_files = check_secrets.scan_files
scan_text = check_secrets.scan_text

# テスト用のダミー値は実在しない形で、文字列結合で組み立てる（このファイル自体が検出されないように）
FAKE_OPENAI = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
FAKE_GH = "ghp_" + "a" * 36


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        (f"OPENAI_API_KEY={FAKE_OPENAI}", "OpenAI API キー"),
        (f"token: {FAKE_GH}", "GitHub トークン"),
        ("-----BEGIN OPENSSH " + "PRIVATE KEY-----", "秘密鍵"),
        ("PC は 192." + "168.11.135", "LAN の IP アドレス"),
        ("PC_HOST=taros-Mac" + "Book-Air.local", "個人の端末名"),
        ("cd /Us" + "ers/taro/app", "ホームディレクトリのパス"),
        ("host 10." + "0.0.12", "LAN の IP アドレス"),
    ],
)
def test_detects_secrets(line: str, rule: str) -> None:
    assert scan_text(line) == [(1, rule)]


@pytest.mark.parametrize(
    "line",
    [
        "OPENAI_API_KEY=sk-...",  # プレースホルダ
        "localhost / 127.0.0.1 / 0.0.0.0",
        "バージョン 1.10.10.2 ではない",  # 先頭が私設 IP 帯でない
        "172.32.0.1 は私設帯の外",
    ],
)
def test_ignores_placeholders_and_public_ranges(line: str) -> None:
    assert scan_text(line) == []


def test_allow_marker_suppresses() -> None:
    assert scan_text(f"{FAKE_OPENAI}  # secret-scan: allow") == []


def test_reports_line_numbers() -> None:
    assert scan_text(f"ok\nok\n{FAKE_GH}\n") == [(3, "GitHub トークン")]


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "pc_server/.env",
        "data/people.json",
        "people.json.bak.1",
        "recordings/a.wav",
        "m/G_3700.pth",
    ],
)
def test_forbidden_paths(path: str) -> None:
    assert forbidden_path_reason(path) is not None


@pytest.mark.parametrize("path", [".env.example", "docs/DESIGN.md", "pc_server/person/store.py"])
def test_allowed_paths(path: str) -> None:
    assert forbidden_path_reason(path) is None


def test_scan_files(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text("x = 1\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text(f"KEY = '{FAKE_OPENAI}'\n", encoding="utf-8")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00")
    problems = scan_files([str(good), str(bad), str(binary), str(tmp_path / "missing.py")])
    assert problems == [f"{bad}:1: OpenAI API キー"]


def test_lockfile_content_is_not_scanned(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text('version = "' + "10.4" + '.0.35"\n', encoding="utf-8")
    assert scan_files([str(lock)]) == []


def test_repository_is_clean() -> None:
    """リポジトリ自体に秘密情報が入っていないこと（CI でも全ファイル検査する）。"""
    assert check_secrets.main(["--all"]) == 0
