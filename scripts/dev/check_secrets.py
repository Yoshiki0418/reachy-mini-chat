"""コミットしてはいけない秘密情報・個人データを検出する（Public リポジトリ用の簡易ガード）。

usage:
    python scripts/dev/check_secrets.py FILE...   # pre-commit から（ステージされたファイル）
    python scripts/dev/check_secrets.py --all     # CI から（git 管理下の全ファイル）

検出したら行番号つきで報告し、終了コード 1 を返す。
誤検知を許可したい行には `secret-scan: allow` を書く。
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

ALLOW_MARKER = "secret-scan: allow"

# (名前, パターン)。プレースホルダ（sk-... 等）に当たらないよう十分な長さを要求する。
CONTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI API キー", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Anthropic API キー", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    (
        "GitHub トークン",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    ),
    ("秘密鍵", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "LAN の IP アドレス",
        re.compile(r"\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    ),
]

# パスで禁止するもの（顔データ・人物 DB・環境変数・モデル重み）。
_FORBIDDEN_NAMES = {".env", "people.json"}
_FORBIDDEN_DIRS = {"data", "recordings"}
_FORBIDDEN_SUFFIXES = {".pth", ".onnx", ".npy", ".wav"}
_ALLOWED_NAMES = {".env.example"}
# 自動生成ファイルは内容を検査しない（4 つ組のパッケージ版番号が IP に誤マッチするため）
_SKIP_CONTENT_NAMES = {"uv.lock"}


def forbidden_path_reason(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.name in _ALLOWED_NAMES:
        return None
    if p.name in _FORBIDDEN_NAMES or p.name.startswith("people.json"):
        return f"コミット禁止のファイル名 ({p.name})"
    if p.parts and p.parts[0] in _FORBIDDEN_DIRS:
        return f"コミット禁止のディレクトリ ({p.parts[0]}/)"
    if p.suffix in _FORBIDDEN_SUFFIXES:
        return f"モデル重み・音声などのバイナリ ({p.suffix})"
    return None


def scan_text(text: str) -> list[tuple[int, str]]:
    """(行番号, ルール名) のリストを返す。"""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        for name, pattern in CONTENT_RULES:
            if pattern.search(line):
                hits.append((lineno, name))
    return hits


def scan_files(paths: Iterable[str]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        reason = forbidden_path_reason(path)
        if reason:
            problems.append(f"{path}: {reason}")
            continue
        if PurePosixPath(path).name in _SKIP_CONTENT_NAMES:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # バイナリ・削除済みは内容を見ない
        problems.extend(f"{path}:{lineno}: {name}" for lineno, name in scan_text(text))
    return problems


def _git_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def main(argv: list[str]) -> int:
    paths = _git_files() if argv == ["--all"] else argv
    problems = scan_files(paths)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print(
            f"\n秘密情報・個人データの可能性: {len(problems)} 件。"
            f"誤検知なら該当行に `{ALLOW_MARKER}` を書いてください。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
