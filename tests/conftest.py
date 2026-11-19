"""共通フィクスチャ。

- 実機 SDK (`reachy_mini`) は CI に無いのでスタブを差し込む。
- `robot/` のスクリプトは実機上で「スクリプトのディレクトリ = sys.path」として動くので、
  テストでも同様に `robot/` を sys.path に追加する。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ROBOT_DIR = ROOT / "robot"


def _install_reachy_mini_stub() -> None:
    if "reachy_mini" in sys.modules:
        return
    try:
        import reachy_mini  # noqa: F401  # pyright: ignore[reportMissingImports]

        return
    except ImportError:
        pass
    pkg = types.ModuleType("reachy_mini")
    utils = types.ModuleType("reachy_mini.utils")

    class ReachyMini:  # 型注釈と import のためだけのダミー
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("reachy_mini stub: 実機 SDK はテストでは使えません")

    def create_head_pose(*args, **kwargs):
        return None

    pkg.ReachyMini = ReachyMini  # type: ignore[attr-defined]
    utils.create_head_pose = create_head_pose  # type: ignore[attr-defined]
    pkg.utils = utils  # type: ignore[attr-defined]
    sys.modules["reachy_mini"] = pkg
    sys.modules["reachy_mini.utils"] = utils


_install_reachy_mini_stub()
if str(ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOT_DIR))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """テストが実データ (data/people.json) や API キーに触れないようにする。"""
    monkeypatch.setenv("PERSON_DB_PATH", str(tmp_path / "people.json"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
