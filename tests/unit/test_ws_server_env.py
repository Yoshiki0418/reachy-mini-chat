"""ws_server は pipeline を import する前に .env を読み込む（#89）。

pipeline 以下はモジュール読み込み時に環境変数を読むので、順序が逆だと .env の設定が無視される。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PROBE = """
import os
import dotenv

def fake_load_dotenv(*args, **kwargs):
    # .env に書いてある想定の値
    os.environ["OPENAI_MODEL"] = "model-from-dotenv"
    os.environ["PERSON_MATCH_THRESHOLD"] = "0.5"
    return True

dotenv.load_dotenv = fake_load_dotenv
import pc_server.ws_server  # noqa: F401
from pc_server.agent import graph
from pc_server.person import store
print(graph.DEFAULT_MODEL, store.COSINE_THRESHOLD)
"""


def test_dotenv_is_loaded_before_pipeline_import() -> None:
    env = {"PATH": "", "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["model-from-dotenv", "0.5"]
