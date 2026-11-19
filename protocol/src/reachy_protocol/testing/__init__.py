"""契約テストの部品。設計書 §5。

- :class:`FakeServer`: プロトコル準拠の偽サーバー（ロボット側のテストで相手役にする）
- :class:`FakeRobotClient`: 偽ロボット（サーバー側のテストで相手役にする）
- :mod:`.scenarios`: どのサーバー実装にも流せる汎用シナリオと実行器
- :class:`MonitoredTransport`: 通信を横から観測して ConversationChecker に渡す

本物のサーバー（#15〜）・ロボット（#34〜）は、同じシナリオを流して契約を守っていることを確かめる。
"""

from __future__ import annotations

from .fake_robot import FakeRobotClient, TurnResult, default_context
from .fake_server import FakeServer, FakeServerConfig
from .scenarios import (
    CONVERSATION_SCENARIOS,
    HANDSHAKE_SCENARIOS,
    ServerHandler,
    run_conversation_scenario,
    run_handshake_scenario,
)
from .tap import MonitoredTransport

__all__ = [
    "CONVERSATION_SCENARIOS",
    "HANDSHAKE_SCENARIOS",
    "FakeRobotClient",
    "FakeServer",
    "FakeServerConfig",
    "MonitoredTransport",
    "ServerHandler",
    "TurnResult",
    "default_context",
    "run_conversation_scenario",
    "run_handshake_scenario",
]
