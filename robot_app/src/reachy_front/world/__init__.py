"""イベントを畳み込んだ状況スナップショット（在室者・話者・ロボット状態・会話状態）。設計書 §3.1。"""

from __future__ import annotations

from .reducer import reduce
from .state import ConversationState, RobotState, TrackState, WorldState
from .world import World

__all__ = ["ConversationState", "RobotState", "TrackState", "World", "WorldState", "reduce"]
