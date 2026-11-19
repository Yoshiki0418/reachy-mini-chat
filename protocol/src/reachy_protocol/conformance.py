"""会話の順序規約（プロトコルの「契約」のうち、メッセージ単体の型では表せない部分）。

:class:`ConversationChecker` に、1 本の接続で流れたメッセージとバイナリフレームを
順に ``observe()`` させ、最後に ``finish()`` を呼ぶと、規約違反の一覧が得られる。
契約テスト（:mod:`reachy_protocol.testing`）のほか、実機の記録の検査にも使える。

観測点は 1 か所（通常はロボット側）を想定する。ネットワーク上の遅延によって生じうる
正当な行き違い（例: サーバーが turn.end を送った直後にロボットが turn.cancel を送る）は
違反にしない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .channel import FRAME_KINDS_FROM, BinaryFrame, Incoming
from .messages import (
    ROBOT_TO_SERVER,
    SERVER_TO_ROBOT,
    Error,
    FloorDecision,
    Hello,
    MemoryAck,
    MemoryOp,
    Motion,
    Ping,
    Pong,
    ProactiveRequest,
    Sentence,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCancel,
    TurnEnd,
    TurnStart,
    UtteranceBegin,
    UtteranceEnd,
    Welcome,
)

__all__ = ["ConversationChecker", "Direction", "ProtocolViolation"]

Direction: TypeAlias = Literal["robot->server", "server->robot"]


class ProtocolViolation(AssertionError):
    """順序規約の違反。``violations`` に全件が入る。"""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("プロトコル違反:\n" + "\n".join(f"  - {v}" for v in violations))


@dataclass
class _Utterance:
    open: bool = True
    next_chunk: int = 0
    transcribed: bool = False
    decision: str | None = None


@dataclass
class _Turn:
    ref: str
    active: bool = True
    next_idx: int = 0
    audio_sent: set[int] = field(default_factory=set[int])
    cancel_requested: bool = False
    status: str | None = None


class ConversationChecker:
    """1 本の接続のやり取りを観測し、順序規約の違反を集める。"""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self._hello = False
        self._welcome = False
        self._utts: dict[str, _Utterance] = {}
        self._intents: set[str] = set()
        self._turns: dict[str, _Turn] = {}
        self._started_refs: set[str] = set()
        self._memory_ops: dict[str, bool] = {}  # op_id -> acked
        self._tool_calls: dict[str, bool] = {}  # call_id -> answered
        self._pings: dict[Direction, set[str]] = {"robot->server": set(), "server->robot": set()}
        self._errored_refs: set[str] = set()
        self._finished = False

    # ------------------------------------------------------------ 公開 API

    def observe(self, direction: Direction, item: Incoming) -> None:
        """1 件観測する。違反は ``violations`` に追記する（例外は投げない）。"""
        if isinstance(item, BinaryFrame):
            self._on_binary(direction, item)
        else:
            self._on_message(direction, item)

    def finish(self) -> list[str]:
        """接続の終了時に呼ぶ。応答されないままの義務を違反として加え、全違反を返す。"""
        if not self._finished:
            self._finished = True
            self._check_dangling()
        return list(self.violations)

    def assert_ok(self) -> None:
        """``finish()`` して、違反があれば ProtocolViolation を送出する。"""
        violations = self.finish()
        if violations:
            raise ProtocolViolation(violations)

    # ------------------------------------------------------------ 内部

    def _bad(self, text: str) -> None:
        self.violations.append(text)

    def _on_binary(self, direction: Direction, frame: BinaryFrame) -> None:
        h = frame.header
        sender = "robot" if direction == "robot->server" else "server"
        if h.kind not in FRAME_KINDS_FROM[sender]:
            self._bad(f"{sender} が binary {h.kind} を送った")
            return
        if not self._handshaken(direction, f"binary {h.kind}"):
            return
        if h.kind == "audio_in":
            utt = self._utts.get(h.ref)
            if utt is None or not utt.open:
                self._bad(f"audio_in(ref={h.ref}) が開いていない発話に送られた")
                return
            if h.seq != utt.next_chunk:
                self._bad(f"audio_in(ref={h.ref}) の seq が {utt.next_chunk} でなく {h.seq}")
            utt.next_chunk = h.seq + 1
        elif h.kind == "frame":
            if h.ref not in self._utts and h.ref not in self._intents:
                self._bad(f"frame(ref={h.ref}) が未知の発話・自発発話を参照している")
        else:  # audio_out
            turn = self._active_turn(h.ref, "audio_out")
            if turn is None:
                return
            if h.seq >= turn.next_idx:
                self._bad(f"audio_out(turn={h.ref}, seq={h.seq}) に対応する sentence がまだない")
            elif h.seq in turn.audio_sent:
                self._bad(f"audio_out(turn={h.ref}, seq={h.seq}) が重複している")
            turn.audio_sent.add(h.seq)

    def _handshaken(self, direction: Direction, what: str) -> bool:
        if direction == "robot->server" and not self._hello:
            self._bad(f"hello の前に {what} を送った")
            return False
        if direction == "server->robot" and not self._welcome:
            self._bad(f"welcome の前に {what} を送った")
            return False
        return True

    def _active_turn(self, turn_id: str, what: str) -> _Turn | None:
        turn = self._turns.get(turn_id)
        if turn is None:
            self._bad(f"{what} が未知のターン {turn_id} を参照している")
            return None
        if not turn.active:
            self._bad(f"{what} が終了済みのターン {turn_id} に送られた")
            return None
        return turn

    def _on_message(self, direction: Direction, msg: object) -> None:
        mtype: str = getattr(msg, "type", "?")
        allowed = ROBOT_TO_SERVER if direction == "robot->server" else SERVER_TO_ROBOT
        if mtype not in allowed:
            self._bad(f"{direction} に {mtype} は送れない")
            return

        # --- ハンドシェイク
        if isinstance(msg, Hello):
            if self._hello:
                self._bad("hello が 2 回送られた")
            self._hello = True
            return
        if isinstance(msg, Welcome):
            if not self._hello:
                self._bad("hello の前に welcome を送った")
            if self._welcome:
                self._bad("welcome が 2 回送られた")
            self._welcome = True
            return
        if isinstance(msg, Error):
            # エラーはいつ送ってもよい（認証失敗など）。ref の義務を免除する
            if msg.ref:
                self._errored_refs.add(msg.ref)
            return
        if not self._handshaken(direction, mtype):
            return

        # --- 発話と判定
        if isinstance(msg, UtteranceBegin):
            if msg.utt_id in self._utts:
                self._bad(f"utterance.begin の utt_id {msg.utt_id} が重複している")
                return
            self._utts[msg.utt_id] = _Utterance()
        elif isinstance(msg, UtteranceEnd):
            utt = self._utts.get(msg.utt_id)
            if utt is None or not utt.open:
                self._bad(f"utterance.end(utt={msg.utt_id}) に対応する開いた発話がない")
                return
            utt.open = False
        elif isinstance(msg, Transcript):
            utt = self._utts.get(msg.utt_id)
            if utt is None or utt.open:
                self._bad(f"transcript(utt={msg.utt_id}) が終了していない発話に送られた")
            elif utt.transcribed:
                self._bad(f"transcript(utt={msg.utt_id}) が 2 回送られた")
            else:
                utt.transcribed = True
        elif isinstance(msg, FloorDecision):
            utt = self._utts.get(msg.utt_id)
            if utt is None or not utt.transcribed:
                self._bad(f"floor.decision(utt={msg.utt_id}) が文字起こし前に送られた")
            elif utt.decision is not None:
                self._bad(f"floor.decision(utt={msg.utt_id}) が 2 回送られた")
            else:
                utt.decision = msg.decision
        elif isinstance(msg, ProactiveRequest):
            if msg.intent_id in self._intents or msg.intent_id in self._utts:
                self._bad(f"proactive.request の intent_id {msg.intent_id} が重複している")
            self._intents.add(msg.intent_id)

        # --- ターン
        elif isinstance(msg, TurnStart):
            self._on_turn_start(msg)
        elif isinstance(msg, Sentence):
            turn = self._active_turn(msg.turn_id, "sentence")
            if turn is not None:
                if msg.idx != turn.next_idx:
                    self._bad(
                        f"sentence(turn={msg.turn_id}) の idx が {turn.next_idx} でなく {msg.idx}"
                    )
                turn.next_idx = max(turn.next_idx, msg.idx + 1)
        elif isinstance(msg, Motion):
            self._active_turn(msg.turn_id, "motion")
        elif isinstance(msg, TurnCancel):
            turn = self._turns.get(msg.turn_id)
            if turn is None:
                self._bad(f"turn.cancel が未知のターン {msg.turn_id} を参照している")
            else:
                # 終了済みへの取消は行き違いとして許す
                turn.cancel_requested = True
        elif isinstance(msg, TurnEnd):
            turn = self._active_turn(msg.turn_id, "turn.end")
            if turn is not None:
                turn.active = False
                turn.status = msg.status

        # --- 記憶・ツール・死活監視
        elif isinstance(msg, MemoryOp):
            if msg.op_id in self._memory_ops:
                self._bad(f"memory.op の op_id {msg.op_id} が重複している")
            self._memory_ops[msg.op_id] = False
        elif isinstance(msg, MemoryAck):
            acked = self._memory_ops.get(msg.op_id)
            if acked is None:
                self._bad(f"memory.ack が未知の op_id {msg.op_id} を参照している")
            elif acked:
                self._bad(f"memory.ack(op={msg.op_id}) が 2 回送られた")
            else:
                self._memory_ops[msg.op_id] = True
        elif isinstance(msg, ToolCall):
            if msg.call_id in self._tool_calls:
                self._bad(f"tool.call の call_id {msg.call_id} が重複している")
            self._tool_calls[msg.call_id] = False
        elif isinstance(msg, ToolResult):
            answered = self._tool_calls.get(msg.call_id)
            if answered is None:
                self._bad(f"tool.result が未知の call_id {msg.call_id} を参照している")
            elif answered:
                self._bad(f"tool.result(call={msg.call_id}) が 2 回送られた")
            else:
                self._tool_calls[msg.call_id] = True
        elif isinstance(msg, Ping):
            # 応答待ちの nonce を再利用されると、1 つの pong で両方が消えて取りこぼすので違反にする
            if msg.nonce in self._pings[direction]:
                self._bad(f"{direction} の ping の nonce {msg.nonce} が応答待ちのまま再利用された")
            self._pings[direction].add(msg.nonce)
        elif isinstance(msg, Pong):
            opposite: Direction = (
                "server->robot" if direction == "robot->server" else "robot->server"
            )
            if msg.nonce not in self._pings[opposite]:
                self._bad(f"pong(nonce={msg.nonce}) に対応する ping がない")
            self._pings[opposite].discard(msg.nonce)
        # PersonForget / SessionClose は順序の制約なし

    def _on_turn_start(self, msg: TurnStart) -> None:
        if msg.turn_id in self._turns:
            self._bad(f"turn.start の turn_id {msg.turn_id} が重複している")
            return
        utt = self._utts.get(msg.ref)
        if utt is not None:
            if utt.decision != "respond":
                self._bad(
                    f"turn.start(ref={msg.ref}) は respond と判定されていない発話への応答"
                    f"（判定: {utt.decision}）"
                )
        elif msg.ref not in self._intents:
            self._bad(f"turn.start が未知の ref {msg.ref} を参照している")
        if msg.ref in self._started_refs:
            self._bad(f"ref {msg.ref} に対してターンが 2 回始まった")
        self._started_refs.add(msg.ref)
        self._turns[msg.turn_id] = _Turn(ref=msg.ref)

    def _check_dangling(self) -> None:
        exempt = self._errored_refs
        for utt_id, utt in self._utts.items():
            if utt_id in exempt:
                continue
            if not utt.open and not utt.transcribed:
                self._bad(f"発話 {utt_id} に transcript が返らなかった")
            if utt.transcribed and utt.decision is None:
                self._bad(f"発話 {utt_id} に floor.decision が送られなかった")
            if utt.decision == "respond" and utt_id not in self._started_refs:
                self._bad(f"respond と判定した発話 {utt_id} にターンが始まらなかった")
        for intent_id in self._intents:
            if intent_id not in exempt and intent_id not in self._started_refs:
                self._bad(f"自発発話の依頼 {intent_id} にターンが始まらなかった")
        for turn_id, turn in self._turns.items():
            if turn.active and turn_id not in exempt:
                self._bad(f"ターン {turn_id} が turn.end で閉じられなかった")
            # 取消・エラーで終わったターンは、文を送った後に音声を破棄してよい
            completed = turn.status == "ok" and not turn.cancel_requested
            missing = set(range(turn.next_idx)) - turn.audio_sent
            if completed and missing and turn_id not in exempt:
                self._bad(f"ターン {turn_id} の sentence {sorted(missing)} に audio_out がない")
        for op_id, acked in self._memory_ops.items():
            if not acked and op_id not in exempt:
                self._bad(f"memory.op {op_id} に memory.ack が返らなかった")
        for call_id, answered in self._tool_calls.items():
            if not answered and call_id not in exempt:
                self._bad(f"tool.call {call_id} に tool.result が返らなかった")
        for direction, nonces in self._pings.items():
            for nonce in nonces:
                self._bad(f"{direction} の ping(nonce={nonce}) に pong が返らなかった")
