"""ロボット ⇄ サーバー間の制御メッセージ（WebSocket の text フレーム）。設計書 §2。

すべてのメッセージは ``type`` フィールドで判別する。受信側は方向つきの
:func:`parse_from_robot` / :func:`parse_from_server` で検証して読む。

互換性の方針（:mod:`reachy_protocol.version`）:
- 未知のフィールドは無視する（新しいマイナー版の相手から来ても壊れない）
- メッセージはイミュータブル
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    TypeAdapter,
    ValidationError,
    field_serializer,
    model_validator,
)

from .version import PROTOCOL_VERSION

__all__ = [
    "ROBOT_TO_SERVER",
    "SERVER_TO_ROBOT",
    "Emotion",
    "Error",
    "Familiarity",
    "FloorDecision",
    "Hello",
    "MemoryAck",
    "MemoryOp",
    "Message",
    "MessageDirectionError",
    "Motion",
    "PersonForget",
    "PersonProfile",
    "Ping",
    "Pong",
    "ProactiveRequest",
    "RobotMessage",
    "Sentence",
    "ServerMessage",
    "SessionClose",
    "Situation",
    "ToolCall",
    "ToolResult",
    "Transcript",
    "TurnCancel",
    "TurnContext",
    "TurnEnd",
    "TurnStart",
    "UtteranceBegin",
    "UtteranceEnd",
    "UtteranceFeatures",
    "ValidationError",
    "Welcome",
    "dump_message",
    "parse_from_robot",
    "parse_from_server",
    "parse_message",
]

# ---------------------------------------------------------------- 共通の型

Id = Annotated[str, Field(min_length=1, max_length=64)]
"""utt_id / turn_id / op_id などの識別子。発行側が一意に採番する。"""

ShortText = Annotated[str, Field(max_length=2000)]

Emotion: TypeAlias = Literal["neutral", "happy", "surprised", "thinking", "sad", "shy", "proud"]
"""文ごとの感情タグ。ロボットの ExpressionPlanner がモーションに変換する。"""

Familiarity: TypeAlias = Literal[
    "first_time", "new", "daily", "regular", "long_absence", "returning_today"
]
"""来訪履歴から算出する親密度区分（設計書 §3.5）。"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Msg(_Model):
    ts: float = Field(default_factory=time.time, description="送信時刻（UNIX 秒）")


# ---------------------------------------------------------------- 文脈（ロボットが毎ターン添付する）


class PersonProfile(_Model):
    """相手のプロフィール。人物の記憶の正本はロボット側にある。"""

    id: Id = Field(description="ロボットが採番した不透明な person_id")
    name: str | None = Field(default=None, max_length=32)
    familiarity: Familiarity
    visit_count: int = Field(ge=0)
    streak_days: int = Field(default=0, ge=0, description="連続来訪日数")
    days_absent: int | None = Field(default=None, ge=0, description="前回来訪からの日数")
    facts: list[Annotated[str, Field(max_length=200)]] = Field(default_factory=list, max_length=20)
    last_summary: str | None = Field(default=None, max_length=1000)


class Situation(_Model):
    people_count: int = Field(ge=0)
    local_time: AwareDatetime = Field(
        description="ロボット設置場所の現地時刻。タイムゾーン必須（受け手で解釈がぶれないように）"
    )
    others_present: list[Annotated[str, Field(max_length=32)]] = Field(
        default_factory=list, max_length=20, description="同席している既知の人の名前"
    )
    first_visitor_today: bool = False


class TurnContext(_Model):
    person: PersonProfile | None = Field(default=None, description="同定できなければ None")
    situation: Situation


class UtteranceFeatures(_Model):
    """発話区間についてロボットが観測した特徴（宛先判定の材料。サーバーはログ用に受け取る）。"""

    duration_s: float = Field(ge=0)
    eye_contact_ratio: float = Field(ge=0, le=1, description="発話区間のうち正面を向いていた割合")
    people_count: int = Field(ge=0)
    speaker_person_id: Id | None = None


# ---------------------------------------------------------------- robot → server


class Hello(_Msg):
    type: Literal["hello"] = "hello"
    protocol_version: str = PROTOCOL_VERSION
    robot_id: Id
    token: SecretStr = Field(description="事前共有トークン。repr やログには出ない")
    capabilities: list[str] = Field(default_factory=list)

    @field_serializer("token", when_used="json")
    def _reveal_token_on_wire(self, token: SecretStr) -> str:
        # 送信（JSON 化）のときだけ平文にする。repr / model_dump() では伏せたまま
        return token.get_secret_value()


class UtteranceBegin(_Msg):
    type: Literal["utterance.begin"] = "utterance.begin"
    utt_id: Id
    doa_rad: float | None = Field(default=None, description="音源方向（ラジアン）")
    person_hint: Id | None = Field(default=None, description="話者と思われる person_id")


class UtteranceEnd(_Msg):
    type: Literal["utterance.end"] = "utterance.end"
    utt_id: Id
    features: UtteranceFeatures


class FloorDecision(_Msg):
    """文字起こしを受けたロボットの判定。respond のときだけサーバーは生成する。"""

    type: Literal["floor.decision"] = "floor.decision"
    utt_id: Id
    decision: Literal["respond", "backchannel", "ignore"]
    reasons: list[str] = Field(default_factory=list)
    context: TurnContext | None = None

    @model_validator(mode="after")
    def _respond_needs_context(self) -> Self:
        if self.decision == "respond" and self.context is None:
            raise ValueError("decision=respond には context が必要")
        return self


class ProactiveRequest(_Msg):
    """ロボットの判断による自発発話の依頼（入室・アイコンタクト・見送り）。"""

    type: Literal["proactive.request"] = "proactive.request"
    intent_id: Id
    kind: Literal["greet", "engage", "see_off"]
    strategy: str | None = Field(default=None, max_length=64)
    context: TurnContext


class MemoryAck(_Msg):
    type: Literal["memory.ack"] = "memory.ack"
    op_id: Id
    result: Literal["applied", "rejected"]
    reason: str | None = None


class ToolResult(_Msg):
    type: Literal["tool.result"] = "tool.result"
    call_id: Id
    ok: bool
    result: JsonValue = None
    error: str | None = None

    @model_validator(mode="after")
    def _error_when_not_ok(self) -> Self:
        if not self.ok and not self.error:
            raise ValueError("ok=False には error が必要")
        return self


class TurnCancel(_Msg):
    type: Literal["turn.cancel"] = "turn.cancel"
    turn_id: Id
    reason: Literal["barge_in", "person_left", "other"] = "other"


class PersonForget(_Msg):
    type: Literal["person.forget"] = "person.forget"
    person_id: Id


class SessionClose(_Msg):
    """相手の退出。サーバーは会話を要約して memory.op(add_episode) で返す。"""

    type: Literal["session.close"] = "session.close"
    person_id: Id


# ---------------------------------------------------------------- server → robot


class Welcome(_Msg):
    type: Literal["welcome"] = "welcome"
    protocol_version: str = PROTOCOL_VERSION
    server_version: str
    models: dict[str, str] = Field(
        default_factory=dict, description='例: {"stt": "large-v3-turbo"}'
    )


class Transcript(_Msg):
    type: Literal["transcript"] = "transcript"
    utt_id: Id
    text: ShortText
    lang: str = "ja"
    confidence: float | None = Field(default=None, ge=0, le=1)


class TurnStart(_Msg):
    type: Literal["turn.start"] = "turn.start"
    turn_id: Id
    ref: Id = Field(description="応答のきっかけ（utt_id か intent_id）")


class Sentence(_Msg):
    type: Literal["sentence"] = "sentence"
    turn_id: Id
    idx: int = Field(ge=0, description="ターン内の文番号。audio_out の seq と対応する")
    text: ShortText
    emotion: Emotion = "neutral"


class Motion(_Msg):
    type: Literal["motion"] = "motion"
    turn_id: Id
    name: str = Field(min_length=1, max_length=32)


class MemoryOp(_Msg):
    """LLM が決めた記憶操作。ロボットが検証してから保存し、memory.ack で結果を返す。"""

    type: Literal["memory.op"] = "memory.op"
    op_id: Id
    person_id: Id
    op: Literal["set_name", "add_fact", "add_episode", "forget"]
    value: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _value_rules(self) -> Self:
        if self.op == "forget":
            if self.value is not None:
                raise ValueError("op=forget に value は不要")
        elif not (self.value and self.value.strip()):
            raise ValueError(f"op={self.op} には value が必要")
        return self


class ToolCall(_Msg):
    """ロボット側で実行するツールの呼び出し（lab_presence など）。"""

    type: Literal["tool.call"] = "tool.call"
    call_id: Id
    name: str = Field(min_length=1, max_length=64)
    args: dict[str, JsonValue] = Field(default_factory=dict)


class TurnEnd(_Msg):
    type: Literal["turn.end"] = "turn.end"
    turn_id: Id
    status: Literal["ok", "cancelled", "error"] = "ok"
    error: str | None = None


# ---------------------------------------------------------------- 双方向


class Ping(_Msg):
    type: Literal["ping"] = "ping"
    nonce: Id


class Pong(_Msg):
    type: Literal["pong"] = "pong"
    nonce: Id


class Error(_Msg):
    type: Literal["error"] = "error"
    code: Literal[
        "bad_message",
        "incompatible_version",
        "unauthorized",
        "unknown_ref",
        "timeout",
        "internal",
    ]
    message: str = ""
    ref: Id | None = Field(default=None, description="関係する utt_id / turn_id など")


# ---------------------------------------------------------------- union と parse

RobotMessage: TypeAlias = (
    Hello
    | UtteranceBegin
    | UtteranceEnd
    | FloorDecision
    | ProactiveRequest
    | MemoryAck
    | ToolResult
    | TurnCancel
    | PersonForget
    | SessionClose
    | Ping
    | Pong
    | Error
)
"""ロボットが送ってよいメッセージ。"""

ServerMessage: TypeAlias = (
    Welcome
    | Transcript
    | TurnStart
    | Sentence
    | Motion
    | MemoryOp
    | ToolCall
    | TurnEnd
    | Ping
    | Pong
    | Error
)
"""サーバーが送ってよいメッセージ。"""

Message: TypeAlias = Annotated[RobotMessage | ServerMessage, Field(discriminator="type")]

_ADAPTER: TypeAdapter[Message] = TypeAdapter(Message)


def _types_of(models: Any) -> frozenset[str]:
    return frozenset(m.model_fields["type"].default for m in models.__args__)


ROBOT_TO_SERVER: frozenset[str] = _types_of(RobotMessage)
SERVER_TO_ROBOT: frozenset[str] = _types_of(ServerMessage)


class MessageDirectionError(ValueError):
    """逆方向のメッセージを受け取った（例: ロボットから memory.op が来た）。"""


def parse_message(data: str | bytes | dict[str, Any]) -> Message:
    """JSON 文字列 / バイト列 / dict を検証してメッセージにする。

    不正なら pydantic の ValidationError を送出する（未知の type も含む）。
    """
    if isinstance(data, dict):
        return _ADAPTER.validate_python(data)
    return _ADAPTER.validate_json(data)


def _parse_directed(
    data: str | bytes | dict[str, Any], allowed: frozenset[str], who: str
) -> Message:
    msg = parse_message(data)
    if msg.type not in allowed:
        raise MessageDirectionError(f"{who} は {msg.type!r} を送れない")
    return msg


def parse_from_robot(data: str | bytes | dict[str, Any]) -> Message:
    """サーバー側の受信口。ロボットが送れないメッセージは MessageDirectionError。"""
    return _parse_directed(data, ROBOT_TO_SERVER, "robot")


def parse_from_server(data: str | bytes | dict[str, Any]) -> Message:
    """ロボット側の受信口。サーバーが送れないメッセージは MessageDirectionError。"""
    return _parse_directed(data, SERVER_TO_ROBOT, "server")


def dump_message(msg: Message) -> str:
    """送信用の JSON 文字列にする。"""
    return msg.model_dump_json()
