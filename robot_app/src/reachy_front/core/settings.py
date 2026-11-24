"""ロボット側（フロントエンド）の設定。環境変数から読む。設計書 §3.1。

**環境変数の名前は、移行前の ``robot/`` と同じ**（``EC_*``・``PROACTIVE_*``・``GATE_*``・``GAZE_*``・
``PC_HOST``・``PC_PORT``）なので、実機の起動スクリプトをそのまま使える。接頭辞は付けない。
しきい値は実機で調整するもの（初期値は旧実装のまま）。判断ロジックはこの設定を引数で受け取り、
環境変数を直接読まない（テストで値を差し替えられるように）。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Env(BaseSettings):
    """接頭辞なし・名前は各フィールドの別名（旧実装の環境変数名）。"""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)


class EyeContactSettings(_Env):
    """アイコンタクト判定のしきい値。相手が本当にこちらを見ているときだけ True にする（保守的に）。"""

    frontal_max: float = Field(
        0.35, alias="EC_FRONTAL_MAX", gt=0, description="|鼻-両目中点|/両目間（左右の正面度）"
    )
    level_max: float = Field(
        0.35, alias="EC_LEVEL_MAX", gt=0, description="両目の高さのズレ/両目間（傾き）"
    )
    face_min_frac: float = Field(
        0.14, alias="EC_FACE_MIN_FRAC", gt=0, le=1, description="顔幅/画面幅（近いほど関心がある）"
    )
    center_max: float = Field(
        0.35, alias="EC_CENTER_MAX", gt=0, description="|dx|,|dy|（こちらへ顔を向けている）"
    )
    vert_min: float = Field(
        0.15, alias="EC_VERT_MIN", description="鼻が両目より下にある度合いの下限（下向きを弾く）"
    )
    vert_max: float = Field(1.6, alias="EC_VERT_MAX", description="同、上限（上向きすぎを弾く）")
    grace_s: float = Field(
        0.4, alias="EC_GRACE", ge=0, description="この時間、生の検知が無いとアイコンタクトを解除"
    )


class ProactiveSettings(_Env):
    """自発発話。タイマー式の話しかけは無い（相手がじっと見ているときなど、契機があるときだけ）。"""

    enabled: bool = Field(True, alias="PROACTIVE_ENABLED")
    engage_s: float = Field(
        2.0, alias="PROACTIVE_ENGAGE_SEC", gt=0, description="この秒数、目が合い続けたら声をかける"
    )
    min_gap_s: float = Field(
        15.0, alias="PROACTIVE_MIN_GAP", ge=0, description="自発発話どうしの最小間隔"
    )
    poll_s: float = Field(
        0.5, alias="PROACTIVE_POLL_SEC", gt=0, description="アイコンタクト判定の粒度"
    )


class GateSettings(_Env):
    """発話を「自分に向けられた」とみなす条件。"""

    eye_s: float = Field(
        0.5,
        alias="GATE_EYE_SEC",
        ge=0,
        description="発話時点までの継続アイコンタクトの秒数。ちら見では応答しない",
    )


class GazeSettings(_Env):
    """視線の追従（首の制御）。向きは実機で確認した値。"""

    enabled: bool = Field(True, alias="GAZE_ENABLED")
    kp_yaw: float = Field(3.0, alias="GAZE_KP_YAW", gt=0)
    kp_pitch: float = Field(2.5, alias="GAZE_KP_PITCH", gt=0)
    yaw_sign: float = Field(-1.0, alias="GAZE_YAW_SIGN")
    pitch_sign: float = Field(1.0, alias="GAZE_PITCH_SIGN")
    debug: bool = Field(False, alias="GAZE_DEBUG")


class ServerLinkSettings(_Env):
    """GPU サーバーへの接続。トークンはロボットとサーバーで同じ値（ログには出ない）。"""

    host: str = Field("localhost", alias="PC_HOST", description="サーバーのホスト名（mDNS 名など）")
    port: int = Field(8765, alias="PC_PORT", ge=1, le=65535)
    token: SecretStr | None = Field(None, alias="REACHY_SERVER_TOKEN", min_length=16)

    @property
    def session_url(self) -> str:
        return f"ws://{self.host}:{self.port}/v1/session"

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class FrontSettings(_Env):
    """ロボット側の設定すべて。``FrontSettings()`` で環境変数から読む。"""

    eye_contact: EyeContactSettings = Field(default_factory=lambda: EyeContactSettings())  # pyright: ignore[reportCallIssue]
    proactive: ProactiveSettings = Field(default_factory=lambda: ProactiveSettings())  # pyright: ignore[reportCallIssue]
    gate: GateSettings = Field(default_factory=lambda: GateSettings())  # pyright: ignore[reportCallIssue]
    gaze: GazeSettings = Field(default_factory=lambda: GazeSettings())  # pyright: ignore[reportCallIssue]
    server: ServerLinkSettings = Field(default_factory=lambda: ServerLinkSettings())  # pyright: ignore[reportCallIssue]

    def dump(self) -> dict[str, Any]:
        """起動時のログ用。トークンは値を出さず、設定の有無だけ示す。"""
        out = self.model_dump(mode="json", exclude={"server": {"token"}})
        out["server"]["token"] = "**********" if self.server.token else None
        return out


__all__ = [
    "EyeContactSettings",
    "FrontSettings",
    "GateSettings",
    "GazeSettings",
    "ProactiveSettings",
    "ServerLinkSettings",
]
