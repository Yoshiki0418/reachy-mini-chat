"""宛先判定の補助 API。設計書 §3.3。

ロボットのルール（名前呼び・視線・人数など）で決めきれない発話だけを、サーバーの LLM に聞く。
  POST /v1/classify   ``Authorization: Bearer <トークン>``
ロボットは 400 ms 待って、返らなければルールだけで決める（**迷ったら黙る**）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AddresseeLabel = Literal["robot", "other", "unsure"]


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=500, description="文字起こし")
    people_count: int | None = Field(default=None, ge=0, le=50, description="見えている人数")
    robot_asked_recently: bool = Field(
        default=False, description="直近でロボットがこの人に質問した（会話の継続）"
    )


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: AddresseeLabel = Field(
        description="robot=ロボット宛て / other=人同士の会話・独り言 / unsure=判断できない"
    )


__all__ = ["AddresseeLabel", "ClassifyRequest", "ClassifyResponse"]
