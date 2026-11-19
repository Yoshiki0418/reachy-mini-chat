"""OpenAI のチャットモデル（LangChain の ChatOpenAI）。API キーは環境変数 OPENAI_API_KEY。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


@dataclass
class OpenAILLM:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 200
    name: str = "openai"

    def chat_model(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            streaming=True,
        )
