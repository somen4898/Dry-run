"""OpenAIClient — implements LLMPort using the OpenAI API."""
from __future__ import annotations
from openai import AsyncOpenAI
from dryrun.domain.ports.llm import LLMPort


class OpenAIClient(LLMPort):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
