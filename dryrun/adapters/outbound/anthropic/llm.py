"""AnthropicClient — implements LLMPort using the Anthropic API."""

from __future__ import annotations
from anthropic import AsyncAnthropic
from dryrun.domain.ports.llm import LLMPort


class AnthropicClient(LLMPort):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        # Anthropic uses a separate system parameter, not a system message in the list
        system_text = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                chat_messages.append(msg)

        # Anthropic requires alternating user/assistant — merge consecutive same-role messages
        merged: list[dict] = []
        for msg in chat_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n" + msg["content"]
            else:
                merged.append(dict(msg))

        # Ensure first message is from user (Anthropic requirement)
        if not merged or merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "Hello"})

        kwargs: dict = {
            "model": self._model,
            "messages": merged,
            "max_tokens": 1024,
            "temperature": temperature,
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""
