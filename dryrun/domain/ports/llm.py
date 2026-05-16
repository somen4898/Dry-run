"""LLMPort — the contract for LLM completion providers."""
from __future__ import annotations
from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """Send messages to an LLM and return the completion text."""
        ...
