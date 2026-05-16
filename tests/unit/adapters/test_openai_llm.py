"""Tests for OpenAIClient — implements LLMPort."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dryrun.adapters.outbound.openai.llm import OpenAIClient
from dryrun.domain.ports.llm import LLMPort


class TestOpenAIClient:
    def test_implements_llm_port(self):
        with patch("dryrun.adapters.outbound.openai.llm.AsyncOpenAI"):
            client = OpenAIClient(model="gpt-4o-mini")
            assert isinstance(client, LLMPort)

    @patch("dryrun.adapters.outbound.openai.llm.AsyncOpenAI")
    def test_complete_returns_string(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o-mini")
        result = asyncio.run(client.complete(
            [{"role": "user", "content": "Hi"}]
        ))
        assert result == "Hello!"

    @patch("dryrun.adapters.outbound.openai.llm.AsyncOpenAI")
    def test_temperature_passed_through(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o-mini")
        asyncio.run(client.complete(
            [{"role": "user", "content": "test"}],
            temperature=0.0,
        ))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
