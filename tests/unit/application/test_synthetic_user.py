"""Tests for SyntheticUser — uses a mock LLMPort."""

import pytest
import asyncio
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.models.scenario import Persona
from dryrun.application.synthetic_user import SyntheticUser


class MockLLMPort(LLMPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return next(self._responses)


class TestSyntheticUser:
    @pytest.fixture
    def persona(self) -> Persona:
        return Persona(
            goal="Buy a laptop",
            tone="polite",
            knowledge_level="novice",
            background="College student",
            goal_reveal_strategy="incremental",
        )

    def test_next_message_returns_string(self, persona):
        llm = MockLLMPort(["I'm looking for something affordable", "yes"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "How can I help?"}]
        result = asyncio.run(user.next_message(history))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_goal_achieved_signal(self, persona):
        llm = MockLLMPort(["GOAL_ACHIEVED"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "Your order is confirmed!"}]
        result = asyncio.run(user.next_message(history))
        assert result == "GOAL_ACHIEVED"

    def test_goal_abandoned_signal(self, persona):
        llm = MockLLMPort(["GOAL_ABANDONED"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "Sorry, we can't help."}]
        result = asyncio.run(user.next_message(history))
        assert result == "GOAL_ABANDONED"

    def test_system_prompt_contains_goal_reveal_strategy(self, persona):
        llm = MockLLMPort(["response"])
        user = SyntheticUser(persona=persona, llm=llm)
        prompt = user._build_system_prompt()
        assert (
            "do not state your full goal" in prompt.lower()
            or "state only your immediate need" in prompt.lower()
        )
        assert persona.goal in prompt

    def test_system_prompt_evasive_strategy(self):
        persona = Persona(
            goal="Get a refund",
            tone="frustrated",
            knowledge_level="expert",
            background="Repeat customer",
            goal_reveal_strategy="evasive",
        )
        llm = MockLLMPort(["response"])
        user = SyntheticUser(persona=persona, llm=llm)
        prompt = user._build_system_prompt()
        assert "do not volunteer information" in prompt.lower()

    def test_persona_drift_check_passes_good_message(self, persona):
        llm = MockLLMPort(["yes"])
        user = SyntheticUser(persona=persona, llm=llm)
        result = asyncio.run(user._check_persona_drift("I'd like a budget laptop please"))
        assert result is True

    def test_persona_drift_check_fails_on_ai_reveal(self, persona):
        llm = MockLLMPort(["no"])
        user = SyntheticUser(persona=persona, llm=llm)
        result = asyncio.run(
            user._check_persona_drift("As an AI language model, I cannot actually buy things")
        )
        assert result is False

    def test_drift_retry_then_accept(self, persona):
        """On drift failure, regenerate once. On second failure, accept with warning."""
        llm = MockLLMPort(
            [
                "As an AI, I can't buy things",  # first generation (bad)
                "no",  # drift check fails
                "I'd like a laptop please",  # retry generation (good)
                "yes",  # drift check passes
            ]
        )
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "How can I help?"}]
        result = asyncio.run(user.next_message(history))
        assert "AI" not in result
