"""Tests that port ABCs cannot be instantiated and enforce the contract."""

import pytest
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort


class TestAgentPortABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AgentPort()  # type: ignore

    def test_has_required_methods(self):
        assert hasattr(AgentPort, "new_session")
        assert hasattr(AgentPort, "step")
        assert hasattr(AgentPort, "get_state")


class TestLLMPortABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMPort()  # type: ignore

    def test_has_complete_method(self):
        assert hasattr(LLMPort, "complete")
