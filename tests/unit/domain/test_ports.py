"""Tests that port ABCs cannot be instantiated and enforce the contract."""

import pytest
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.ports.store import StorePort


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


class TestStorePortABC:
    def test_cannot_instantiate(self):
        import pytest

        with pytest.raises(TypeError):
            StorePort()

    def test_has_required_methods(self):
        methods = [
            "upsert_scenario",
            "find_similar_scenarios",
            "is_near_duplicate",
            "save_run",
            "get_run",
            "get_latest_run",
            "get_golden_scenarios",
            "mark_golden",
            "find_similar_failures",
        ]
        for method in methods:
            assert hasattr(StorePort, method)
