"""Tests for scenario domain models."""
import pytest
from dryrun.domain.models.scenario import Persona, Expectation, Constraints, Scenario


class TestPersona:
    def test_create_with_defaults(self):
        p = Persona(
            goal="Buy a laptop",
            tone="polite",
            knowledge_level="novice",
            background="College student on a budget",
        )
        assert p.goal_reveal_strategy == "incremental"

    def test_goal_reveal_strategy_values(self):
        for strategy in ("incremental", "upfront", "evasive"):
            p = Persona(
                goal="x", tone="x", knowledge_level="x",
                background="x", goal_reveal_strategy=strategy,
            )
            assert p.goal_reveal_strategy == strategy


class TestConstraints:
    def test_defaults(self):
        c = Constraints()
        assert c.max_turns == 10
        assert c.timeout_seconds == 120
        assert c.max_tokens == 8000


class TestScenario:
    def test_full_scenario(self):
        s = Scenario(
            id="test-001",
            name="Happy path",
            description="User buys a laptop",
            persona=Persona(
                goal="Buy a laptop", tone="polite",
                knowledge_level="novice", background="Student",
            ),
            opening_input="Hi, I need a laptop",
            expectations=Expectation(
                required_tools=["search_inventory"],
                required_tool_args={"search_inventory": {"query": "laptop"}},
                terminal_state=None,
                output_must_contain=["laptop"],
            ),
            constraints=Constraints(max_turns=5),
        )
        assert s.golden is False
        assert s.tags == []
        assert s.constraints.max_turns == 5

    def test_scenario_golden_flag(self):
        s = Scenario(
            id="g-001", name="Golden", description="x",
            persona=Persona(goal="x", tone="x", knowledge_level="x", background="x"),
            opening_input="x",
            expectations=Expectation(
                required_tools=[], required_tool_args={},
                terminal_state=None, output_must_contain=[],
            ),
            constraints=Constraints(),
            golden=True, tags=["smoke"],
        )
        assert s.golden is True
        assert "smoke" in s.tags
