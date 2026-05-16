"""Tests for scenario domain models."""

from dryrun.domain.models.scenario import Persona, Expectation, Constraints, Scenario
from dryrun.domain.models.trace import ToolCall, AgentTurn, Trace


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
                goal="x",
                tone="x",
                knowledge_level="x",
                background="x",
                goal_reveal_strategy=strategy,
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
                goal="Buy a laptop",
                tone="polite",
                knowledge_level="novice",
                background="Student",
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
            id="g-001",
            name="Golden",
            description="x",
            persona=Persona(goal="x", tone="x", knowledge_level="x", background="x"),
            opening_input="x",
            expectations=Expectation(
                required_tools=[],
                required_tool_args={},
                terminal_state=None,
                output_must_contain=[],
            ),
            constraints=Constraints(),
            golden=True,
            tags=["smoke"],
        )
        assert s.golden is True
        assert "smoke" in s.tags


class TestToolCall:
    def test_create(self):
        tc = ToolCall(
            tool_name="search_inventory",
            arguments={"query": "laptop"},
            output={"results": [{"name": "ThinkPad"}]},
            latency_ms=150,
        )
        assert tc.tool_name == "search_inventory"


class TestAgentTurn:
    def test_visible_output_separate_from_output(self):
        turn = AgentTurn(
            turn_number=1,
            agent_id="support",
            input_text="Hi",
            output_text="[internal reasoning] Hello! How can I help?",
            tool_calls=[],
            state_before={},
            state_after={"greeted": True},
            latency_ms=500,
            tokens_used=100,
            visible_output_text="Hello! How can I help?",
        )
        assert turn.visible_output_text != turn.output_text
        assert "[internal" not in turn.visible_output_text


class TestTrace:
    def test_empty_trace(self):
        t = Trace(
            scenario_id="test-001",
            turns=[],
            final_state={},
            total_turns=0,
            total_tokens=0,
            total_latency_ms=0,
            terminal_reason="max_turns",
        )
        assert t.terminal_reason == "max_turns"

    def test_trace_with_turns(self):
        turn = AgentTurn(
            turn_number=1,
            agent_id="support",
            input_text="Hi",
            output_text="Hello!",
            tool_calls=[],
            state_before={},
            state_after={},
            latency_ms=200,
            tokens_used=50,
            visible_output_text="Hello!",
        )
        t = Trace(
            scenario_id="test-001",
            turns=[turn],
            final_state={"done": True},
            total_turns=1,
            total_tokens=50,
            total_latency_ms=200,
            terminal_reason="goal_met",
        )
        assert len(t.turns) == 1
        assert t.turns[0].agent_id == "support"
