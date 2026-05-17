"""Tests for deterministic scoring functions."""

import pytest
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.trace import Trace, AgentTurn, ToolCall
from dryrun.domain.services.scoring import (
    score_tool_correctness,
    score_argument_correctness,
    score_step_efficiency,
    score_constraint_adherence,
)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="test-001",
        name="Test",
        description="Test scenario",
        persona=Persona(
            goal="Buy a laptop",
            tone="polite",
            knowledge_level="novice",
            background="Student",
        ),
        opening_input="Hi",
        expectations=Expectation(
            required_tools=["search_inventory", "add_to_cart", "process_checkout"],
            required_tool_args={"add_to_cart": {"item_id": "laptop-001"}},
            terminal_state=None,
            output_must_contain=[],
        ),
        constraints=Constraints(max_turns=8, timeout_seconds=120, max_tokens=8000),
    )


def _make_trace(
    tool_calls: list[list[ToolCall]],
    total_turns: int = 3,
    total_tokens: int = 500,
    total_latency_ms: int = 5000,
) -> Trace:
    turns = []
    for i, tcs in enumerate(tool_calls):
        turns.append(
            AgentTurn(
                turn_number=i + 1,
                agent_id="test",
                input_text="input",
                output_text="output",
                tool_calls=tcs,
                state_before={},
                state_after={},
                latency_ms=total_latency_ms // max(len(tool_calls), 1),
                tokens_used=total_tokens // max(len(tool_calls), 1),
                visible_output_text="output",
            )
        )
    return Trace(
        scenario_id="test-001",
        turns=turns,
        final_state={},
        total_turns=total_turns,
        total_tokens=total_tokens,
        total_latency_ms=total_latency_ms,
        terminal_reason="goal_met",
    )


class TestToolCorrectness:
    def test_all_tools_called(self, scenario):
        trace = _make_trace(
            [
                [ToolCall(tool_name="search_inventory", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="add_to_cart", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="process_checkout", arguments={}, output=None, latency_ms=0)],
            ]
        )
        result = score_tool_correctness(trace, scenario.expectations)
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_one_tool(self, scenario):
        trace = _make_trace(
            [
                [ToolCall(tool_name="search_inventory", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="add_to_cart", arguments={}, output=None, latency_ms=0)],
            ]
        )
        result = score_tool_correctness(trace, scenario.expectations)
        assert result.score == pytest.approx(2 / 3)
        assert result.passed is False

    def test_no_tools_required(self, scenario):
        scenario.expectations.required_tools = []
        trace = _make_trace([[]])
        result = score_tool_correctness(trace, scenario.expectations)
        assert result.score == 1.0
        assert result.passed is True

    def test_extra_tools_dont_penalize(self, scenario):
        trace = _make_trace(
            [
                [ToolCall(tool_name="search_inventory", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="add_to_cart", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="process_checkout", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="update_cart", arguments={}, output=None, latency_ms=0)],
            ]
        )
        result = score_tool_correctness(trace, scenario.expectations)
        assert result.score == 1.0


class TestArgumentCorrectness:
    def test_correct_arguments(self, scenario):
        trace = _make_trace(
            [
                [
                    ToolCall(
                        tool_name="add_to_cart",
                        arguments={"item_id": "laptop-001", "qty": 1},
                        output=None,
                        latency_ms=0,
                    )
                ],
            ]
        )
        result = score_argument_correctness(trace, scenario.expectations)
        assert result.score == 1.0
        assert result.passed is True

    def test_wrong_arguments(self, scenario):
        trace = _make_trace(
            [
                [
                    ToolCall(
                        tool_name="add_to_cart",
                        arguments={"item_id": "tablet-001"},
                        output=None,
                        latency_ms=0,
                    )
                ],
            ]
        )
        result = score_argument_correctness(trace, scenario.expectations)
        assert result.score == 0.0
        assert result.passed is False

    def test_no_args_expected(self, scenario):
        scenario.expectations.required_tool_args = {}
        trace = _make_trace([[]])
        result = score_argument_correctness(trace, scenario.expectations)
        assert result.score == 1.0

    def test_tool_not_called_scores_zero(self, scenario):
        trace = _make_trace(
            [
                [ToolCall(tool_name="search_inventory", arguments={}, output=None, latency_ms=0)],
            ]
        )
        result = score_argument_correctness(trace, scenario.expectations)
        assert result.score == 0.0


class TestStepEfficiency:
    def test_clean_path(self, scenario):
        trace = _make_trace(
            [
                [ToolCall(tool_name="search_inventory", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="add_to_cart", arguments={}, output=None, latency_ms=0)],
                [ToolCall(tool_name="process_checkout", arguments={}, output=None, latency_ms=0)],
            ]
        )
        result = score_step_efficiency(trace)
        assert result.score == 1.0
        assert result.passed is True

    def test_redundant_calls_penalized(self, scenario):
        trace = _make_trace(
            [
                [
                    ToolCall(
                        tool_name="search_inventory",
                        arguments={"q": "laptop"},
                        output=None,
                        latency_ms=0,
                    )
                ],
                [
                    ToolCall(
                        tool_name="search_inventory",
                        arguments={"q": "laptop"},
                        output=None,
                        latency_ms=0,
                    )
                ],
                [
                    ToolCall(
                        tool_name="search_inventory",
                        arguments={"q": "laptop"},
                        output=None,
                        latency_ms=0,
                    )
                ],
            ]
        )
        result = score_step_efficiency(trace)
        assert result.score < 1.0

    def test_loop_detected(self, scenario):
        """Same agent_id visited >3 times suggests a loop."""
        turns = [
            AgentTurn(
                turn_number=i + 1,
                agent_id="sales",
                input_text="x",
                output_text="y",
                tool_calls=[],
                state_before={},
                state_after={},
                latency_ms=100,
                tokens_used=50,
                visible_output_text="y",
            )
            for i in range(5)
        ]
        trace = Trace(
            scenario_id="test-001",
            turns=turns,
            final_state={},
            total_turns=5,
            total_tokens=250,
            total_latency_ms=500,
            terminal_reason="max_turns",
        )
        result = score_step_efficiency(trace)
        assert result.score < 1.0
        assert "loop" in result.reason.lower()


class TestConstraintAdherence:
    def test_all_constraints_met(self, scenario):
        trace = _make_trace([[]], total_turns=3, total_tokens=500, total_latency_ms=5000)
        result = score_constraint_adherence(trace, scenario.constraints)
        assert result.score == 1.0
        assert result.passed is True

    def test_turns_exceeded(self, scenario):
        scenario.constraints.max_turns = 2
        trace = _make_trace([[], [], []], total_turns=3, total_tokens=500, total_latency_ms=5000)
        result = score_constraint_adherence(trace, scenario.constraints)
        assert result.score < 1.0
        assert result.passed is False
        assert "turn" in result.reason.lower()

    def test_tokens_exceeded(self, scenario):
        scenario.constraints.max_tokens = 100
        trace = _make_trace([[]], total_turns=1, total_tokens=500, total_latency_ms=5000)
        result = score_constraint_adherence(trace, scenario.constraints)
        assert result.score < 1.0
        assert "token" in result.reason.lower()

    def test_timeout_exceeded(self, scenario):
        scenario.constraints.timeout_seconds = 1
        trace = _make_trace([[]], total_turns=1, total_tokens=100, total_latency_ms=5000)
        result = score_constraint_adherence(trace, scenario.constraints)
        assert result.score < 1.0
        assert "timeout" in result.reason.lower()
