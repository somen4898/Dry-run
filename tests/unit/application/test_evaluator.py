"""Tests for Evaluator orchestrator."""

import json

import pytest

from dryrun.application.evaluator import Evaluator
from dryrun.config import ThresholdsConfig
from dryrun.domain.models.scenario import Constraints, Expectation, Persona, Scenario
from dryrun.domain.models.trace import AgentTurn, ToolCall, Trace
from dryrun.domain.ports.llm import LLMPort


class MockLLMPort(LLMPort):
    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return json.dumps({"score": 0.85, "reason": "Mock judge"})


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="eval-001",
        name="Test",
        description="Test",
        persona=Persona(
            goal="Buy laptop",
            tone="polite",
            knowledge_level="novice",
            background="Student",
        ),
        opening_input="Hi",
        expectations=Expectation(
            required_tools=["search_inventory", "add_to_cart"],
            required_tool_args={"add_to_cart": {"item_id": "laptop-001"}},
            terminal_state=None,
            output_must_contain=[],
        ),
        constraints=Constraints(max_turns=8),
    )


@pytest.fixture
def trace() -> Trace:
    return Trace(
        scenario_id="eval-001",
        turns=[
            AgentTurn(
                turn_number=1,
                agent_id="sales",
                input_text="Hi",
                output_text="Welcome!",
                tool_calls=[
                    ToolCall(
                        tool_name="search_inventory",
                        arguments={"query": "laptop"},
                        output=[],
                        latency_ms=100,
                    )
                ],
                state_before={},
                state_after={},
                latency_ms=200,
                tokens_used=100,
                visible_output_text="Welcome!",
            ),
            AgentTurn(
                turn_number=2,
                agent_id="sales",
                input_text="Budget one",
                output_text="Added",
                tool_calls=[
                    ToolCall(
                        tool_name="add_to_cart",
                        arguments={"item_id": "laptop-001"},
                        output={},
                        latency_ms=50,
                    )
                ],
                state_before={},
                state_after={},
                latency_ms=150,
                tokens_used=80,
                visible_output_text="Added",
            ),
        ],
        final_state={},
        total_turns=2,
        total_tokens=180,
        total_latency_ms=350,
        terminal_reason="goal_met",
    )


class TestEvaluator:
    @pytest.mark.asyncio
    async def test_returns_7_dimensions(self, trace, scenario):
        evaluator = Evaluator()
        result = await evaluator.evaluate(trace, scenario, MockLLMPort(), ThresholdsConfig())
        assert len(result.dimensions) == 7

    @pytest.mark.asyncio
    async def test_all_dimension_names_present(self, trace, scenario):
        evaluator = Evaluator()
        result = await evaluator.evaluate(trace, scenario, MockLLMPort(), ThresholdsConfig())
        names = {d.dimension for d in result.dimensions}
        assert names == {
            "tool_correctness",
            "argument_correctness",
            "step_efficiency",
            "constraint_adherence",
            "goal_achievement",
            "trajectory_efficiency",
            "persona_fit",
        }

    @pytest.mark.asyncio
    async def test_passes_when_all_good(self, trace, scenario):
        evaluator = Evaluator()
        result = await evaluator.evaluate(trace, scenario, MockLLMPort(), ThresholdsConfig())
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_scenario_id_set(self, trace, scenario):
        evaluator = Evaluator()
        result = await evaluator.evaluate(trace, scenario, MockLLMPort(), ThresholdsConfig())
        assert result.scenario_id == "eval-001"
