"""Tests for LLM-based judge evaluators."""

import json
import pytest
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.trace import Trace, AgentTurn, ToolCall
from dryrun.application.llm_judge import (
    judge_goal_achievement,
    judge_trajectory_efficiency,
    judge_persona_fit,
)


class MockLLMPort(LLMPort):
    def __init__(self, response: str):
        self._response = response

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return self._response


class MockLLMPortInvalidThenValid(LLMPort):
    def __init__(self):
        self._call_count = 0

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        self._call_count += 1
        if self._call_count == 1:
            return "NOT VALID JSON"
        return json.dumps({"score": 0.8, "reason": "Recovered on retry"})


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="judge-001",
        name="Judge Test",
        description="Test",
        persona=Persona(
            goal="Get a refund for order #123",
            tone="frustrated",
            knowledge_level="intermediate",
            background="Repeat customer, unhappy with service",
        ),
        opening_input="I want a refund",
        expectations=Expectation(
            required_tools=["lookup_order", "process_refund"],
            required_tool_args={},
            terminal_state=None,
            output_must_contain=["refund"],
        ),
        constraints=Constraints(max_turns=10),
    )


@pytest.fixture
def trace() -> Trace:
    return Trace(
        scenario_id="judge-001",
        turns=[
            AgentTurn(
                turn_number=1,
                agent_id="support",
                input_text="I want a refund",
                output_text="Let me look up your order.",
                tool_calls=[
                    ToolCall(
                        tool_name="lookup_order",
                        arguments={"order_id": "123"},
                        output={"status": "delivered"},
                        latency_ms=200,
                    )
                ],
                state_before={},
                state_after={"order": "found"},
                latency_ms=500,
                tokens_used=200,
                visible_output_text="Let me look up your order.",
            ),
            AgentTurn(
                turn_number=2,
                agent_id="support",
                input_text="Yes, order 123",
                output_text="I've processed your refund of $49.99.",
                tool_calls=[
                    ToolCall(
                        tool_name="process_refund",
                        arguments={"order_id": "123", "amount": 49.99},
                        output={"success": True},
                        latency_ms=300,
                    )
                ],
                state_before={"order": "found"},
                state_after={"refund": "processed"},
                latency_ms=600,
                tokens_used=250,
                visible_output_text="I've processed your refund of $49.99.",
            ),
        ],
        final_state={"refund": "processed"},
        total_turns=2,
        total_tokens=450,
        total_latency_ms=1100,
        terminal_reason="goal_met",
    )


class TestGoalAchievement:
    @pytest.mark.asyncio
    async def test_high_score(self, trace, scenario):
        llm = MockLLMPort(json.dumps({"score": 0.95, "reason": "Goal fully achieved"}))
        result = await judge_goal_achievement(trace, scenario, llm)
        assert result.dimension == "goal_achievement"
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, trace, scenario):
        llm = MockLLMPortInvalidThenValid()
        result = await judge_goal_achievement(trace, scenario, llm)
        assert result.score == 0.8


class TestTrajectoryEfficiency:
    @pytest.mark.asyncio
    async def test_efficient_path(self, trace, scenario):
        llm = MockLLMPort(json.dumps({"score": 0.9, "reason": "Direct path"}))
        result = await judge_trajectory_efficiency(trace, scenario, llm)
        assert result.dimension == "trajectory_efficiency"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, trace, scenario):
        llm = MockLLMPortInvalidThenValid()
        result = await judge_trajectory_efficiency(trace, scenario, llm)
        assert result.score == 0.8


class TestPersonaFit:
    @pytest.mark.asyncio
    async def test_good_fit(self, trace, scenario):
        llm = MockLLMPort(json.dumps({"score": 0.85, "reason": "Appropriate tone"}))
        result = await judge_persona_fit(trace, scenario, llm)
        assert result.dimension == "persona_fit"
        assert result.score == 0.85

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, trace, scenario):
        llm = MockLLMPortInvalidThenValid()
        result = await judge_persona_fit(trace, scenario, llm)
        assert result.score == 0.8
