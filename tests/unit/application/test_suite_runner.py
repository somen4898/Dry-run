"""Tests for suite runner."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.trace import Trace, AgentTurn, ToolCall
from dryrun.domain.models.evaluation import EvalResult, DimensionScore, RunResult
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.ports.agent import AgentPort
from dryrun.config import DryRunConfig, ThresholdsConfig
from dryrun.application.run_suite import RunSuiteUseCase


class MockLLMPort(LLMPort):
    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return json.dumps({"score": 0.85, "reason": "Mock"})


@pytest.fixture
def scenarios_dir(tmp_path):
    """Create a temp directory with 2 scenario YAML files."""
    s1 = tmp_path / "scenario1.yaml"
    s1.write_text("""
id: "s1"
name: "Scenario 1"
description: "Test 1"
persona:
  goal: "Buy laptop"
  tone: "polite"
  knowledge_level: "novice"
  background: "Student"
opening_input: "Hi"
expectations:
  required_tools: ["search"]
  required_tool_args: {}
  terminal_state: null
  output_must_contain: []
constraints:
  max_turns: 5
""")
    s2 = tmp_path / "scenario2.yaml"
    s2.write_text("""
id: "s2"
name: "Scenario 2"
description: "Test 2"
persona:
  goal: "Get refund"
  tone: "frustrated"
  knowledge_level: "intermediate"
  background: "Repeat customer"
opening_input: "I want a refund"
expectations:
  required_tools: ["lookup_order"]
  required_tool_args: {}
  terminal_state: null
  output_must_contain: []
constraints:
  max_turns: 8
""")
    return tmp_path


class TestRunSuite:
    @pytest.mark.asyncio
    async def test_run_suite_returns_run_result(self, scenarios_dir):
        """run_suite should return a RunResult with results for each scenario."""
        config = DryRunConfig(agent_module="x", agent_object="y")

        # Mock the run_scenario method to return a fake trace
        mock_trace = Trace(
            scenario_id="s1",
            turns=[AgentTurn(
                turn_number=1, agent_id="a", input_text="Hi", output_text="Hello",
                tool_calls=[ToolCall(tool_name="search", arguments={}, output={}, latency_ms=100)],
                state_before={}, state_after={}, latency_ms=200, tokens_used=100,
                visible_output_text="Hello",
            )],
            final_state={}, total_turns=1, total_tokens=100, total_latency_ms=200,
            terminal_reason="goal_met",
        )

        use_case = RunSuiteUseCase(
            agent_port=MagicMock(spec=AgentPort),
            llm_port=MockLLMPort(),
            config=config,
        )

        # Patch run_scenario to return mock trace
        use_case.run_scenario = AsyncMock(return_value=mock_trace)

        result = await use_case.run_suite(scenarios_dir)
        assert isinstance(result, RunResult)
        assert result.total_scenarios == 2
        assert len(result.eval_results) == 2

    @pytest.mark.asyncio
    async def test_run_suite_loads_yaml_files(self, scenarios_dir):
        """run_suite should discover and load all .yaml files in directory."""
        config = DryRunConfig(agent_module="x", agent_object="y")

        mock_trace = Trace(
            scenario_id="s1",
            turns=[AgentTurn(
                turn_number=1, agent_id="a", input_text="Hi", output_text="Hello",
                tool_calls=[],
                state_before={}, state_after={}, latency_ms=200, tokens_used=100,
                visible_output_text="Hello",
            )],
            final_state={}, total_turns=1, total_tokens=100, total_latency_ms=200,
            terminal_reason="goal_met",
        )

        use_case = RunSuiteUseCase(
            agent_port=MagicMock(spec=AgentPort),
            llm_port=MockLLMPort(),
            config=config,
        )
        use_case.run_scenario = AsyncMock(return_value=mock_trace)

        result = await use_case.run_suite(scenarios_dir)
        # Should have called run_scenario twice (once per yaml file)
        assert use_case.run_scenario.call_count == 2
