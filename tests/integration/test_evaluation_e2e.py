"""End-to-end evaluation test — validates Phase 2 exit criterion.

Runs 10 scenarios through the full pipeline with mock agent + mock LLM judges.
No API key required.
"""

import json

import pytest
from pathlib import Path

from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.models.trace import Trace, AgentTurn, ToolCall
from dryrun.domain.models.evaluation import RunResult
from dryrun.config import DryRunConfig
from dryrun.application.run_suite import RunSuiteUseCase
from dryrun.adapters.outbound.reporters.terminal import TerminalReporter


class MockLLMPort(LLMPort):
    """Mock LLM that returns passing scores for judge calls and GOAL_ACHIEVED for synthetic user."""

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return json.dumps({"score": 0.85, "reason": "Mock judge: good performance"})


class MockAgentPort(AgentPort):
    """Mock agent that simulates tool calls based on scenario expectations."""

    def __init__(self):
        self._sessions: dict[str, int] = {}

    def new_session(self) -> str:
        import uuid

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = 0
        return session_id

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        self._sessions[session_id] = self._sessions.get(session_id, 0) + 1
        turn_num = self._sessions[session_id]

        tool_calls = []
        if turn_num == 1:
            tool_calls = [
                ToolCall(
                    tool_name="search_inventory",
                    arguments={"query": "laptop"},
                    output=[{"id": "laptop-001", "name": "Budget Laptop", "price": 499}],
                    latency_ms=100,
                ),
            ]

        return AgentTurn(
            turn_number=turn_num,
            agent_id="mock-agent",
            input_text=user_input,
            output_text=f"Mock response for turn {turn_num}. I'll help with your order.",
            tool_calls=tool_calls,
            state_before={},
            state_after={"turn": turn_num},
            latency_ms=200,
            tokens_used=150,
            visible_output_text=f"Mock response for turn {turn_num}. I'll help with your order.",
        )

    def get_state(self, session_id: str) -> dict:
        return {"turn": self._sessions.get(session_id, 0)}


def _make_mock_run_scenario():
    """Create a mock run_scenario that returns a predictable trace."""

    async def mock_run_scenario(scenario):
        """Return a trace with tools that partially match expectations."""
        # Use first 2 required tools from the scenario
        tool_calls = [
            ToolCall(
                tool_name=t, arguments={}, output={"success": True}, latency_ms=100
            )
            for t in scenario.expectations.required_tools[:2]
        ]

        return Trace(
            scenario_id=scenario.id,
            turns=[
                AgentTurn(
                    turn_number=1,
                    agent_id="mock",
                    input_text=scenario.opening_input,
                    output_text=f"Handling: {scenario.persona.goal}. Your order refund has been processed.",
                    tool_calls=tool_calls,
                    state_before={},
                    state_after={"done": True},
                    latency_ms=300,
                    tokens_used=200,
                    visible_output_text=f"Handling: {scenario.persona.goal}. Your order refund has been processed.",
                ),
            ],
            final_state={"done": True},
            total_turns=1,
            total_tokens=200,
            total_latency_ms=300,
            terminal_reason="goal_met",
        )

    return mock_run_scenario


@pytest.fixture
def scenarios_dir():
    """Path to the example scenarios directory."""
    return Path(__file__).parent.parent.parent / "example" / "scenarios"


@pytest.fixture
def use_case():
    """Create a RunSuiteUseCase with mocked run_scenario."""
    config = DryRunConfig(agent_module="mock", agent_object="mock")
    mock_llm = MockLLMPort()
    mock_agent = MockAgentPort()

    uc = RunSuiteUseCase(agent_port=mock_agent, llm_port=mock_llm, config=config)
    uc.run_scenario = _make_mock_run_scenario()
    return uc


class TestEvaluationE2E:
    @pytest.mark.asyncio
    async def test_full_pipeline_runs_all_scenarios(self, use_case, scenarios_dir):
        """10 scenarios run through runner -> evaluator -> reporter."""
        result = await use_case.run_suite(scenarios_dir)

        assert isinstance(result, RunResult)
        assert result.total_scenarios == 10
        assert len(result.eval_results) == 10

    @pytest.mark.asyncio
    async def test_each_result_has_7_dimensions(self, use_case, scenarios_dir):
        """Each eval result should have exactly 7 scored dimensions."""
        result = await use_case.run_suite(scenarios_dir)

        for eval_result in result.eval_results:
            assert len(eval_result.dimensions) == 7, (
                f"Scenario {eval_result.scenario_id} has {len(eval_result.dimensions)} dimensions, expected 7"
            )

    @pytest.mark.asyncio
    async def test_dimension_names_correct(self, use_case, scenarios_dir):
        """All 7 expected dimension names should be present in each result."""
        result = await use_case.run_suite(scenarios_dir)

        expected_dims = {
            "tool_correctness",
            "argument_correctness",
            "step_efficiency",
            "constraint_adherence",
            "goal_achievement",
            "trajectory_efficiency",
            "persona_fit",
        }

        for eval_result in result.eval_results:
            actual_dims = {d.dimension for d in eval_result.dimensions}
            assert actual_dims == expected_dims, (
                f"Scenario {eval_result.scenario_id}: expected {expected_dims}, got {actual_dims}"
            )

    @pytest.mark.asyncio
    async def test_pass_fail_logic(self, use_case, scenarios_dir):
        """Verify pass/fail is computed (passed + failed should equal total)."""
        result = await use_case.run_suite(scenarios_dir)

        assert result.passed + result.failed == result.total_scenarios

    @pytest.mark.asyncio
    async def test_reporter_does_not_crash(self, use_case, scenarios_dir):
        """TerminalReporter should handle the full RunResult without errors."""
        result = await use_case.run_suite(scenarios_dir)

        reporter = TerminalReporter()
        # Should not raise
        for eval_result in result.eval_results:
            reporter.report_scenario(eval_result)
        reporter.report_suite(result)

    @pytest.mark.asyncio
    async def test_per_dimension_scores_populated(self, use_case, scenarios_dir):
        """RunResult should have per-dimension average scores."""
        result = await use_case.run_suite(scenarios_dir)

        assert len(result.per_dimension_scores) == 7
        for dim, score in result.per_dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} score {score} out of range"
