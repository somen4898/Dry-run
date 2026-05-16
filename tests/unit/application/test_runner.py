"""Tests for the runner — uses mock AgentPort and mock LLMPort."""
import pytest
import asyncio
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.trace import AgentTurn, Trace
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.application.run_suite import RunSuiteUseCase


class MockAgentPort(AgentPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self._sessions: dict[str, int] = {}
        self._counter = 0

    def new_session(self) -> str:
        self._counter += 1
        sid = f"mock-{self._counter}"
        self._sessions[sid] = 0
        return sid

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        self._sessions[session_id] += 1
        turn_num = self._sessions[session_id]
        response = next(self._responses)
        return AgentTurn(
            turn_number=turn_num,
            agent_id="mock-agent",
            input_text=user_input,
            output_text=response,
            tool_calls=[],
            state_before={"turn": turn_num - 1},
            state_after={"turn": turn_num},
            latency_ms=100,
            tokens_used=50,
            visible_output_text=response,
        )

    def get_state(self, session_id: str) -> dict:
        return {"turn": self._sessions.get(session_id, 0)}


class MockLLMPort(LLMPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return next(self._responses)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="test-001",
        name="Test",
        description="Test scenario",
        persona=Persona(
            goal="Buy a laptop", tone="polite",
            knowledge_level="novice", background="Student",
        ),
        opening_input="Hi, I need a laptop",
        expectations=Expectation(
            required_tools=[], required_tool_args={},
            terminal_state=None, output_must_contain=[],
        ),
        constraints=Constraints(max_turns=5),
    )


class TestRunSuiteUseCase:
    def test_goal_met_terminates(self, scenario):
        agent = MockAgentPort(["Here's a laptop for you!", "Order confirmed!"])
        llm = MockLLMPort([
            "I'd like the ThinkPad",   # user turn 1
            "yes",                      # drift check
            "GOAL_ACHIEVED",            # user turn 2 (terminal)
        ])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert isinstance(trace, Trace)
        assert trace.terminal_reason == "goal_met"
        assert trace.total_turns == 2

    def test_max_turns_terminates(self, scenario):
        scenario.constraints.max_turns = 2
        agent = MockAgentPort(["response1", "response2"])
        llm = MockLLMPort([
            "user msg 1", "yes",  # turn 1 + drift check
            "user msg 2", "yes",  # turn 2 + drift check
        ])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.terminal_reason == "max_turns"

    def test_trace_captures_scenario_id(self, scenario):
        agent = MockAgentPort(["Hello!"])
        llm = MockLLMPort(["GOAL_ACHIEVED"])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.scenario_id == "test-001"

    def test_trace_accumulates_tokens(self, scenario):
        agent = MockAgentPort(["r1", "r2"])
        llm = MockLLMPort(["msg", "yes", "GOAL_ACHIEVED"])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.total_tokens == 100  # 50 per turn, 2 turns
