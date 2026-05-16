"""Contract tests for AgentPort — any implementation must pass these."""
import pytest
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.models.trace import AgentTurn, ToolCall


class StubAgentAdapter(AgentPort):
    """Minimal in-memory stub that proves the ABC works without LangGraph."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._counter = 0

    def new_session(self) -> str:
        self._counter += 1
        sid = f"stub-{self._counter}"
        self._sessions[sid] = []
        return sid

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        history = self._sessions[session_id]
        turn_number = len(history) + 1
        history.append({"input": user_input})
        return AgentTurn(
            turn_number=turn_number,
            agent_id="stub-agent",
            input_text=user_input,
            output_text=f"Stub response to: {user_input}",
            tool_calls=[],
            state_before={"turns": turn_number - 1},
            state_after={"turns": turn_number},
            latency_ms=1,
            tokens_used=10,
            visible_output_text=f"Stub response to: {user_input}",
        )

    def get_state(self, session_id: str) -> dict:
        return {"turns": len(self._sessions.get(session_id, []))}


class TestAgentPortContract:
    """Any AgentPort implementation must pass these tests."""

    @pytest.fixture
    def adapter(self) -> AgentPort:
        return StubAgentAdapter()

    def test_new_session_returns_string(self, adapter: AgentPort):
        sid = adapter.new_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_sessions_are_isolated(self, adapter: AgentPort):
        s1 = adapter.new_session()
        s2 = adapter.new_session()
        assert s1 != s2
        adapter.step(s1, "hello")
        state1 = adapter.get_state(s1)
        state2 = adapter.get_state(s2)
        assert state1 != state2

    def test_step_returns_agent_turn(self, adapter: AgentPort):
        sid = adapter.new_session()
        turn = adapter.step(sid, "test input")
        assert isinstance(turn, AgentTurn)
        assert turn.turn_number == 1
        assert turn.input_text == "test input"
        assert len(turn.visible_output_text) > 0

    def test_step_increments_turn_number(self, adapter: AgentPort):
        sid = adapter.new_session()
        t1 = adapter.step(sid, "first")
        t2 = adapter.step(sid, "second")
        assert t1.turn_number == 1
        assert t2.turn_number == 2

    def test_visible_output_text_is_populated(self, adapter: AgentPort):
        sid = adapter.new_session()
        turn = adapter.step(sid, "hi")
        assert turn.visible_output_text is not None
        assert isinstance(turn.visible_output_text, str)

    def test_get_state_returns_dict(self, adapter: AgentPort):
        sid = adapter.new_session()
        state = adapter.get_state(sid)
        assert isinstance(state, dict)
