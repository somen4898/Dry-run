"""RunSuiteUseCase — orchestrates scenario execution."""
from __future__ import annotations
import time
import logging
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.trace import Trace, AgentTurn
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.application.synthetic_user import SyntheticUser

logger = logging.getLogger(__name__)

_TERMINAL_SIGNALS = frozenset({"GOAL_ACHIEVED", "GOAL_ABANDONED"})


class RunSuiteUseCase:
    def __init__(self, agent_port: AgentPort, llm_port: LLMPort):
        self._agent = agent_port
        self._llm = llm_port

    async def run_scenario(self, scenario: Scenario) -> Trace:
        session_id = self._agent.new_session()
        user = SyntheticUser(persona=scenario.persona, llm=self._llm)

        turns: list[AgentTurn] = []
        history: list[dict] = []
        current_input = scenario.opening_input
        total_tokens = 0
        total_latency_ms = 0
        start_time = time.monotonic()

        for turn_idx in range(scenario.constraints.max_turns):
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > scenario.constraints.timeout_seconds:
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, "timeout"
                )

            # Run agent step
            agent_turn = self._agent.step(session_id, current_input)
            turns.append(agent_turn)
            total_tokens += agent_turn.tokens_used
            total_latency_ms += agent_turn.latency_ms

            # IMPORTANT: pass ONLY visible_output_text to synthetic user
            history.append({"role": "assistant", "content": agent_turn.visible_output_text})

            # Check token budget
            if total_tokens > scenario.constraints.max_tokens:
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, "token_budget"
                )

            # Get synthetic user response
            next_input = await user.next_message(history)

            if next_input in _TERMINAL_SIGNALS:
                reason = "goal_met" if next_input == "GOAL_ACHIEVED" else "goal_abandoned"
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, reason
                )

            history.append({"role": "user", "content": next_input})
            current_input = next_input

        return self._build_trace(
            scenario, turns, session_id, total_tokens, total_latency_ms, "max_turns"
        )

    def _build_trace(
        self,
        scenario: Scenario,
        turns: list[AgentTurn],
        session_id: str,
        total_tokens: int,
        total_latency_ms: int,
        terminal_reason: str,
    ) -> Trace:
        final_state = self._agent.get_state(session_id) if turns else {}
        return Trace(
            scenario_id=scenario.id,
            turns=turns,
            final_state=final_state,
            total_turns=len(turns),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            terminal_reason=terminal_reason,
        )
