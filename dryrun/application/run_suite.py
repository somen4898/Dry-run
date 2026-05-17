"""RunSuiteUseCase — orchestrates scenario execution."""

from __future__ import annotations
import time
import logging
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

import yaml

from dryrun.config import DryRunConfig
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.trace import Trace, AgentTurn
from dryrun.domain.models.evaluation import RunResult
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.application.synthetic_user import SyntheticUser, _TERMINAL_SIGNALS

logger = logging.getLogger(__name__)


class RunSuiteUseCase:
    def __init__(self, agent_port: AgentPort, llm_port: LLMPort, config: DryRunConfig | None = None):
        self._agent = agent_port
        self._llm = llm_port
        self._config = config or DryRunConfig(agent_module="", agent_object="")

    async def run_suite(self, scenarios_dir: Path) -> RunResult:
        """Run all scenarios in a directory and return aggregated results."""
        from dryrun.application.evaluator import Evaluator

        scenario_files = sorted(scenarios_dir.glob("*.yaml"))
        scenarios = [Scenario(**yaml.safe_load(f.read_text())) for f in scenario_files]

        evaluator = Evaluator()
        eval_results = []
        total_tokens = 0

        for scenario in scenarios:
            trace = await self.run_scenario(scenario)
            result = await evaluator.evaluate(trace, scenario, self._llm, self._config.thresholds)
            eval_results.append(result)
            total_tokens += trace.total_tokens

        # Compute per-dimension averages
        dim_totals: dict[str, list[float]] = {}
        for er in eval_results:
            for d in er.dimensions:
                dim_totals.setdefault(d.dimension, []).append(d.score)
        per_dimension_scores = {k: sum(v) / len(v) for k, v in dim_totals.items()}

        passed_count = sum(1 for r in eval_results if r.passed)
        failed_count = len(eval_results) - passed_count
        aggregate = sum(r.aggregate_score for r in eval_results) / len(eval_results) if eval_results else 0.0

        return RunResult(
            run_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            total_scenarios=len(eval_results),
            passed=passed_count,
            failed=failed_count,
            aggregate_score=aggregate,
            per_dimension_scores=per_dimension_scores,
            eval_results=eval_results,
            token_cost_actual=total_tokens,
        )

    async def run_scenario(self, scenario: Scenario) -> Trace:
        session_id = self._agent.new_session()
        user = SyntheticUser(persona=scenario.persona, llm=self._llm)

        turns: list[AgentTurn] = []
        history: list[dict] = [{"role": "user", "content": scenario.opening_input}]
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
