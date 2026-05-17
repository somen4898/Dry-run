"""Evaluator orchestrator — runs all 7 dimensions, aggregates into EvalResult."""

from __future__ import annotations

import asyncio

from dryrun.application.llm_judge import (
    judge_goal_achievement,
    judge_persona_fit,
    judge_trajectory_efficiency,
)
from dryrun.config import ThresholdsConfig
from dryrun.domain.models.evaluation import EvalResult
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.trace import Trace
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.services.aggregation import aggregate_scores
from dryrun.domain.services.scoring import (
    score_argument_correctness,
    score_constraint_adherence,
    score_step_efficiency,
    score_tool_correctness,
)


class Evaluator:
    """Orchestrates all 7 evaluation dimensions and produces an EvalResult."""

    async def evaluate(
        self,
        trace: Trace,
        scenario: Scenario,
        llm: LLMPort,
        thresholds: ThresholdsConfig,
    ) -> EvalResult:
        # 1. Deterministic scorers (sync, instant)
        deterministic = [
            score_tool_correctness(trace, scenario.expectations, thresholds.tool_correctness),
            score_argument_correctness(trace, scenario.expectations, thresholds.argument_correctness),
            score_step_efficiency(trace, thresholds.step_efficiency),
            score_constraint_adherence(trace, scenario.constraints, thresholds.constraint_adherence),
        ]

        # 2. LLM judges (async, concurrent)
        llm_results = await asyncio.gather(
            judge_goal_achievement(trace, scenario, llm),
            judge_trajectory_efficiency(trace, scenario, llm),
            judge_persona_fit(trace, scenario, llm),
        )

        # 3. Aggregate
        all_dims = deterministic + list(llm_results)
        threshold_map = {
            "tool_correctness": thresholds.tool_correctness,
            "argument_correctness": thresholds.argument_correctness,
            "step_efficiency": thresholds.step_efficiency,
            "constraint_adherence": thresholds.constraint_adherence,
            "goal_achievement": thresholds.goal_achievement,
            "trajectory_efficiency": thresholds.trajectory_efficiency,
            "persona_fit": thresholds.persona_fit,
        }
        result = aggregate_scores(all_dims, threshold_map, thresholds.aggregate)
        result.scenario_id = scenario.id
        return result
