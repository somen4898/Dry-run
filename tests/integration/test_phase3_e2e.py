"""End-to-end test for Phase 3 — generation, storage, diff, similar-failure."""

import pytest
from unittest.mock import patch
from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter
from dryrun.application.generator import ScenarioGenerator
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.evaluation import RunResult, EvalResult, DimensionScore
from dryrun.domain.services.diff import compute_diff
from dryrun.domain.services.embedding import embed_scenario


MOCK_YAML = """
id: "gen-e2e-001"
name: "E2E Generated"
description: "Return a defective product"
persona:
  goal: "Return broken headphones"
  tone: "frustrated"
  knowledge_level: "novice"
  background: "First-time buyer"
opening_input: "These headphones are broken"
expectations:
  required_tools: ["check_order_status"]
  required_tool_args: {}
  output_must_contain: []
  terminal_state: null
constraints:
  max_turns: 6
"""


def _seed() -> Scenario:
    return Scenario(
        id="seed-001", name="Seed", description="Buy a laptop",
        persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[], terminal_state=None),
        constraints=Constraints(),
    )


class TestPhase3E2E:
    @pytest.mark.asyncio
    async def test_generate_store_diff_flow(self):
        """Full flow: generate → store → run → diff → similar failures."""
        store = InMemoryStoreAdapter()

        # 1. Generate a scenario
        generator = ScenarioGenerator(store=store, model="claude-haiku-4-5")
        with patch.object(generator, "_call_dspy", return_value=MOCK_YAML):
            generated = await generator.generate(seeds=[_seed()], count=1)

        assert len(generated) == 1
        assert generated[0].id == "gen-e2e-001"

        # 2. Simulate first run (pass)
        run1 = RunResult(
            run_id="run-1", timestamp="2026-05-17T10:00:00",
            total_scenarios=1, passed=1, failed=0, aggregate_score=0.85,
            per_dimension_scores={"tool_correctness": 0.9},
            eval_results=[EvalResult(
                scenario_id="gen-e2e-001", passed=True, aggregate_score=0.85,
                dimensions=[DimensionScore(dimension="tool_correctness", score=0.9, passed=True, reason="OK")],
            )],
            token_cost_actual=100,
        )
        await store.save_run(run1)

        # 3. Simulate second run (fail — regression)
        run2 = RunResult(
            run_id="run-2", timestamp="2026-05-17T11:00:00",
            total_scenarios=1, passed=0, failed=1, aggregate_score=0.4,
            per_dimension_scores={"tool_correctness": 0.4},
            eval_results=[EvalResult(
                scenario_id="gen-e2e-001", passed=False, aggregate_score=0.4,
                dimensions=[DimensionScore(dimension="tool_correctness", score=0.4, passed=False, reason="Missing tools")],
            )],
            token_cost_actual=100,
        )
        await store.save_run(run2)

        # 4. Compute diff
        diff = compute_diff(run1, run2)
        assert diff.score_delta == pytest.approx(-0.45)
        assert len(diff.newly_failing) == 1
        assert diff.newly_failing[0].scenario_id == "gen-e2e-001"

        # 5. Similar-failure lookup
        embedding = embed_scenario(generated[0])
        failures = await store.find_similar_failures(embedding, top_k=3)
        assert len(failures) >= 1
        assert failures[0].scenario_id == "gen-e2e-001"

    @pytest.mark.asyncio
    async def test_latest_run_retrieval(self):
        store = InMemoryStoreAdapter()
        run1 = RunResult(
            run_id="r1", timestamp="2026-05-17T09:00:00",
            total_scenarios=1, passed=1, failed=0, aggregate_score=0.9,
            per_dimension_scores={}, eval_results=[], token_cost_actual=0,
        )
        run2 = RunResult(
            run_id="r2", timestamp="2026-05-17T10:00:00",
            total_scenarios=1, passed=1, failed=0, aggregate_score=0.8,
            per_dimension_scores={}, eval_results=[], token_cost_actual=0,
        )
        await store.save_run(run1)
        await store.save_run(run2)
        latest = await store.get_latest_run()
        assert latest.run_id == "r2"
