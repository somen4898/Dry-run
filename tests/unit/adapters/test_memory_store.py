"""Tests for InMemoryStoreAdapter."""

import pytest
from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter
from dryrun.domain.ports.store import StorePort
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.evaluation import RunResult, EvalResult, DimensionScore


def _scenario(id: str, name: str = "Test") -> Scenario:
    return Scenario(
        id=id, name=name, description=f"Scenario {id}",
        persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(
            required_tools=["search"], required_tool_args={}, output_must_contain=[],
            terminal_state=None,
        ),
        constraints=Constraints(),
    )


def _run_result(run_id: str, scenario_ids: list[str], passed: bool = True) -> RunResult:
    return RunResult(
        run_id=run_id,
        timestamp="2026-05-17T10:00:00",
        total_scenarios=len(scenario_ids),
        passed=sum(1 for _ in scenario_ids) if passed else 0,
        failed=0 if passed else len(scenario_ids),
        aggregate_score=0.85 if passed else 0.4,
        per_dimension_scores={"tool_correctness": 0.9},
        eval_results=[
            EvalResult(
                scenario_id=sid, passed=passed, aggregate_score=0.85 if passed else 0.4,
                dimensions=[DimensionScore(dimension="tool_correctness", score=0.9, passed=True, reason="OK")],
            )
            for sid in scenario_ids
        ],
        token_cost_actual=100,
    )


class TestInMemoryStoreAdapter:
    def test_implements_store_port(self):
        store = InMemoryStoreAdapter()
        assert isinstance(store, StorePort)

    @pytest.mark.asyncio
    async def test_upsert_and_find_similar(self):
        store = InMemoryStoreAdapter()
        s = _scenario("s1")
        embedding = [1.0, 0.0, 0.0]
        await store.upsert_scenario(s, embedding)
        results = await store.find_similar_scenarios([0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == "s1"

    @pytest.mark.asyncio
    async def test_is_near_duplicate_true(self):
        store = InMemoryStoreAdapter()
        await store.upsert_scenario(_scenario("s1"), [1.0, 0.0, 0.0])
        assert await store.is_near_duplicate([1.0, 0.0, 0.0], threshold=0.99) is True

    @pytest.mark.asyncio
    async def test_is_near_duplicate_false(self):
        store = InMemoryStoreAdapter()
        await store.upsert_scenario(_scenario("s1"), [1.0, 0.0, 0.0])
        assert await store.is_near_duplicate([0.0, 1.0, 0.0], threshold=0.9) is False

    @pytest.mark.asyncio
    async def test_save_and_get_run(self):
        store = InMemoryStoreAdapter()
        run = _run_result("run-1", ["s1", "s2"])
        saved_id = await store.save_run(run)
        assert saved_id == "run-1"
        retrieved = await store.get_run("run-1")
        assert retrieved is not None
        assert retrieved.run_id == "run-1"

    @pytest.mark.asyncio
    async def test_get_latest_run(self):
        store = InMemoryStoreAdapter()
        await store.save_run(_run_result("run-1", ["s1"]))
        await store.save_run(_run_result("run-2", ["s1"]))
        latest = await store.get_latest_run()
        assert latest is not None
        assert latest.run_id == "run-2"

    @pytest.mark.asyncio
    async def test_golden_suite(self):
        store = InMemoryStoreAdapter()
        s = _scenario("s1")
        s.golden = True
        await store.upsert_scenario(s, [1.0, 0.0, 0.0])
        await store.mark_golden("s1")
        golden = await store.get_golden_scenarios()
        assert len(golden) == 1
        assert golden[0].id == "s1"

    @pytest.mark.asyncio
    async def test_find_similar_failures(self):
        store = InMemoryStoreAdapter()
        run = _run_result("run-1", ["s1"], passed=False)
        await store.save_run(run)
        await store.upsert_scenario(_scenario("s1"), [1.0, 0.0, 0.0])
        matches = await store.find_similar_failures([0.9, 0.1, 0.0], top_k=3)
        assert len(matches) >= 1
        assert matches[0].scenario_id == "s1"
