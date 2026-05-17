"""InMemoryStoreAdapter — test-friendly store using dicts + cosine similarity."""

from __future__ import annotations
import math
from dryrun.domain.models.diff import FailureMatch
from dryrun.domain.models.evaluation import RunResult
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.ports.store import StorePort


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryStoreAdapter(StorePort):
    def __init__(self):
        self._scenarios: dict[str, tuple[Scenario, list[float]]] = {}
        self._runs: list[RunResult] = []
        self._golden_ids: set[str] = set()

    async def upsert_scenario(self, scenario: Scenario, embedding: list[float]) -> None:
        self._scenarios[scenario.id] = (scenario, embedding)

    async def find_similar_scenarios(
        self, embedding: list[float], top_k: int = 5
    ) -> list[Scenario]:
        scored = [
            (scenario, _cosine_similarity(embedding, emb))
            for scenario, emb in self._scenarios.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]

    async def is_near_duplicate(self, embedding: list[float], threshold: float = 0.92) -> bool:
        for _, emb in self._scenarios.values():
            if _cosine_similarity(embedding, emb) >= threshold:
                return True
        return False

    async def save_run(self, result: RunResult) -> str:
        self._runs.append(result)
        return result.run_id

    async def get_run(self, run_id: str) -> RunResult | None:
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    async def get_latest_run(self) -> RunResult | None:
        return self._runs[-1] if self._runs else None

    async def get_golden_scenarios(self) -> list[Scenario]:
        return [
            scenario
            for scenario, _ in self._scenarios.values()
            if scenario.id in self._golden_ids or getattr(scenario, "golden", False)
        ]

    async def mark_golden(self, scenario_id: str) -> None:
        self._golden_ids.add(scenario_id)

    async def find_similar_failures(
        self, embedding: list[float], top_k: int = 3
    ) -> list[FailureMatch]:
        matches: list[tuple[FailureMatch, float]] = []
        for run in self._runs:
            for er in run.eval_results:
                if not er.passed and er.scenario_id in self._scenarios:
                    _, emb = self._scenarios[er.scenario_id]
                    sim = _cosine_similarity(embedding, emb)
                    matches.append(
                        (
                            FailureMatch(
                                scenario_id=er.scenario_id,
                                run_id=run.run_id,
                                run_timestamp=str(run.timestamp),
                                similarity_score=sim,
                                failed_dimensions=[
                                    d.dimension for d in er.dimensions if not d.passed
                                ],
                                failure_reasons=[d.reason for d in er.dimensions if not d.passed],
                            ),
                            sim,
                        )
                    )
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matches[:top_k]]
