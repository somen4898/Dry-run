"""QdrantAdapter — StorePort implementation backed by Qdrant vector database."""

from __future__ import annotations
from qdrant_client import AsyncQdrantClient as QdrantAsyncClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue,
)
from dryrun.domain.models.diff import FailureMatch
from dryrun.domain.models.evaluation import RunResult
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.ports.store import StorePort


class QdrantAdapter(StorePort):
    def __init__(self, url: str = "http://localhost:6333", prefix: str = "dryrun_"):
        self._client = QdrantAsyncClient(url=url)
        self._scenarios_col = f"{prefix}scenarios"
        self._runs_col = f"{prefix}runs"

    async def ensure_collections(self, vector_size: int = 384) -> None:
        """Create collections if they don't exist. Call once at startup."""
        collections = [c.name for c in (await self._client.get_collections()).collections]
        if self._scenarios_col not in collections:
            await self._client.create_collection(
                collection_name=self._scenarios_col,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        if self._runs_col not in collections:
            await self._client.create_collection(
                collection_name=self._runs_col,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    async def upsert_scenario(self, scenario: Scenario, embedding: list[float]) -> None:
        point = PointStruct(
            id=hash(scenario.id) & 0xFFFFFFFF,
            vector=embedding,
            payload={
                "scenario_id": scenario.id,
                "scenario_json": scenario.model_dump_json(),
                "golden": getattr(scenario, "golden", False),
            },
        )
        await self._client.upsert(collection_name=self._scenarios_col, points=[point])

    async def find_similar_scenarios(
        self, embedding: list[float], top_k: int = 5
    ) -> list[Scenario]:
        results = await self._client.search(
            collection_name=self._scenarios_col, query_vector=embedding, limit=top_k,
        )
        return [Scenario.model_validate_json(r.payload["scenario_json"]) for r in results]

    async def is_near_duplicate(self, embedding: list[float], threshold: float = 0.92) -> bool:
        results = await self._client.search(
            collection_name=self._scenarios_col, query_vector=embedding, limit=1,
            score_threshold=threshold,
        )
        return len(results) > 0

    async def save_run(self, result: RunResult) -> str:
        point = PointStruct(
            id=hash(result.run_id) & 0xFFFFFFFF,
            vector=[0.0],
            payload={
                "run_id": result.run_id,
                "timestamp": str(result.timestamp),
                "run_json": result.model_dump_json(),
            },
        )
        await self._client.upsert(collection_name=self._runs_col, points=[point])
        return result.run_id

    async def get_run(self, run_id: str) -> RunResult | None:
        results = await self._client.scroll(
            collection_name=self._runs_col,
            scroll_filter=Filter(must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]),
            limit=1,
        )
        points = results[0]
        if points:
            return RunResult.model_validate_json(points[0].payload["run_json"])
        return None

    async def get_latest_run(self) -> RunResult | None:
        results = await self._client.scroll(
            collection_name=self._runs_col, limit=100,
        )
        points = results[0]
        if not points:
            return None
        points.sort(key=lambda p: p.payload.get("timestamp", ""), reverse=True)
        return RunResult.model_validate_json(points[0].payload["run_json"])

    async def get_golden_scenarios(self) -> list[Scenario]:
        results = await self._client.scroll(
            collection_name=self._scenarios_col,
            scroll_filter=Filter(must=[FieldCondition(key="golden", match=MatchValue(value=True))]),
            limit=100,
        )
        return [Scenario.model_validate_json(p.payload["scenario_json"]) for p in results[0]]

    async def mark_golden(self, scenario_id: str) -> None:
        results = await self._client.scroll(
            collection_name=self._scenarios_col,
            scroll_filter=Filter(must=[FieldCondition(key="scenario_id", match=MatchValue(value=scenario_id))]),
            limit=1,
        )
        points = results[0]
        if points:
            await self._client.set_payload(
                collection_name=self._scenarios_col,
                payload={"golden": True},
                points=[points[0].id],
            )

    async def find_similar_failures(
        self, embedding: list[float], top_k: int = 3
    ) -> list[FailureMatch]:
        similar = await self._client.search(
            collection_name=self._scenarios_col, query_vector=embedding, limit=top_k * 3,
        )
        matches: list[FailureMatch] = []
        all_runs = await self._client.scroll(collection_name=self._runs_col, limit=100)
        runs = [RunResult.model_validate_json(p.payload["run_json"]) for p in all_runs[0]]

        for scored_point in similar:
            sid = scored_point.payload["scenario_id"]
            for run in runs:
                for er in run.eval_results:
                    if er.scenario_id == sid and not er.passed:
                        matches.append(FailureMatch(
                            scenario_id=sid,
                            run_id=run.run_id,
                            run_timestamp=str(run.timestamp),
                            similarity_score=scored_point.score,
                            failed_dimensions=[d.dimension for d in er.dimensions if not d.passed],
                            failure_reasons=[d.reason for d in er.dimensions if not d.passed],
                        ))
            if len(matches) >= top_k:
                break

        return matches[:top_k]
