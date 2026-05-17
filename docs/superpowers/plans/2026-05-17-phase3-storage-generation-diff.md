# Phase 3: Storage, Generation, Diff — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dryrun generate` produces valid new scenarios from seeds; `dryrun run --diff` produces a correct diff showing score delta and newly failing/passing scenarios; failed scenarios show top-3 similar past failures from Qdrant.

**Architecture:** Hexagonal. StorePort in domain/ports. QdrantAdapter and InMemoryStoreAdapter as outbound adapters. Embedding service in domain/services. Generator and diff logic in application layer. DSPy for scenario generation.

**Tech Stack:** Python 3.11+, Pydantic v2, Qdrant (qdrant-client), sentence-transformers (local embeddings), DSPy (dspy-ai), existing Anthropic/Click/Rich stack.

**Prerequisites:** Phase 2 PR (#4) must be merged first.

---

## Dependency Graph

```
Task 1 (Config) ────────────────────┐
Task 2 (Domain Models) ─────────────┤
Task 3 (StorePort) ─────────────────┤──→ Task 5 (Qdrant Adapter)
Task 4 (InMemory Adapter) ──────────┤──→ Task 6 (Embedding Service)
                                    │         │
                                    │         ▼
                                    ├──→ Task 7 (Generator)
                                    ├──→ Task 8 (Diff Service)
                                    ├──→ Task 9 (Similar-Failure Lookup)
                                    └──→ Task 10 (CLI Commands + Integration)
Task 11 (E2E Test) ── after all
```

**Execution order:** 1 → 2 → 3 → 4 → 5,6 (parallel) → 7,8,9 (parallel) → 10 → 11

---

### Task 1: Config Extensions

**Why first:** Every subsequent task references StoreConfig and GateConfig.

**Files:**
- Modify: `dryrun/config.py`
- Modify: `example/dryrun.yaml`
- Test: `tests/unit/test_config.py` (append)

- [ ] **Step 1: Write test**

Append to `tests/unit/test_config.py`:

```python
from dryrun.config import DryRunConfig, StoreConfig, GateConfig


class TestStoreConfig:
    def test_defaults(self):
        s = StoreConfig()
        assert s.provider == "qdrant"
        assert s.url == "http://localhost:6333"
        assert s.collection_prefix == "dryrun_"

    def test_config_includes_store(self):
        cfg = DryRunConfig(agent_module="x", agent_object="y")
        assert cfg.store.provider == "qdrant"


class TestGateConfig:
    def test_defaults(self):
        g = GateConfig()
        assert g.regression_threshold == 0.05
        assert g.golden_must_pass is True

    def test_config_includes_gate(self):
        cfg = DryRunConfig(agent_module="x", agent_object="y")
        assert cfg.gate.regression_threshold == 0.05
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/test_config.py::TestStoreConfig -v`

- [ ] **Step 3: Implement**

Add to `dryrun/config.py` before `DryRunConfig`:

```python
class StoreConfig(BaseModel):
    provider: Literal["qdrant", "memory"] = "qdrant"
    url: str = "http://localhost:6333"
    collection_prefix: str = "dryrun_"


class GateConfig(BaseModel):
    regression_threshold: float = 0.05
    golden_must_pass: bool = True
```

Add fields to `ModelConfig`:

```python
judge: str = "claude-haiku-4-5"
generator: str = "claude-haiku-4-5"
```

Add fields to `DryRunConfig`:

```python
store: StoreConfig = StoreConfig()
gate: GateConfig = GateConfig()
```

- [ ] **Step 4: Update `example/dryrun.yaml`**

Append store and gate sections:

```yaml
store:
  provider: "qdrant"
  url: "http://localhost:6333"
  collection_prefix: "dryrun_"

gate:
  regression_threshold: 0.05
  golden_must_pass: true
```

Also update the models section to use haiku defaults:

```yaml
models:
  provider: "anthropic"
  synthetic_user: "claude-haiku-4-5"
  agent: "claude-haiku-4-5"
  judge: "claude-haiku-4-5"
  generator: "claude-haiku-4-5"
```

- [ ] **Step 5: Run test — verify passes**

Run: `python3 -m pytest tests/unit/test_config.py -v`

- [ ] **Step 6: Commit**

```
feat(config): add StoreConfig, GateConfig, and model defaults for haiku
```

---

### Task 2: Domain Models (Diff + FailureMatch)

**Files:**
- Create: `dryrun/domain/models/diff.py`
- Test: `tests/unit/domain/test_diff_models.py`

- [ ] **Step 1: Write test**

Create `tests/unit/domain/test_diff_models.py`:

```python
"""Tests for diff domain models."""

from dryrun.domain.models.diff import ScenarioDelta, RunDiff, FailureMatch


class TestScenarioDelta:
    def test_create(self):
        d = ScenarioDelta(
            scenario_id="s1",
            previous_score=0.8,
            current_score=0.5,
            delta=-0.3,
            dimension_deltas={"tool_correctness": -0.2},
        )
        assert d.scenario_id == "s1"
        assert d.delta == -0.3

    def test_positive_delta(self):
        d = ScenarioDelta(
            scenario_id="s2",
            previous_score=0.4,
            current_score=0.9,
            delta=0.5,
            dimension_deltas={},
        )
        assert d.delta == 0.5


class TestRunDiff:
    def test_create(self):
        diff = RunDiff(
            previous_run_id="run-1",
            current_run_id="run-2",
            score_delta=-0.04,
            newly_failing=[],
            newly_passing=[],
            stable_pass=8,
            stable_fail=2,
        )
        assert diff.score_delta == -0.04
        assert diff.stable_pass == 8


class TestFailureMatch:
    def test_create(self):
        fm = FailureMatch(
            scenario_id="refund-003",
            run_id="run-old",
            run_timestamp="2026-05-15T10:00:00",
            similarity_score=0.91,
            failed_dimensions=["tool_correctness"],
            failure_reasons=["Missing: [initiate_refund]"],
        )
        assert fm.similarity_score == 0.91
        assert "tool_correctness" in fm.failed_dimensions
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/domain/test_diff_models.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/domain/models/diff.py`:

```python
"""Domain models for run diffs and failure matching."""

from __future__ import annotations
from pydantic import BaseModel


class ScenarioDelta(BaseModel):
    scenario_id: str
    previous_score: float
    current_score: float
    delta: float
    dimension_deltas: dict[str, float]


class RunDiff(BaseModel):
    previous_run_id: str
    current_run_id: str
    score_delta: float
    newly_failing: list[ScenarioDelta]
    newly_passing: list[ScenarioDelta]
    stable_pass: int
    stable_fail: int


class FailureMatch(BaseModel):
    scenario_id: str
    run_id: str
    run_timestamp: str
    similarity_score: float
    failed_dimensions: list[str]
    failure_reasons: list[str]
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(domain): add RunDiff, ScenarioDelta, FailureMatch models
```

---

### Task 3: StorePort ABC

**Files:**
- Create: `dryrun/domain/ports/store.py`
- Test: `tests/unit/domain/test_ports.py` (append)

- [ ] **Step 1: Write test**

Append to `tests/unit/domain/test_ports.py`:

```python
from dryrun.domain.ports.store import StorePort


class TestStorePortABC:
    def test_cannot_instantiate(self):
        import pytest
        with pytest.raises(TypeError):
            StorePort()

    def test_has_required_methods(self):
        methods = [
            "upsert_scenario",
            "find_similar_scenarios",
            "is_near_duplicate",
            "save_run",
            "get_run",
            "get_latest_run",
            "get_golden_scenarios",
            "mark_golden",
            "find_similar_failures",
        ]
        for method in methods:
            assert hasattr(StorePort, method)
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/domain/test_ports.py::TestStorePortABC -v`

- [ ] **Step 3: Implement**

Create `dryrun/domain/ports/store.py`:

```python
"""StorePort — contract for scenario/run storage with vector search."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.evaluation import RunResult
from dryrun.domain.models.diff import FailureMatch


class StorePort(ABC):
    @abstractmethod
    async def upsert_scenario(self, scenario: Scenario, embedding: list[float]) -> None:
        """Store or update a scenario with its embedding vector."""
        ...

    @abstractmethod
    async def find_similar_scenarios(
        self, embedding: list[float], top_k: int = 5
    ) -> list[Scenario]:
        """Find scenarios most similar to the given embedding."""
        ...

    @abstractmethod
    async def is_near_duplicate(self, embedding: list[float], threshold: float = 0.92) -> bool:
        """Check if a scenario with similarity >= threshold already exists."""
        ...

    @abstractmethod
    async def save_run(self, result: RunResult) -> str:
        """Persist a run result. Returns the run_id."""
        ...

    @abstractmethod
    async def get_run(self, run_id: str) -> RunResult | None:
        """Retrieve a specific run by ID."""
        ...

    @abstractmethod
    async def get_latest_run(self) -> RunResult | None:
        """Get the most recent run result."""
        ...

    @abstractmethod
    async def get_golden_scenarios(self) -> list[Scenario]:
        """Get all scenarios marked as golden."""
        ...

    @abstractmethod
    async def mark_golden(self, scenario_id: str) -> None:
        """Mark a scenario as part of the golden suite."""
        ...

    @abstractmethod
    async def find_similar_failures(
        self, embedding: list[float], top_k: int = 3
    ) -> list[FailureMatch]:
        """Find past failures most similar to the given embedding."""
        ...
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(domain): add StorePort ABC with vector search contract
```

---

### Task 4: InMemory Store Adapter

**Files:**
- Create: `dryrun/adapters/outbound/memory/__init__.py`
- Create: `dryrun/adapters/outbound/memory/store.py`
- Test: `tests/unit/adapters/test_memory_store.py`

- [ ] **Step 1: Write test**

Create `tests/unit/adapters/test_memory_store.py`:

```python
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
        expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[]),
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
        # Store a failed run
        run = _run_result("run-1", ["s1"], passed=False)
        await store.save_run(run)
        await store.upsert_scenario(_scenario("s1"), [1.0, 0.0, 0.0])
        # Search for similar failures
        matches = await store.find_similar_failures([0.9, 0.1, 0.0], top_k=3)
        assert len(matches) >= 1
        assert matches[0].scenario_id == "s1"
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/adapters/test_memory_store.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/adapters/outbound/memory/__init__.py` (empty).

Create `dryrun/adapters/outbound/memory/store.py`:

```python
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
                    matches.append((
                        FailureMatch(
                            scenario_id=er.scenario_id,
                            run_id=run.run_id,
                            run_timestamp=str(run.timestamp),
                            similarity_score=sim,
                            failed_dimensions=[d.dimension for d in er.dimensions if not d.passed],
                            failure_reasons=[d.reason for d in er.dimensions if not d.passed],
                        ),
                        sim,
                    ))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matches[:top_k]]
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(adapters): add InMemoryStoreAdapter with cosine similarity search
```

---

### Task 5: Qdrant Store Adapter

**Files:**
- Create: `dryrun/adapters/outbound/qdrant/__init__.py`
- Create: `dryrun/adapters/outbound/qdrant/store.py`
- Test: `tests/unit/adapters/test_qdrant_store.py`

- [ ] **Step 1: Write test**

Create `tests/unit/adapters/test_qdrant_store.py`:

```python
"""Tests for QdrantAdapter — skipped if Qdrant is not running."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dryrun.adapters.outbound.qdrant.store import QdrantAdapter
from dryrun.domain.ports.store import StorePort
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints


def _scenario(id: str) -> Scenario:
    return Scenario(
        id=id, name="Test", description=f"Scenario {id}",
        persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[]),
        constraints=Constraints(),
    )


class TestQdrantAdapter:
    def test_implements_store_port(self):
        with patch("dryrun.adapters.outbound.qdrant.store.QdrantAsyncClient"):
            adapter = QdrantAdapter(url="http://localhost:6333", prefix="test_")
            assert isinstance(adapter, StorePort)
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/adapters/test_qdrant_store.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/adapters/outbound/qdrant/__init__.py` (empty).

Create `dryrun/adapters/outbound/qdrant/store.py`:

```python
"""QdrantAdapter — StorePort implementation backed by Qdrant vector database."""

from __future__ import annotations
import json
from qdrant_client import AsyncQdrantClient as QdrantAsyncClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, ScoredPoint,
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
            vector=[0.0],  # runs don't need vector search
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
        # Sort by timestamp descending
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
        # Find the point by scenario_id and update payload
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
        # Search scenarios by embedding, then cross-reference with failed runs
        similar = await self._client.search(
            collection_name=self._scenarios_col, query_vector=embedding, limit=top_k * 3,
        )
        matches: list[FailureMatch] = []
        # Get all runs to find failures
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
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(adapters): add QdrantAdapter implementing StorePort with vector search
```

---

### Task 6: Embedding Service

**Files:**
- Create: `dryrun/domain/services/embedding.py`
- Test: `tests/unit/domain/test_embedding.py`

- [ ] **Step 1: Write test**

Create `tests/unit/domain/test_embedding.py`:

```python
"""Tests for embedding service."""

import pytest
from dryrun.domain.services.embedding import embed_scenario, embed_failure
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="s1", name="Test", description="Buy a laptop for college",
        persona=Persona(goal="Buy budget laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(required_tools=["search", "add_to_cart"], required_tool_args={}, output_must_contain=[]),
        constraints=Constraints(),
    )


class TestEmbedScenario:
    def test_returns_list_of_floats(self, scenario):
        embedding = embed_scenario(scenario)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_similar_scenarios_have_high_similarity(self):
        s1 = Scenario(
            id="s1", name="A", description="Buy a laptop",
            persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
            opening_input="Hi",
            expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[]),
            constraints=Constraints(),
        )
        s2 = Scenario(
            id="s2", name="B", description="Purchase a laptop computer",
            persona=Persona(goal="Purchase laptop", tone="polite", knowledge_level="novice", background="Student"),
            opening_input="Hello",
            expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[]),
            constraints=Constraints(),
        )
        e1 = embed_scenario(s1)
        e2 = embed_scenario(s2)
        # Cosine similarity should be high
        dot = sum(a * b for a, b in zip(e1, e2))
        assert dot > 0.8  # Normalized vectors, so dot product = cosine similarity


class TestEmbedFailure:
    def test_returns_list_of_floats(self, scenario):
        embedding = embed_failure(scenario, ["tool_correctness", "goal_achievement"])
        assert isinstance(embedding, list)
        assert len(embedding) == 384
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/domain/test_embedding.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/domain/services/embedding.py`:

```python
"""Embedding service — local sentence-transformers for scenario vectors."""

from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from dryrun.domain.models.scenario import Scenario


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached singleton)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_scenario(scenario: Scenario) -> list[float]:
    """Embed a scenario using description + goal + required tools."""
    text = (
        f"{scenario.description}. "
        f"Goal: {scenario.persona.goal}. "
        f"Tools: {', '.join(scenario.expectations.required_tools)}"
    )
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_failure(scenario: Scenario, failed_dimensions: list[str]) -> list[float]:
    """Embed a failure context for similar-failure lookup."""
    text = (
        f"{scenario.description}. "
        f"Goal: {scenario.persona.goal}. "
        f"Failed: {', '.join(failed_dimensions)}"
    )
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
```

- [ ] **Step 4: Run test — verify passes**

Note: First run will download the model (~90MB). Subsequent runs use cache.

- [ ] **Step 5: Commit**

```
feat(domain): add embedding service using sentence-transformers
```

---

### Task 7: DSPy Scenario Generator

**Files:**
- Create: `dryrun/application/generator.py`
- Test: `tests/unit/application/test_generator.py`

- [ ] **Step 1: Write test**

Create `tests/unit/application/test_generator.py`:

```python
"""Tests for DSPy-based scenario generator."""

import pytest
from unittest.mock import patch, MagicMock
from dryrun.application.generator import ScenarioGenerator
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter


def _seed_scenario() -> Scenario:
    return Scenario(
        id="seed-001", name="Seed", description="Buy a laptop for college",
        persona=Persona(goal="Buy budget laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi, I need a laptop",
        expectations=Expectation(required_tools=["search_inventory", "add_to_cart"], required_tool_args={}, output_must_contain=["order"]),
        constraints=Constraints(max_turns=8),
    )


MOCK_GENERATED_YAML = """
id: "gen-001"
name: "Generated Scenario"
description: "Buy headphones as a gift"
persona:
  goal: "Find wireless headphones under $50"
  tone: "casual"
  knowledge_level: "novice"
  background: "College student buying a birthday gift"
opening_input: "Hey, looking for some headphones"
expectations:
  required_tools: ["search_inventory"]
  required_tool_args: {}
  output_must_contain: []
constraints:
  max_turns: 8
"""


class TestScenarioGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_scenarios(self):
        store = InMemoryStoreAdapter()
        generator = ScenarioGenerator(store=store)

        with patch.object(generator, "_call_dspy", return_value=MOCK_GENERATED_YAML):
            results = await generator.generate(seeds=[_seed_scenario()], count=1)

        assert len(results) == 1
        assert results[0].id == "gen-001"
        assert results[0].persona.tone == "casual"

    @pytest.mark.asyncio
    async def test_skips_near_duplicates(self):
        store = InMemoryStoreAdapter()
        # Pre-store a scenario with similar embedding
        existing = _seed_scenario()
        await store.upsert_scenario(existing, [1.0] * 384)

        generator = ScenarioGenerator(store=store)

        with patch.object(generator, "_call_dspy", return_value=MOCK_GENERATED_YAML):
            with patch("dryrun.application.generator.embed_scenario", return_value=[1.0] * 384):
                results = await generator.generate(seeds=[_seed_scenario()], count=1)

        # Should be skipped as near-duplicate
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_validates_against_pydantic(self):
        store = InMemoryStoreAdapter()
        generator = ScenarioGenerator(store=store)

        invalid_yaml = "id: missing_fields\nname: Bad"

        with patch.object(generator, "_call_dspy", return_value=invalid_yaml):
            results = await generator.generate(seeds=[_seed_scenario()], count=1)

        # Invalid YAML should be skipped
        assert len(results) == 0
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/application/test_generator.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/application/generator.py`:

```python
"""DSPy-based scenario generator — produces varied scenarios from seeds."""

from __future__ import annotations
import logging
import random
import yaml
import dspy
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.ports.store import StorePort
from dryrun.domain.services.embedding import embed_scenario

logger = logging.getLogger(__name__)

VARIATION_STRATEGIES = [
    "tone_shift: Keep the same goal but change persona tone (e.g., polite→frustrated, casual→direct)",
    "edge_case: Add constraints that stress the agent (fewer max_turns, evasive goal reveal, impatient user)",
    "goal_variation: Same domain but a different user goal (e.g., buy→return, search→compare)",
    "persona_swap: Different knowledge level and background (e.g., novice→expert, student→professional)",
]


class GenerateScenario(dspy.Signature):
    """Generate a new, diverse test scenario for an AI agent based on seed examples.
    Output ONLY valid YAML for a scenario with fields: id, name, description, persona (goal, tone, knowledge_level, background), opening_input, expectations (required_tools, required_tool_args, output_must_contain), constraints (max_turns)."""

    seed_scenarios: str = dspy.InputField(desc="2-3 example scenarios as YAML for reference")
    variation_strategy: str = dspy.InputField(desc="How to make the new scenario different from seeds")
    new_scenario: str = dspy.OutputField(desc="Complete scenario as valid YAML")


class ScenarioGenerator:
    def __init__(self, store: StorePort, model: str = "claude-haiku-4-5"):
        self._store = store
        self._model = model

    async def generate(
        self, seeds: list[Scenario], count: int = 5, max_retries: int = 2
    ) -> list[Scenario]:
        """Generate new scenarios from seeds, dedup against store."""
        generated: list[Scenario] = []

        for i in range(count):
            strategy = random.choice(VARIATION_STRATEGIES)
            seed_yaml = self._format_seeds(random.sample(seeds, min(2, len(seeds))))

            for attempt in range(max_retries + 1):
                raw_yaml = await self._call_dspy(seed_yaml, strategy)
                scenario = self._parse_and_validate(raw_yaml, i)
                if scenario is None:
                    logger.warning("Generated invalid YAML (attempt %d)", attempt + 1)
                    continue

                # Dedup check
                embedding = embed_scenario(scenario)
                if await self._store.is_near_duplicate(embedding, threshold=0.92):
                    logger.info("Skipping near-duplicate: %s", scenario.id)
                    if attempt < max_retries:
                        strategy = random.choice(VARIATION_STRATEGIES)
                        continue
                    break

                # Store and collect
                await self._store.upsert_scenario(scenario, embedding)
                generated.append(scenario)
                break

        return generated

    async def _call_dspy(self, seed_yaml: str, strategy: str) -> str:
        """Call DSPy predictor. Override in tests."""
        lm = dspy.LM(f"anthropic/{self._model}")
        with dspy.context(lm=lm):
            predictor = dspy.Predict(GenerateScenario)
            result = predictor(seed_scenarios=seed_yaml, variation_strategy=strategy)
            return result.new_scenario

    def _format_seeds(self, seeds: list[Scenario]) -> str:
        """Format seed scenarios as YAML string."""
        return "\n---\n".join(
            yaml.dump(s.model_dump(exclude_none=True), default_flow_style=False)
            for s in seeds
        )

    def _parse_and_validate(self, raw_yaml: str, index: int) -> Scenario | None:
        """Parse YAML and validate as Scenario. Returns None if invalid."""
        try:
            # Strip markdown code fences if present
            cleaned = raw_yaml.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])

            data = yaml.safe_load(cleaned)
            if not isinstance(data, dict):
                return None
            return Scenario(**data)
        except Exception as e:
            logger.debug("Validation failed: %s", e)
            return None
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(application): add DSPy-based scenario generator with dedup gate
```

---

### Task 8: Diff Service

**Files:**
- Create: `dryrun/domain/services/diff.py`
- Test: `tests/unit/domain/test_diff_service.py`

- [ ] **Step 1: Write test**

Create `tests/unit/domain/test_diff_service.py`:

```python
"""Tests for diff service."""

import pytest
from dryrun.domain.models.evaluation import RunResult, EvalResult, DimensionScore
from dryrun.domain.models.diff import RunDiff
from dryrun.domain.services.diff import compute_diff


def _eval(sid: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(
        scenario_id=sid, passed=passed, aggregate_score=score,
        dimensions=[DimensionScore(dimension="tool_correctness", score=score, passed=passed, reason="test")],
    )


def _run(run_id: str, evals: list[EvalResult]) -> RunResult:
    passed = sum(1 for e in evals if e.passed)
    return RunResult(
        run_id=run_id, timestamp="2026-05-17", total_scenarios=len(evals),
        passed=passed, failed=len(evals) - passed,
        aggregate_score=sum(e.aggregate_score for e in evals) / len(evals),
        per_dimension_scores={}, eval_results=evals, token_cost_actual=0,
    )


class TestComputeDiff:
    def test_no_changes(self):
        prev = _run("r1", [_eval("s1", 0.9, True), _eval("s2", 0.8, True)])
        curr = _run("r2", [_eval("s1", 0.9, True), _eval("s2", 0.8, True)])
        diff = compute_diff(prev, curr)
        assert diff.score_delta == pytest.approx(0.0)
        assert diff.newly_failing == []
        assert diff.newly_passing == []
        assert diff.stable_pass == 2

    def test_newly_failing(self):
        prev = _run("r1", [_eval("s1", 0.9, True), _eval("s2", 0.8, True)])
        curr = _run("r2", [_eval("s1", 0.4, False), _eval("s2", 0.8, True)])
        diff = compute_diff(prev, curr)
        assert len(diff.newly_failing) == 1
        assert diff.newly_failing[0].scenario_id == "s1"
        assert diff.newly_failing[0].delta == pytest.approx(-0.5)

    def test_newly_passing(self):
        prev = _run("r1", [_eval("s1", 0.4, False)])
        curr = _run("r2", [_eval("s1", 0.9, True)])
        diff = compute_diff(prev, curr)
        assert len(diff.newly_passing) == 1
        assert diff.newly_passing[0].scenario_id == "s1"

    def test_score_delta(self):
        prev = _run("r1", [_eval("s1", 0.9, True)])
        curr = _run("r2", [_eval("s1", 0.7, True)])
        diff = compute_diff(prev, curr)
        assert diff.score_delta == pytest.approx(-0.2)

    def test_new_scenario_in_current(self):
        prev = _run("r1", [_eval("s1", 0.9, True)])
        curr = _run("r2", [_eval("s1", 0.9, True), _eval("s2", 0.5, False)])
        diff = compute_diff(prev, curr)
        # s2 is new, not "newly failing" — it has no previous state
        assert diff.stable_pass == 1
        assert len(diff.newly_failing) == 0
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/domain/test_diff_service.py -v`

- [ ] **Step 3: Implement**

Create `dryrun/domain/services/diff.py`:

```python
"""Diff service — compares two RunResults and produces a RunDiff."""

from __future__ import annotations
from dryrun.domain.models.evaluation import RunResult
from dryrun.domain.models.diff import RunDiff, ScenarioDelta


def compute_diff(previous: RunResult, current: RunResult) -> RunDiff:
    """Compare two runs and produce a structured diff.

    Only compares scenarios present in BOTH runs. New scenarios in current
    are counted as stable (not "newly failing/passing" since there's no baseline).
    """
    prev_map = {er.scenario_id: er for er in previous.eval_results}
    curr_map = {er.scenario_id: er for er in current.eval_results}

    # Only compare scenarios in both runs
    common_ids = set(prev_map.keys()) & set(curr_map.keys())

    newly_failing: list[ScenarioDelta] = []
    newly_passing: list[ScenarioDelta] = []
    stable_pass = 0
    stable_fail = 0

    for sid in common_ids:
        prev_er = prev_map[sid]
        curr_er = curr_map[sid]

        delta = curr_er.aggregate_score - prev_er.aggregate_score

        # Compute per-dimension deltas
        prev_dims = {d.dimension: d.score for d in prev_er.dimensions}
        curr_dims = {d.dimension: d.score for d in curr_er.dimensions}
        dim_deltas = {
            dim: curr_dims.get(dim, 0) - prev_dims.get(dim, 0)
            for dim in set(prev_dims.keys()) | set(curr_dims.keys())
        }

        if prev_er.passed and not curr_er.passed:
            newly_failing.append(ScenarioDelta(
                scenario_id=sid,
                previous_score=prev_er.aggregate_score,
                current_score=curr_er.aggregate_score,
                delta=delta,
                dimension_deltas=dim_deltas,
            ))
        elif not prev_er.passed and curr_er.passed:
            newly_passing.append(ScenarioDelta(
                scenario_id=sid,
                previous_score=prev_er.aggregate_score,
                current_score=curr_er.aggregate_score,
                delta=delta,
                dimension_deltas=dim_deltas,
            ))
        elif curr_er.passed:
            stable_pass += 1
        else:
            stable_fail += 1

    # Scenarios only in current (new) count as stable
    new_in_current = set(curr_map.keys()) - common_ids
    for sid in new_in_current:
        if curr_map[sid].passed:
            stable_pass += 1
        else:
            stable_fail += 1

    score_delta = current.aggregate_score - previous.aggregate_score

    return RunDiff(
        previous_run_id=previous.run_id,
        current_run_id=current.run_id,
        score_delta=score_delta,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        stable_pass=stable_pass,
        stable_fail=stable_fail,
    )
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(domain): add diff service for run-to-run comparison
```

---

### Task 9: Similar-Failure Lookup Integration

**Files:**
- Modify: `dryrun/adapters/outbound/reporters/terminal.py`
- Test: `tests/unit/adapters/test_terminal_reporter.py` (append)

- [ ] **Step 1: Write test**

Append to `tests/unit/adapters/test_terminal_reporter.py`:

```python
from dryrun.domain.models.diff import FailureMatch


class TestTerminalReporterWithFailures:
    def test_report_scenario_with_similar_failures(self, eval_result):
        """report_scenario should not crash with similar_failures attached."""
        reporter = TerminalReporter()
        failures = [
            FailureMatch(
                scenario_id="old-001", run_id="run-old",
                run_timestamp="2026-05-15", similarity_score=0.89,
                failed_dimensions=["tool_correctness"],
                failure_reasons=["Missing: [refund_tool]"],
            ),
        ]
        # Should not raise
        reporter.report_scenario(eval_result, similar_failures=failures)
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/adapters/test_terminal_reporter.py::TestTerminalReporterWithFailures -v`

- [ ] **Step 3: Implement**

Modify `dryrun/adapters/outbound/reporters/terminal.py` — update `report_scenario` to accept optional `similar_failures` parameter:

```python
def report_scenario(self, result: EvalResult, similar_failures: list[FailureMatch] | None = None) -> None:
    """Print a single scenario result as a compact table."""
    # ... existing code ...

    # Append similar failures section if provided
    if similar_failures and not result.passed:
        self._console.print("\n  [dim]Similar past failures:[/dim]")
        for fm in similar_failures:
            self._console.print(
                f"    [dim]→ {fm.scenario_id} (run {fm.run_timestamp}): "
                f"{', '.join(fm.failed_dimensions)}[/dim]"
            )
```

Also add the import at the top:

```python
from dryrun.domain.models.diff import FailureMatch
```

- [ ] **Step 4: Run test — verify passes**

- [ ] **Step 5: Commit**

```
feat(reporters): add similar-failure display to TerminalReporter
```

---

### Task 10: CLI Commands (generate, run --diff)

**Files:**
- Modify: `dryrun/adapters/inbound/cli/commands.py`
- Create: `dryrun/adapters/outbound/store_factory.py`
- Test: `tests/unit/adapters/test_cli_generate.py`

- [ ] **Step 1: Write test**

Create `tests/unit/adapters/test_cli_generate.py`:

```python
"""Tests for CLI generate command."""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


class TestGenerateCommand:
    def test_generate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--seeds" in result.output
        assert "--count" in result.output
        assert "--output" in result.output
```

- [ ] **Step 2: Run test — verify fails**

Run: `python3 -m pytest tests/unit/adapters/test_cli_generate.py -v`

- [ ] **Step 3: Implement store factory**

Create `dryrun/adapters/outbound/store_factory.py`:

```python
"""Store factory — creates the right StorePort based on config."""

from __future__ import annotations
from dryrun.config import StoreConfig
from dryrun.domain.ports.store import StorePort


def create_store(config: StoreConfig) -> StorePort:
    """Create a StorePort implementation based on config."""
    if config.provider == "memory":
        from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter
        return InMemoryStoreAdapter()
    elif config.provider == "qdrant":
        from dryrun.adapters.outbound.qdrant.store import QdrantAdapter
        return QdrantAdapter(url=config.url, prefix=config.collection_prefix)
    else:
        raise ValueError(f"Unknown store provider: '{config.provider}'")
```

- [ ] **Step 4: Add CLI commands**

Add to `dryrun/adapters/inbound/cli/commands.py`:

```python
@cli.command()
@click.option("--seeds", type=click.Path(exists=True), required=True, help="Directory of seed scenarios")
@click.option("--count", type=int, default=5, help="Number of scenarios to generate")
@click.option("--output", type=click.Path(), required=True, help="Output directory for generated scenarios")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
def generate(seeds: str, count: int, output: str, config_path: str | None):
    """Generate new scenarios from seed examples using DSPy."""
    import yaml as _yaml
    from dryrun.application.generator import ScenarioGenerator
    from dryrun.adapters.outbound.store_factory import create_store
    from dryrun.domain.models.scenario import Scenario

    config = _load_config(config_path)
    store = create_store(config.store)

    # Load seeds
    seeds_dir = Path(seeds)
    seed_scenarios = [
        Scenario(**_yaml.safe_load(f.read_text()))
        for f in sorted(seeds_dir.glob("*.yaml"))
    ]

    console.print(f"\n[bold]Generating {count} scenarios from {len(seed_scenarios)} seeds...[/bold]")

    generator = ScenarioGenerator(store=store, model=config.models.generator)
    results = asyncio.run(generator.generate(seeds=seed_scenarios, count=count))

    # Write to output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in results:
        out_path = output_dir / f"{scenario.id}.yaml"
        out_path.write_text(_yaml.dump(scenario.model_dump(exclude_none=True), default_flow_style=False))
        console.print(f"  [green]✓[/green] {out_path}")

    console.print(f"\n[bold]Generated {len(results)} scenarios[/bold]")
```

Also add `--diff` flag to the `run` command and wire in store + diff logic:

```python
# In the run command, add:
@click.option("--diff/--no-diff", default=False, help="Show diff against previous run")
```

And in the suite mode branch:

```python
if target.is_dir():
    console.print(f"\n[bold]Running suite:[/bold] {target} (concurrency: {max_concurrent})")
    run_result = asyncio.run(runner.run_suite(target, max_concurrent=max_concurrent))

    # Store run and compute diff if requested
    if diff:
        from dryrun.adapters.outbound.store_factory import create_store
        from dryrun.domain.services.diff import compute_diff

        store = create_store(config.store)
        previous = asyncio.run(store.get_latest_run())
        asyncio.run(store.save_run(run_result))
        if previous:
            run_diff = compute_diff(previous, run_result)
            _print_diff(run_diff)

    _print_run_result(run_result)
```

Extract config loading into a helper:

```python
def _load_config(config_path: str | None) -> DryRunConfig:
    if config_path:
        return DryRunConfig.from_yaml(Path(config_path))
    for candidate in [Path("dryrun.yaml"), Path("example/dryrun.yaml")]:
        if candidate.exists():
            return DryRunConfig.from_yaml(candidate)
    console.print("[red]No dryrun.yaml config found. Use --config.[/red]")
    sys.exit(1)
```

- [ ] **Step 5: Run test — verify passes**

- [ ] **Step 6: Commit**

```
feat(cli): add generate command and --diff flag to run command
```

---

### Task 11: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_phase3_e2e.py`

- [ ] **Step 1: Write e2e test**

Create `tests/integration/test_phase3_e2e.py`:

```python
"""End-to-end test for Phase 3 — generation, storage, diff, similar-failure."""

import pytest
from pathlib import Path
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
constraints:
  max_turns: 6
"""


def _seed() -> Scenario:
    return Scenario(
        id="seed-001", name="Seed", description="Buy a laptop",
        persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[]),
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
```

- [ ] **Step 2: Run — verify passes**

Run: `python3 -m pytest tests/integration/test_phase3_e2e.py -v`

- [ ] **Step 3: Commit**

```
test(integration): add Phase 3 end-to-end test for generate-store-diff-failure flow
```

---

## Verification

After all tasks complete:

```bash
python3 -m pytest tests/ -v --timeout=30
```

Expected: ~110+ tests passing.

Dependencies to add to `pyproject.toml`:
```toml
qdrant-client = ">=1.9"
sentence-transformers = ">=3.0"
dspy-ai = ">=2.5"
```

---

## Phase 4 Plan

Phase 4 (CI Gate, Capture, Package) will be a separate plan document written after Phase 3 is complete.
