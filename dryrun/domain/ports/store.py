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
