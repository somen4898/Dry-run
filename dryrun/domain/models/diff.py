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
