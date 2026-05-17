"""Evaluation domain models — DimensionScore, EvalResult, RunResult."""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class DimensionScore(BaseModel):
    dimension: str
    score: float  # 0.0 – 1.0
    passed: bool
    reason: str


class EvalResult(BaseModel):
    scenario_id: str
    passed: bool
    aggregate_score: float
    dimensions: list[DimensionScore]
    similar_past_failures: list[dict] = []


class RunResult(BaseModel):
    run_id: str
    timestamp: datetime
    total_scenarios: int
    passed: int
    failed: int
    aggregate_score: float
    per_dimension_scores: dict[str, float]
    eval_results: list[EvalResult]
    token_cost_actual: int = 0
