"""Aggregation service — combines DimensionScores into an EvalResult."""

from __future__ import annotations
from dryrun.domain.models.evaluation import DimensionScore, EvalResult


def aggregate_scores(
    dimensions: list[DimensionScore],
    thresholds: dict[str, float],
    aggregate_threshold: float,
) -> EvalResult:
    """Equal-weight average. Pass = aggregate >= threshold AND no dimension below its threshold."""
    if not dimensions:
        return EvalResult(scenario_id="", passed=False, aggregate_score=0.0, dimensions=[])

    aggregate = sum(d.score for d in dimensions) / len(dimensions)

    updated_dims: list[DimensionScore] = []
    all_pass = True
    for d in dimensions:
        dim_threshold = thresholds.get(d.dimension)
        dim_passed = d.score >= dim_threshold if dim_threshold is not None else d.passed
        updated_dims.append(
            DimensionScore(dimension=d.dimension, score=d.score, passed=dim_passed, reason=d.reason)
        )
        if not dim_passed:
            all_pass = False

    passed = aggregate >= aggregate_threshold and all_pass

    return EvalResult(scenario_id="", passed=passed, aggregate_score=aggregate, dimensions=updated_dims)
