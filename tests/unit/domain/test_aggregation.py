"""Tests for aggregation service."""

import pytest
from dryrun.domain.models.evaluation import DimensionScore
from dryrun.domain.services.aggregation import aggregate_scores


def _dim(name: str, score: float) -> DimensionScore:
    return DimensionScore(dimension=name, score=score, passed=score >= 0.5, reason="test")


class TestAggregateScores:
    def test_all_pass(self):
        dims = [_dim("tool_correctness", 0.9), _dim("goal_achievement", 0.8)]
        thresholds = {"tool_correctness": 0.8, "goal_achievement": 0.7}
        result = aggregate_scores(dims, thresholds, aggregate_threshold=0.7)
        assert result.passed is True
        assert result.aggregate_score == pytest.approx(0.85)

    def test_aggregate_below_threshold_fails(self):
        dims = [_dim("tool_correctness", 0.5), _dim("goal_achievement", 0.5)]
        thresholds = {"tool_correctness": 0.4, "goal_achievement": 0.4}
        result = aggregate_scores(dims, thresholds, aggregate_threshold=0.7)
        assert result.passed is False

    def test_one_dimension_below_its_threshold_fails(self):
        dims = [_dim("tool_correctness", 0.5), _dim("goal_achievement", 0.95)]
        thresholds = {"tool_correctness": 0.8, "goal_achievement": 0.7}
        result = aggregate_scores(dims, thresholds, aggregate_threshold=0.5)
        assert result.passed is False

    def test_empty_dimensions(self):
        result = aggregate_scores([], {}, aggregate_threshold=0.5)
        assert result.passed is False
        assert result.aggregate_score == 0.0

    def test_equal_weights(self):
        dims = [_dim("a", 1.0), _dim("b", 0.0)]
        result = aggregate_scores(dims, {}, aggregate_threshold=0.0)
        assert result.aggregate_score == pytest.approx(0.5)
