"""Tests for diff service."""

import pytest
from dryrun.domain.models.evaluation import RunResult, EvalResult, DimensionScore
from dryrun.domain.services.diff import compute_diff


def _eval(sid: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(
        scenario_id=sid,
        passed=passed,
        aggregate_score=score,
        dimensions=[
            DimensionScore(dimension="tool_correctness", score=score, passed=passed, reason="test")
        ],
    )


def _run(run_id: str, evals: list[EvalResult]) -> RunResult:
    passed = sum(1 for e in evals if e.passed)
    return RunResult(
        run_id=run_id,
        timestamp="2026-05-17",
        total_scenarios=len(evals),
        passed=passed,
        failed=len(evals) - passed,
        aggregate_score=sum(e.aggregate_score for e in evals) / len(evals),
        per_dimension_scores={},
        eval_results=evals,
        token_cost_actual=0,
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
        assert diff.stable_pass == 1
        assert len(diff.newly_failing) == 0
