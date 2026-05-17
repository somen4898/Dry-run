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
