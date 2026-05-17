"""Tests for TerminalReporter."""

import pytest
from io import StringIO
from unittest.mock import patch
from dryrun.domain.models.evaluation import DimensionScore, EvalResult, RunResult
from dryrun.adapters.outbound.reporters.terminal import TerminalReporter


@pytest.fixture
def eval_result() -> EvalResult:
    return EvalResult(
        scenario_id="test-001",
        passed=True,
        aggregate_score=0.85,
        dimensions=[
            DimensionScore(dimension="tool_correctness", score=0.9, passed=True, reason="All tools called"),
            DimensionScore(dimension="goal_achievement", score=0.8, passed=True, reason="Goal met"),
        ],
    )


@pytest.fixture
def run_result(eval_result) -> RunResult:
    return RunResult(
        run_id="run-001",
        timestamp="2026-05-17T10:00:00",
        total_scenarios=2,
        passed=1,
        failed=1,
        aggregate_score=0.75,
        per_dimension_scores={"tool_correctness": 0.85, "goal_achievement": 0.65},
        eval_results=[
            eval_result,
            EvalResult(
                scenario_id="test-002",
                passed=False,
                aggregate_score=0.55,
                dimensions=[
                    DimensionScore(dimension="tool_correctness", score=0.8, passed=True, reason="OK"),
                    DimensionScore(dimension="goal_achievement", score=0.3, passed=False, reason="Goal not met"),
                ],
            ),
        ],
        token_cost_actual=1000,
    )


class TestTerminalReporter:
    def test_report_scenario_does_not_raise(self, eval_result):
        reporter = TerminalReporter()
        # Should not raise
        reporter.report_scenario(eval_result)

    def test_report_suite_does_not_raise(self, run_result):
        reporter = TerminalReporter()
        # Should not raise
        reporter.report_suite(run_result)

    def test_implements_reporter_port(self):
        from dryrun.domain.ports.reporter import ReporterPort
        reporter = TerminalReporter()
        assert isinstance(reporter, ReporterPort)
