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

    common_ids = set(prev_map.keys()) & set(curr_map.keys())

    newly_failing: list[ScenarioDelta] = []
    newly_passing: list[ScenarioDelta] = []
    stable_pass = 0
    stable_fail = 0

    for sid in common_ids:
        prev_er = prev_map[sid]
        curr_er = curr_map[sid]

        delta = curr_er.aggregate_score - prev_er.aggregate_score

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

    # New scenarios in current count as stable
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
