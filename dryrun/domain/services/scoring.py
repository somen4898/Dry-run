"""Deterministic scoring functions — 4 dimensions, no LLM, pure logic."""

from __future__ import annotations
from collections import Counter
from dryrun.domain.models.evaluation import DimensionScore
from dryrun.domain.models.scenario import Constraints, Expectation
from dryrun.domain.models.trace import Trace


def score_tool_correctness(
    trace: Trace, expectations: Expectation, threshold: float = 0.8
) -> DimensionScore:
    """Did the agent call the tools the scenario expected? Set intersection."""
    required = set(expectations.required_tools)
    if not required:
        return DimensionScore(
            dimension="tool_correctness",
            score=1.0,
            passed=True,
            reason="No tools required",
        )

    called = {tc.tool_name for turn in trace.turns for tc in turn.tool_calls}
    intersection = called & required
    score = len(intersection) / len(required)
    missed = required - called

    return DimensionScore(
        dimension="tool_correctness",
        score=score,
        passed=score >= threshold,
        reason=f"Missed tools: {sorted(missed)}" if missed else "All required tools called",
    )


def score_argument_correctness(
    trace: Trace, expectations: Expectation, threshold: float = 0.75
) -> DimensionScore:
    """Did the agent call tools with correct arguments? Exact match on expected args."""
    required_args = expectations.required_tool_args
    if not required_args:
        return DimensionScore(
            dimension="argument_correctness",
            score=1.0,
            passed=True,
            reason="No argument expectations defined",
        )

    # Collect all tool calls by name
    calls_by_tool: dict[str, list[dict]] = {}
    for turn in trace.turns:
        for tc in turn.tool_calls:
            calls_by_tool.setdefault(tc.tool_name, []).append(tc.arguments)

    scores: list[float] = []
    reasons: list[str] = []

    for tool_name, expected_args in required_args.items():
        if tool_name not in calls_by_tool:
            scores.append(0.0)
            reasons.append(f"{tool_name}: never called")
            continue

        # Check if any call to this tool matches the expected args
        best_match = 0.0
        for actual_args in calls_by_tool[tool_name]:
            matched_keys = 0
            total_keys = len(expected_args)
            for key, expected_val in expected_args.items():
                if key in actual_args and actual_args[key] == expected_val:
                    matched_keys += 1
            if total_keys > 0:
                best_match = max(best_match, matched_keys / total_keys)

        scores.append(best_match)
        if best_match < 1.0:
            reasons.append(f"{tool_name}: partial match ({best_match:.0%})")

    overall_score = sum(scores) / len(scores) if scores else 1.0
    reason = "; ".join(reasons) if reasons else "All arguments correct"

    return DimensionScore(
        dimension="argument_correctness",
        score=overall_score,
        passed=overall_score >= threshold,
        reason=reason,
    )


def score_step_efficiency(trace: Trace, threshold: float = 0.7) -> DimensionScore:
    """Detect loops, thrashing, and redundant tool calls. Pure graph-path analysis."""
    penalty = 0.0
    reasons: list[str] = []

    # 1. Loop detection: same agent_id visited >3 times
    agent_visits = Counter(turn.agent_id for turn in trace.turns)
    for agent_id, count in agent_visits.items():
        if count > 3:
            penalty += 0.15 * (count - 3)
            reasons.append(f"Loop: {agent_id} visited {count} times")

    # 2. Thrashing detection: A→B→A→B oscillation
    if len(trace.turns) >= 4:
        agents = [t.agent_id for t in trace.turns]
        for i in range(len(agents) - 3):
            if (
                agents[i] == agents[i + 2]
                and agents[i + 1] == agents[i + 3]
                and agents[i] != agents[i + 1]
            ):
                penalty += 0.2
                reasons.append(f"Thrashing: {agents[i]}↔{agents[i + 1]}")
                break

    # 3. Redundancy: consecutive identical tool calls with identical arguments
    all_tool_calls = [
        (tc.tool_name, str(tc.arguments)) for turn in trace.turns for tc in turn.tool_calls
    ]
    for i in range(1, len(all_tool_calls)):
        if all_tool_calls[i] == all_tool_calls[i - 1]:
            penalty += 0.1
            if "Redundant" not in " ".join(reasons):
                reasons.append(f"Redundant: consecutive identical {all_tool_calls[i][0]} calls")

    score = max(0.0, 1.0 - penalty)
    reason = "; ".join(reasons) if reasons else "Clean path, no inefficiencies detected"

    return DimensionScore(
        dimension="step_efficiency",
        score=score,
        passed=score >= threshold,
        reason=reason,
    )


def score_constraint_adherence(
    trace: Trace, constraints: Constraints, threshold: float = 0.9
) -> DimensionScore:
    """Did the agent stay within max turns, timeout, and token budget? Pure counting."""
    violations: list[str] = []

    if trace.total_turns > constraints.max_turns:
        violations.append(f"Exceeded max turns: {trace.total_turns}/{constraints.max_turns}")

    if trace.total_latency_ms > constraints.timeout_seconds * 1000:
        violations.append(
            f"Exceeded timeout: {trace.total_latency_ms}ms/{constraints.timeout_seconds * 1000}ms"
        )

    if trace.total_tokens > constraints.max_tokens:
        violations.append(f"Exceeded token budget: {trace.total_tokens}/{constraints.max_tokens}")

    score = max(0.0, 1.0 - (len(violations) * 0.33))
    reason = "; ".join(violations) if violations else "All constraints met"

    return DimensionScore(
        dimension="constraint_adherence",
        score=score,
        passed=len(violations) == 0,
        reason=reason,
    )
