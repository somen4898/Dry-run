"""Deterministic scoring functions — 4 dimensions, no LLM, pure logic.

Each scorer uses research-backed metrics:
- Tool correctness: Recall-based (penalizes missing required tools, tolerates extras)
- Argument correctness: Per-key match ratio with best-call selection
- Step efficiency: Multi-signal penalty model (loops, thrashing, redundancy)
- Constraint adherence: Proportional violation scoring with hard/soft boundaries

Design decisions:
- Tool correctness uses RECALL not F1, because extra tools are exploration
  (a reasonable agent behavior), while missing required tools is a hard failure.
  This aligns with Google Vertex AI's trajectory_recall metric.
- Argument correctness uses best-match across multiple calls to the same tool,
  tolerating retry patterns where the agent self-corrects.
- Step efficiency uses a penalty-based model rather than a ratio model because
  ratio (actual/optimal) is unreliable when optimal is unknown or == 1.
- Constraint adherence uses proportional scoring for continuous constraints
  (tokens, time) and binary for discrete constraints (turns), following
  Amazon's graduated violation approach.
"""

from __future__ import annotations
from collections import Counter
from dryrun.domain.models.evaluation import DimensionScore
from dryrun.domain.models.scenario import Constraints, Expectation
from dryrun.domain.models.trace import Trace


def score_tool_correctness(
    trace: Trace, expectations: Expectation, threshold: float = 0.8
) -> DimensionScore:
    """Recall-based tool correctness: did the agent call all required tools?

    Formula: recall = |required ∩ called| / |required|

    Uses recall (not F1) because:
    - Missing a required tool is a clear failure
    - Calling extra tools is exploration, not a defect
    - This aligns with DeepEval's ToolCorrectnessMetric and
      Google Vertex AI's trajectory_recall

    Extra tools called are noted in the reason but don't reduce score.
    """
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
    recall = len(intersection) / len(required)
    missed = required - called
    extra = called - required

    # Build informative reason
    parts = []
    if missed:
        parts.append(f"Missing: {sorted(missed)}")
    if extra:
        parts.append(f"Extra (not penalized): {sorted(extra)}")
    if not parts:
        parts.append("All required tools called")

    return DimensionScore(
        dimension="tool_correctness",
        score=recall,
        passed=recall >= threshold,
        reason="; ".join(parts),
    )


def score_argument_correctness(
    trace: Trace, expectations: Expectation, threshold: float = 0.75
) -> DimensionScore:
    """Per-key argument match ratio with best-call selection.

    For each required tool+args pair:
    1. Find all calls to that tool across the trace
    2. For each call, compute key_match_ratio = matched_keys / total_expected_keys
    3. Take the BEST match (tolerates self-correction patterns)

    Final score = mean of best_match scores across all required tool+args pairs.

    This design:
    - Tolerates retry patterns (agent calls with wrong args, then corrects)
    - Gives partial credit for partially correct arguments
    - Uses exact value matching (type-sensitive)
    """
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

        if not expected_args:
            # Tool was called, no specific args required
            scores.append(1.0)
            continue

        # Find best-matching call to this tool
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
            reasons.append(f"{tool_name}: best match {best_match:.0%}")

    overall_score = sum(scores) / len(scores) if scores else 1.0
    reason = "; ".join(reasons) if reasons else "All arguments correct"

    return DimensionScore(
        dimension="argument_correctness",
        score=overall_score,
        passed=overall_score >= threshold,
        reason=reason,
    )


def score_step_efficiency(trace: Trace, threshold: float = 0.7) -> DimensionScore:
    """Multi-signal penalty model for execution path inefficiency.

    Detects three classes of inefficiency with calibrated penalties:

    1. LOOP DETECTION (penalty: 0.15 per excess visit beyond 3)
       An agent visiting the same node >3 times indicates a stuck loop.
       Threshold of 3 allows for natural revisits (e.g., tool→agent→tool→agent).

    2. THRASHING DETECTION (penalty: 0.20 per A→B→A→B oscillation)
       Rapid alternation between two agents signals routing confusion.
       Only penalized once per unique pair to avoid compounding.

    3. REDUNDANCY DETECTION (penalty: 0.10 per redundant call)
       Consecutive identical tool calls (same name + same args) are pure waste.
       Self-correction (same tool, different args) is NOT penalized.

    Score = max(0.0, 1.0 - total_penalty)

    This penalty model is preferred over ratio-based (actual/optimal) because:
    - Optimal step count is often unknown or debatable
    - Penalties are interpretable and tunable
    - Multiple signals compound naturally
    """
    penalty = 0.0
    reasons: list[str] = []

    # 1. Loop detection: same agent_id visited >3 times
    agent_visits = Counter(turn.agent_id for turn in trace.turns)
    for agent_id, count in agent_visits.items():
        if count > 3:
            excess = count - 3
            penalty += 0.15 * excess
            reasons.append(f"Loop: {agent_id} visited {count}x (excess: {excess})")

    # 2. Thrashing detection: A→B→A→B oscillation
    #    Only penalize once per unique thrashing pair
    if len(trace.turns) >= 4:
        agents = [t.agent_id for t in trace.turns]
        thrash_pairs: set[tuple[str, str]] = set()
        for i in range(len(agents) - 3):
            if (
                agents[i] == agents[i + 2]
                and agents[i + 1] == agents[i + 3]
                and agents[i] != agents[i + 1]
            ):
                pair = (min(agents[i], agents[i + 1]), max(agents[i], agents[i + 1]))
                if pair not in thrash_pairs:
                    thrash_pairs.add(pair)
                    penalty += 0.2
                    reasons.append(f"Thrashing: {agents[i]}↔{agents[i + 1]}")

    # 3. Redundancy: consecutive identical tool calls (name AND arguments)
    all_tool_calls = [
        (tc.tool_name, str(tc.arguments)) for turn in trace.turns for tc in turn.tool_calls
    ]
    redundant_count = 0
    for i in range(1, len(all_tool_calls)):
        if all_tool_calls[i] == all_tool_calls[i - 1]:
            redundant_count += 1

    if redundant_count > 0:
        penalty += 0.1 * redundant_count
        reasons.append(f"Redundant: {redundant_count} consecutive duplicate tool call(s)")

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
    """Proportional constraint violation scoring.

    For each constraint type, computes a violation severity:
    - Turns: binary (exceeded = 0.33 penalty). Turns are discrete so
      proportional doesn't apply well.
    - Timeout: proportional to overage. penalty = min(0.33, overage_ratio * 0.33)
      where overage_ratio = (actual - limit) / limit. Capped at 0.33.
    - Token budget: proportional to overage. Same formula as timeout.

    Score = max(0.0, 1.0 - total_penalty)
    Passed = no violations at all (binary pass/fail for constraint adherence)

    Proportional scoring for continuous constraints (time, tokens) is preferred
    because "1ms over timeout" should not be penalized equally to "2x over timeout".
    Turns remain binary because exceeding by even 1 turn may indicate a control failure.
    """
    violations: list[str] = []
    penalty = 0.0

    # Turns: binary penalty (discrete constraint)
    if trace.total_turns > constraints.max_turns:
        penalty += 0.33
        violations.append(
            f"Exceeded max turns: {trace.total_turns}/{constraints.max_turns}"
        )

    # Timeout: proportional penalty (continuous constraint)
    timeout_ms = constraints.timeout_seconds * 1000
    if trace.total_latency_ms > timeout_ms:
        overage_ratio = (trace.total_latency_ms - timeout_ms) / timeout_ms
        timeout_penalty = min(0.33, overage_ratio * 0.33)
        penalty += timeout_penalty
        violations.append(
            f"Exceeded timeout: {trace.total_latency_ms}ms/{timeout_ms}ms "
            f"(overage: {overage_ratio:.0%})"
        )

    # Token budget: proportional penalty (continuous constraint)
    if trace.total_tokens > constraints.max_tokens:
        overage_ratio = (trace.total_tokens - constraints.max_tokens) / constraints.max_tokens
        token_penalty = min(0.33, overage_ratio * 0.33)
        penalty += token_penalty
        violations.append(
            f"Exceeded token budget: {trace.total_tokens}/{constraints.max_tokens} "
            f"(overage: {overage_ratio:.0%})"
        )

    score = max(0.0, 1.0 - penalty)
    reason = "; ".join(violations) if violations else "All constraints met"

    return DimensionScore(
        dimension="constraint_adherence",
        score=score,
        passed=len(violations) == 0,
        reason=reason,
    )
