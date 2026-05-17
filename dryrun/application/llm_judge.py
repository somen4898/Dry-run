"""LLM-based judge evaluators for subjective scoring dimensions."""

from __future__ import annotations

import json

from dryrun.domain.models.evaluation import DimensionScore
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.trace import Trace
from dryrun.domain.ports.llm import LLMPort


async def _call_judge(
    llm: LLMPort,
    system_prompt: str,
    user_prompt: str,
    dimension: str,
    max_retries: int = 1,
) -> DimensionScore:
    """Call LLM judge with one-retry logic for JSON parse failures."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        response = await llm.complete(messages, temperature=0.0)
        try:
            data = json.loads(response)
            score = float(data["score"])
            reason = str(data["reason"])
            return DimensionScore(
                dimension=dimension,
                score=max(0.0, min(1.0, score)),
                passed=score >= 0.5,
                reason=reason,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. Please respond with ONLY a JSON"
                            ' object: {"score": <float 0.0-1.0>, "reason": "<explanation>"}'
                        ),
                    }
                )
            else:
                return DimensionScore(
                    dimension=dimension,
                    score=0.0,
                    passed=False,
                    reason=f"Judge failed to produce valid JSON after {max_retries + 1} attempts",
                )


def _format_trace(trace: Trace) -> str:
    """Format trace into readable text for the judge."""
    lines = []
    for turn in trace.turns:
        lines.append(f"Turn {turn.turn_number} (agent: {turn.agent_id}):")
        lines.append(f"  Input: {turn.input_text}")
        lines.append(f"  Output: {turn.output_text}")
        if turn.tool_calls:
            for tc in turn.tool_calls:
                lines.append(
                    f"  Tool: {tc.tool_name}({json.dumps(tc.arguments)})"
                    f" \u2192 {json.dumps(tc.output)}"
                )
        lines.append("")
    lines.append(f"Terminal reason: {trace.terminal_reason}")
    lines.append(f"Total turns: {trace.total_turns}, Total tokens: {trace.total_tokens}")
    return "\n".join(lines)


async def judge_goal_achievement(trace: Trace, scenario: Scenario, llm: LLMPort) -> DimensionScore:
    """Judge whether the agent achieved the user's goal."""
    system_prompt = """You are an expert evaluator assessing whether an AI agent achieved the user's goal.

## SCORING RUBRIC

Score on a scale of 0.0 to 1.0:
- 1.0: Goal fully achieved, all requirements met
- 0.8: Goal substantially achieved with minor gaps
- 0.6: Goal partially achieved, significant requirements unmet
- 0.4: Minimal progress toward goal
- 0.2: Agent attempted but largely failed
- 0.0: No progress or completely wrong direction

## EVALUATION CRITERIA

1. Did the agent understand the user's stated goal?
2. Were all required actions taken to fulfill the goal?
3. Was the final outcome satisfactory from the user's perspective?
4. Did the terminal reason indicate success or failure?

## INDEPENDENCE CONSTRAINT

Score based ONLY on goal achievement. Do NOT factor in:
- How efficiently the agent reached the goal (that's trajectory_efficiency)
- Whether the agent's tone was appropriate (that's persona_fit)
- Whether specific tools were called correctly (that's tool_correctness)

## OUTPUT FORMAT

Respond with ONLY a JSON object:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence explanation>"}"""

    user_prompt = f"""## SCENARIO
Goal: {scenario.persona.goal}
Expected output must contain: {scenario.expectations.output_must_contain}
Terminal state expected: {scenario.expectations.terminal_state}

## AGENT TRACE
{_format_trace(trace)}

Evaluate goal achievement:"""

    return await _call_judge(llm, system_prompt, user_prompt, "goal_achievement")


async def judge_trajectory_efficiency(
    trace: Trace, scenario: Scenario, llm: LLMPort
) -> DimensionScore:
    """Judge whether the agent took an efficient path to the goal."""
    system_prompt = """You are an expert evaluator assessing the efficiency of an AI agent's trajectory.

## SCORING RUBRIC

Score on a scale of 0.0 to 1.0:
- 1.0: Optimal path -- no wasted steps, direct route to goal
- 0.8: Near-optimal -- minor unnecessary steps but mostly direct
- 0.6: Acceptable -- some detours or redundancy but goal reached
- 0.4: Inefficient -- significant wasted effort, unnecessary back-and-forth
- 0.2: Very inefficient -- excessive steps, confusion, repeated failures
- 0.0: Completely lost -- no coherent strategy visible

## EVALUATION CRITERIA

1. Were there unnecessary tool calls or redundant operations?
2. Did the agent take a direct path or meander?
3. Were there failed attempts that a better strategy would have avoided?
4. Is the number of turns reasonable for the task complexity?

## INDEPENDENCE CONSTRAINT

Score based ONLY on path efficiency. Do NOT factor in:
- Whether the goal was actually achieved (that's goal_achievement)
- Whether tools were called with correct arguments (that's argument_correctness)
- Whether the agent's communication style was appropriate (that's persona_fit)

## OUTPUT FORMAT

Respond with ONLY a JSON object:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence explanation>"}"""

    user_prompt = f"""## SCENARIO
Goal: {scenario.persona.goal}
Max turns allowed: {scenario.constraints.max_turns}
Required tools: {scenario.expectations.required_tools}

## AGENT TRACE
{_format_trace(trace)}

Evaluate trajectory efficiency:"""

    return await _call_judge(llm, system_prompt, user_prompt, "trajectory_efficiency")


async def judge_persona_fit(trace: Trace, scenario: Scenario, llm: LLMPort) -> DimensionScore:
    """Judge whether the agent's responses were appropriate for the user persona."""
    system_prompt = """You are an expert evaluator assessing whether an AI agent responded appropriately to the user's persona.

## SCORING RUBRIC

Score on a scale of 0.0 to 1.0:
- 1.0: Perfect fit -- tone, complexity, and approach perfectly matched persona
- 0.8: Good fit -- mostly appropriate with minor mismatches
- 0.6: Acceptable -- generally appropriate but some notable mismatches
- 0.4: Poor fit -- frequent mismatches in tone or complexity
- 0.2: Bad fit -- agent largely ignored persona characteristics
- 0.0: Complete mismatch -- agent responses entirely inappropriate for persona

## EVALUATION CRITERIA

1. Did the agent match the expected tone? (e.g., empathetic for frustrated users, direct for experts)
2. Was the complexity of explanations appropriate for the knowledge level?
3. Did the agent acknowledge the user's emotional state?
4. Was the communication style suitable for the background?

## INDEPENDENCE CONSTRAINT

Score based ONLY on persona fit. Do NOT factor in:
- Whether the goal was achieved (that's goal_achievement)
- Whether the path was efficient (that's trajectory_efficiency)
- Whether correct tools were used (that's tool_correctness)

## OUTPUT FORMAT

Respond with ONLY a JSON object:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence explanation>"}"""

    user_prompt = f"""## PERSONA
Goal: {scenario.persona.goal}
Tone: {scenario.persona.tone}
Knowledge level: {scenario.persona.knowledge_level}
Background: {scenario.persona.background}

## AGENT TRACE (focus on agent's output_text)
{_format_trace(trace)}

Evaluate persona fit:"""

    return await _call_judge(llm, system_prompt, user_prompt, "persona_fit")
