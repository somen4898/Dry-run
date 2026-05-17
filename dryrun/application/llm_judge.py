"""LLM-based judge evaluators for subjective scoring dimensions.

Implements three judge functions using research-backed LLM-as-Judge patterns:
- Chain-of-thought reasoning before scoring (G-Eval pattern, Liu et al.)
- Few-shot calibration anchors to ground the scoring distribution
- Independence constraints to prevent cross-dimension contamination
- Explicit bias disclaimers (length, agreeableness, position)
- Structured rubrics with behavioral anchors at each score level
"""

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
    """Call LLM judge with one-retry logic for JSON parse failures.

    Uses temperature=0.0 for deterministic scoring consistency.
    Parses the final JSON object from the response, allowing chain-of-thought
    text to precede it.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        response = await llm.complete(messages, temperature=0.0)
        try:
            # Extract JSON from response — supports CoT followed by JSON
            data = _extract_json(response)
            score = float(data["score"])
            reason = str(data["reason"])
            return DimensionScore(
                dimension=dimension,
                score=max(0.0, min(1.0, score)),
                passed=score >= 0.5,
                reason=reason,
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response could not be parsed. Please respond with your "
                            "chain-of-thought analysis followed by a JSON object on its own line:\n"
                            '{"score": <float 0.0-1.0>, "reason": "<1-2 sentence summary>"}'
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


def _extract_json(text: str) -> dict:
    """Extract the last JSON object from text that may contain CoT reasoning.

    Searches backwards for the last complete JSON object in the response,
    allowing free-form reasoning text before the structured output.
    """
    # Try parsing the entire response first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the last { ... } block
    last_brace = text.rfind("}")
    if last_brace == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    # Search backwards for matching opening brace
    depth = 0
    for i in range(last_brace, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
        if depth == 0:
            return json.loads(text[i : last_brace + 1])

    raise json.JSONDecodeError("No complete JSON object found", text, 0)


def _format_trace(trace: Trace) -> str:
    """Format trace into structured readable text for the judge.

    Includes turn numbers, agent IDs, inputs/outputs, and tool call details
    to give the judge full visibility into the agent's behavior.
    """
    lines = []
    for turn in trace.turns:
        lines.append(f"--- Turn {turn.turn_number} (agent: {turn.agent_id}) ---")
        lines.append(f"User input: {turn.input_text}")
        lines.append(f"Agent response: {turn.output_text}")
        if turn.tool_calls:
            lines.append("Tool calls:")
            for tc in turn.tool_calls:
                lines.append(
                    f"  - {tc.tool_name}({json.dumps(tc.arguments, default=str)})"
                    f" → {json.dumps(tc.output, default=str)}"
                )
        lines.append("")
    lines.append(f"Terminal reason: {trace.terminal_reason}")
    lines.append(f"Total turns: {trace.total_turns} | Tokens: {trace.total_tokens}")
    return "\n".join(lines)


# =============================================================================
# JUDGE: GOAL ACHIEVEMENT
# =============================================================================

_GOAL_ACHIEVEMENT_SYSTEM = """You are a rigorous evaluator specializing in task completion assessment for AI agents.

## YOUR TASK

Determine whether an AI agent successfully achieved the user's stated goal, based on the conversation trace and expected outcomes.

## EVALUATION PROTOCOL

You MUST follow this protocol exactly:
1. First, identify the user's explicit goal from the scenario description
2. Analyze the trace to determine what actions the agent took toward that goal
3. Check the terminal reason — did the conversation end because the goal was met?
4. Check whether required output content appeared in the agent's responses
5. Write your reasoning (2-4 sentences analyzing the evidence)
6. THEN assign a score

## SCORING RUBRIC (Behavioral Anchors)

- **1.0**: Goal unambiguously achieved. Terminal reason is "goal_met", all expected output content present, user's stated objective fully satisfied.
- **0.8**: Goal substantially achieved. The core objective was met but with minor gaps (e.g., missing a confirmation message, one optional sub-goal incomplete).
- **0.6**: Goal partially achieved. The agent made meaningful progress and completed key steps, but left significant parts of the goal unfinished.
- **0.4**: Minimal progress. The agent understood the goal and started working on it, but did not complete any major sub-objective.
- **0.2**: Goal barely addressed. Agent acknowledged the request but took ineffective or wrong actions.
- **0.0**: No progress. Agent failed to understand or address the goal, or went in a completely wrong direction.

## CALIBRATION EXAMPLES

Example A (score: 1.0): User wanted to buy a laptop. Agent searched inventory, presented options, added selected item to cart, and completed checkout. Terminal reason: "goal_met". Output contains "order".
→ {"score": 1.0, "reason": "Goal fully achieved: agent completed search, selection, cart, and checkout flow. Terminal reason confirms success."}

Example B (score: 0.4): User wanted a refund. Agent looked up the order but then asked irrelevant clarifying questions until max_turns was reached. Terminal reason: "max_turns".
→ {"score": 0.4, "reason": "Agent identified the order but failed to initiate the refund process before running out of turns."}

Example C (score: 0.0): User wanted to check order status. Agent tried to sell them a new product instead. Terminal reason: "goal_abandoned".
→ {"score": 0.0, "reason": "Agent completely misunderstood the goal, attempting a sale instead of providing order status."}

## INDEPENDENCE CONSTRAINT

Score based ONLY on whether the goal was achieved. Do NOT penalize or reward for:
- Efficiency of the path taken (scored separately as trajectory_efficiency)
- Tone or communication style (scored separately as persona_fit)
- Whether specific tools were called (scored separately as tool_correctness)

## BIAS WARNINGS

- Do NOT give higher scores to longer traces. A 2-turn success is as good as a 5-turn success.
- Do NOT be lenient — "attempted but failed" is NOT achievement.
- The terminal_reason field is strong signal but not definitive. Verify against actual trace content.

## OUTPUT FORMAT

Write your chain-of-thought analysis first (2-4 sentences), then provide your score as a JSON object on its own line:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence summary of your judgment>"}"""


async def judge_goal_achievement(trace: Trace, scenario: Scenario, llm: LLMPort) -> DimensionScore:
    """Judge whether the agent achieved the user's goal.

    Uses chain-of-thought evaluation with calibration anchors and
    explicit independence constraints.
    """
    user_prompt = f"""## SCENARIO CONTEXT
User's goal: {scenario.persona.goal}
Expected output must contain: {scenario.expectations.output_must_contain or "(none specified)"}
Expected terminal state: {scenario.expectations.terminal_state or "(none specified)"}

## CONVERSATION TRACE
{_format_trace(trace)}

Evaluate goal achievement following the protocol above:"""

    return await _call_judge(llm, _GOAL_ACHIEVEMENT_SYSTEM, user_prompt, "goal_achievement")


# =============================================================================
# JUDGE: TRAJECTORY EFFICIENCY
# =============================================================================

_TRAJECTORY_EFFICIENCY_SYSTEM = """You are a rigorous evaluator specializing in assessing the efficiency of AI agent execution paths.

## YOUR TASK

Determine whether an AI agent took an efficient path toward completing the user's goal. You are evaluating the QUALITY OF THE STRATEGY, not whether the goal was achieved.

## EVALUATION PROTOCOL

You MUST follow this protocol exactly:
1. Identify the minimum steps an ideal agent would take for this task
2. Count actual steps taken and categorize each as: (a) necessary, (b) redundant, (c) counterproductive
3. Check for common inefficiency patterns: loops, backtracking, repeated failures, unnecessary clarification
4. Compare actual turn count to the theoretical minimum
5. Write your reasoning (2-4 sentences)
6. THEN assign a score

## SCORING RUBRIC (Behavioral Anchors)

- **1.0**: Optimal path. Every action was necessary. No redundant steps. Turn count equals or is within 1 of theoretical minimum. No retries or backtracking.
- **0.8**: Near-optimal. 1-2 steps that weren't strictly necessary but were reasonable (e.g., a confirmation question). Turn ratio ≤ 1.5x optimal.
- **0.6**: Acceptable but wasteful. Some unnecessary steps, minor backtracking, or one retry. Turn ratio ~2x optimal.
- **0.4**: Inefficient. Multiple unnecessary steps, repeated attempts at the same action, or significant detours. Turn ratio ~3x optimal.
- **0.2**: Very inefficient. Extensive wasted effort, multiple retries, confusion about approach, or near-looping behavior. Turn ratio >3x optimal.
- **0.0**: No coherent strategy. Actions appear random, agent is stuck in loops, or every step is counterproductive.

## CALIBRATION EXAMPLES

Example A (score: 0.9): For a "buy laptop" task requiring search → add_to_cart → checkout (3 tool steps), agent used 3 turns with exactly those tools plus one clarifying question about preference.
→ {"score": 0.9, "reason": "Near-optimal: 3 required tools called in sequence with one reasonable clarification. Turn ratio 1.3x optimal."}

Example B (score: 0.5): Agent searched inventory 3 times with slightly different queries before finding the product, then completed the purchase normally.
→ {"score": 0.5, "reason": "Redundant search attempts (3x instead of 1x needed). Core path correct but wasted effort on discovery."}

Example C (score: 0.2): Agent called search_inventory, then checked order status (wrong tool), then searched again, then tried to add wrong item, then searched a third time.
→ {"score": 0.2, "reason": "Multiple wrong tool selections and retries. Only 2 of 7 actions were productive toward the goal."}

## INDEPENDENCE CONSTRAINT

Score based ONLY on path efficiency. Do NOT penalize or reward for:
- Whether the goal was ultimately achieved (scored separately as goal_achievement)
- Whether tool arguments were correct (scored separately as argument_correctness)
- Communication style or tone (scored separately as persona_fit)

## BIAS WARNINGS

- Do NOT conflate "more turns" with "less efficient" when extra turns serve a purpose (e.g., user asked a clarifying question)
- DO distinguish between agent-caused inefficiency vs. user-caused turns
- A failed attempt that was a reasonable strategy is less penalizing than a clearly wrong action

## OUTPUT FORMAT

Write your chain-of-thought analysis first (2-4 sentences), then provide your score as a JSON object on its own line:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence summary of your judgment>"}"""


async def judge_trajectory_efficiency(
    trace: Trace, scenario: Scenario, llm: LLMPort
) -> DimensionScore:
    """Judge whether the agent took an efficient path to the goal.

    Evaluates strategy quality using turn-ratio analysis and
    step categorization (necessary vs. redundant vs. counterproductive).
    """
    user_prompt = f"""## SCENARIO CONTEXT
User's goal: {scenario.persona.goal}
Max turns allowed: {scenario.constraints.max_turns}
Required tools for this task: {scenario.expectations.required_tools}
Minimum theoretical steps: {len(scenario.expectations.required_tools)} tool calls

## CONVERSATION TRACE
{_format_trace(trace)}

Evaluate trajectory efficiency following the protocol above:"""

    return await _call_judge(llm, _TRAJECTORY_EFFICIENCY_SYSTEM, user_prompt, "trajectory_efficiency")


# =============================================================================
# JUDGE: PERSONA FIT
# =============================================================================

_PERSONA_FIT_SYSTEM = """You are a rigorous evaluator specializing in assessing communication appropriateness in AI agent interactions.

## YOUR TASK

Determine whether an AI agent's responses were appropriately calibrated to the user's persona — their tone, knowledge level, emotional state, and background. You are evaluating the agent's ADAPTIVE COMMUNICATION, not the correctness of its actions.

## EVALUATION PROTOCOL

You MUST follow this protocol exactly:
1. Identify the persona characteristics: tone, knowledge level, emotional state, background
2. For each agent response, assess:
   a. Tone match: Did the agent mirror appropriate empathy/directness/formality?
   b. Complexity calibration: Was language complexity appropriate for the knowledge level?
   c. Emotional acknowledgment: Did the agent recognize and respond to emotional cues?
   d. Context sensitivity: Did responses account for the user's background?
3. Identify any persona mismatches (too formal for casual user, too simple for expert, etc.)
4. Write your reasoning (2-4 sentences)
5. THEN assign a score

## SCORING RUBRIC (Behavioral Anchors)

- **1.0**: Perfect calibration. Agent consistently matched tone (empathetic for frustrated, efficient for experts), used appropriate vocabulary complexity, acknowledged emotions, and adapted to background context. No mismatches.
- **0.8**: Good calibration. Agent mostly matched persona with 1 minor mismatch (e.g., slightly too formal once, or missed one emotional cue). Overall appropriate.
- **0.6**: Acceptable but generic. Agent used a neutral/default style that wasn't offensive but also wasn't specifically adapted. No major mismatches but no evidence of persona-awareness either.
- **0.4**: Poor calibration. Agent showed multiple mismatches — e.g., using jargon with a novice, being chatty with an impatient user, or ignoring frustration signals.
- **0.2**: Significant mismatch. Agent's style was actively inappropriate — condescending to an expert, dismissive of a frustrated user, or overly casual in a formal context.
- **0.0**: Complete mismatch. Agent's communication style was counterproductive — escalating an angry user, confusing a novice with technical language, or being inappropriately formal/informal throughout.

## CALIBRATION EXAMPLES

Example A (score: 0.9): Persona is "frustrated, intermediate, repeat customer". Agent said: "I completely understand your frustration — let me look into this right away for you. As a valued repeat customer, I want to make this right." Then proceeded efficiently.
→ {"score": 0.9, "reason": "Excellent persona fit: acknowledged frustration, referenced loyalty, used efficient language matching intermediate level."}

Example B (score: 0.5): Persona is "angry, novice". Agent said: "I'll check the order status using our OMS. The fulfillment pipeline shows your item is in transit with an ETA of 3-5 business days per our SLA." No emotional acknowledgment.
→ {"score": 0.5, "reason": "Used technical jargon (OMS, fulfillment pipeline, SLA) inappropriate for novice. Did not acknowledge anger or emotional state."}

Example C (score: 0.2): Persona is "direct, expert, IT professional". Agent said: "Let me explain how our website works! First, you go to the search bar, and then you type what you're looking for..." with excessive hand-holding.
→ {"score": 0.2, "reason": "Condescending tone for an expert user. Unnecessary over-explanation directly contradicts the persona's directness and expertise."}

## INDEPENDENCE CONSTRAINT

Score based ONLY on communication appropriateness. Do NOT penalize or reward for:
- Whether the goal was achieved (scored separately as goal_achievement)
- Whether the path was efficient (scored separately as trajectory_efficiency)
- Whether correct tools were used (scored separately as tool_correctness)
- Quality of tool outputs or factual accuracy

## BIAS WARNINGS

- Do NOT equate politeness with good persona fit. An expert who wants directness is poorly served by excessive pleasantries.
- Do NOT penalize short responses if the persona values efficiency.
- Do NOT reward long, empathetic responses if the persona is impatient.
- Evaluate EACH agent response, not just the first one. Consistency matters.

## OUTPUT FORMAT

Write your chain-of-thought analysis first (2-4 sentences), then provide your score as a JSON object on its own line:
{"score": <float 0.0-1.0>, "reason": "<1-2 sentence summary of your judgment>"}"""


async def judge_persona_fit(trace: Trace, scenario: Scenario, llm: LLMPort) -> DimensionScore:
    """Judge whether the agent's responses were appropriate for the user persona.

    Evaluates adaptive communication across four axes: tone match,
    complexity calibration, emotional acknowledgment, and context sensitivity.
    """
    user_prompt = f"""## PERSONA PROFILE
Goal: {scenario.persona.goal}
Tone: {scenario.persona.tone}
Knowledge level: {scenario.persona.knowledge_level}
Background: {scenario.persona.background}
Goal reveal strategy: {scenario.persona.goal_reveal_strategy}

## CONVERSATION TRACE (focus on agent's responses and how they adapt to the persona)
{_format_trace(trace)}

Evaluate persona fit following the protocol above:"""

    return await _call_judge(llm, _PERSONA_FIT_SYSTEM, user_prompt, "persona_fit")
