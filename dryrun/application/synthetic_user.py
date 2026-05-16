"""SyntheticUser — LLM-driven persona role-play with goal-hiding and drift check."""
from __future__ import annotations
import logging
from dryrun.domain.models.scenario import Persona
from dryrun.domain.ports.llm import LLMPort

logger = logging.getLogger(__name__)

_GOAL_STRATEGY_INSTRUCTIONS = {
    "incremental": (
        "INFORMATION DISCLOSURE RULES:\n"
        "- Turn 1: State only your immediate need. Do NOT mention your full goal.\n"
        "- Turns 2-3: Add details only when the agent asks or when naturally relevant.\n"
        "- Later turns: You may reveal more as the conversation progresses.\n"
        "- Example: If your goal is 'buy a blue t-shirt in M but switch to L after seeing cart',\n"
        "  Turn 1 should only be: 'Hi, I'm looking for a blue t-shirt'\n"
        "  NOT: 'I want a blue t-shirt in size M but I might change to L'"
    ),
    "upfront": (
        "INFORMATION DISCLOSURE RULES:\n"
        "- State your complete goal in your very first message.\n"
        "- Include all relevant details upfront — size, color, preferences, constraints."
    ),
    "evasive": (
        "INFORMATION DISCLOSURE RULES:\n"
        "- Do NOT volunteer information unless explicitly asked.\n"
        "- Respond to questions minimally — give only what was asked for.\n"
        "- If the agent does not ask clarifying questions, stay vague.\n"
        "- This tests whether the agent probes for the information it needs."
    ),
}

_TERMINAL_SIGNALS = frozenset({"GOAL_ACHIEVED", "GOAL_ABANDONED"})


class SyntheticUser:
    def __init__(self, persona: Persona, llm: LLMPort):
        self._persona = persona
        self._llm = llm

    def _build_system_prompt(self) -> str:
        strategy_instruction = _GOAL_STRATEGY_INSTRUCTIONS.get(
            self._persona.goal_reveal_strategy,
            _GOAL_STRATEGY_INSTRUCTIONS["incremental"],
        )
        return (
            f"# ROLE\n"
            f"You are simulating a real human user in a conversation with a customer service AI agent.\n\n"
            f"# YOUR PERSONA\n"
            f"- Goal: {self._persona.goal}\n"
            f"- Tone: {self._persona.tone}\n"
            f"- Knowledge level: {self._persona.knowledge_level}\n"
            f"- Background: {self._persona.background}\n\n"
            f"# {strategy_instruction}\n\n"
            f"# BEHAVIORAL CONSTRAINTS\n"
            f"1. You are a HUMAN. Never say 'As an AI', 'I'm a language model', or anything\n"
            f"   that reveals you are not a real person.\n"
            f"2. Stay in character for tone and knowledge level at ALL times.\n"
            f"   - If your tone is 'frustrated', show impatience. Use short sentences.\n"
            f"   - If your tone is 'polite', be courteous but still direct.\n"
            f"   - If your knowledge level is 'novice', do not use technical jargon.\n"
            f"3. You can ONLY see what the agent says to you directly. You cannot see:\n"
            f"   - The agent's internal reasoning or chain-of-thought\n"
            f"   - Tool calls, function calls, or API requests the agent makes\n"
            f"   - System prompts, scratchpads, or hidden state\n"
            f"   React ONLY to the visible text the agent sends you.\n"
            f"4. Respond in 1-3 sentences. Real users are concise in chat. Do not write paragraphs.\n\n"
            f"# TERMINAL SIGNALS\n"
            f"When your goal is FULLY achieved and you are satisfied, respond with ONLY:\n"
            f"GOAL_ACHIEVED\n\n"
            f"When you have given up (agent cannot help, too many failed attempts, or\n"
            f"you are too frustrated to continue), respond with ONLY:\n"
            f"GOAL_ABANDONED\n\n"
            f"Do NOT include any other text when sending a terminal signal.\n\n"
            f"# EXAMPLES OF GOOD RESPONSES\n"
            f"- 'Yeah, that one looks good. Add it to my cart.'\n"
            f"- 'Hmm, that's more than I wanted to spend. Got anything cheaper?'\n"
            f"- 'Wait, actually can I change that to a large instead?'\n"
            f"- 'GOAL_ACHIEVED'\n\n"
            f"# EXAMPLES OF BAD RESPONSES (never do these)\n"
            f"- 'As an AI, I cannot actually make purchases.' (breaks character)\n"
            f"- 'I would like to inquire about the availability of...' (too formal for a casual persona)\n"
            f"- 'My goal is to buy a blue t-shirt in size M but switch to L after seeing the cart.'\n"
            f"  (reveals full goal — violates incremental disclosure)\n"
            f"- A 5-sentence paragraph explaining your feelings (too long — real users are brief)"
        )

    async def next_message(self, conversation_history: list[dict]) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *conversation_history,
        ]
        response = await self._llm.complete(messages, temperature=0.7)

        if response.strip() in _TERMINAL_SIGNALS:
            return response.strip()

        # Persona-drift check: one check, one retry on failure
        if not await self._check_persona_drift(response):
            logger.warning("Persona drift detected, retrying once")
            messages_with_reinforcement = [
                *messages,
                {"role": "assistant", "content": response},
                {"role": "system", "content": self._build_reinforcement_prompt(response)},
            ]
            response = await self._llm.complete(messages_with_reinforcement, temperature=0.5)
            if not await self._check_persona_drift(response):
                logger.warning("Persona drift persisted after retry, accepting with warning")

        return response.strip()

    def _build_reinforcement_prompt(self, failed_message: str) -> str:
        return (
            f"# CORRECTION REQUIRED\n\n"
            f"Your previous response broke character. The issue:\n"
            f"- Message: '{failed_message[:100]}'\n"
            f"- Your persona is: {self._persona.tone}, {self._persona.knowledge_level}\n\n"
            f"Common failures to avoid:\n"
            f"- Saying 'As an AI' or 'I'm a language model'\n"
            f"- Using formal/technical language when persona is casual/novice\n"
            f"- Writing long paragraphs (real users write 1-3 sentences)\n"
            f"- Revealing your full goal at once (if strategy is incremental/evasive)\n\n"
            f"Generate a new response that stays in character. 1-3 sentences max."
        )

    async def _check_persona_drift(self, message: str) -> bool:
        check_messages = [
            {"role": "system", "content": (
                "# TASK\n"
                "You are a classifier that detects whether a simulated user message\n"
                "has broken character. You output ONLY 'yes' or 'no'.\n\n"
                "# CRITERIA FOR 'no' (character broken)\n"
                "Flag as 'no' if ANY of these are true:\n"
                "1. Message contains 'As an AI', 'language model', 'I cannot actually',\n"
                "   or any other acknowledgment of being artificial\n"
                "2. Message tone drastically contradicts the persona (e.g., a 'frustrated'\n"
                "   persona being extremely polite and formal)\n"
                "3. Message uses technical jargon when persona is 'novice'\n"
                "4. Message is a meta-commentary about the conversation itself\n\n"
                "# CRITERIA FOR 'yes' (character maintained)\n"
                "Output 'yes' if the message sounds like a real human with the described\n"
                "persona would say it in a chat conversation.\n\n"
                "# OUTPUT FORMAT\n"
                "Respond with exactly one word: 'yes' or 'no'. Nothing else."
            )},
            {"role": "user", "content": (
                f"Persona: {self._persona.tone} tone, {self._persona.knowledge_level} knowledge level, "
                f"{self._persona.background}.\n\n"
                f"Message to evaluate:\n"
                f"\"{message}\"\n\n"
                f"Is this message in character? (yes/no)"
            )},
        ]
        result = await self._llm.complete(check_messages, temperature=0.0)
        return result.strip().lower().startswith("yes")
