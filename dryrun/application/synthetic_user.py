"""SyntheticUser — LLM-driven persona role-play with goal-hiding and drift check."""
from __future__ import annotations
import logging
from dryrun.domain.models.scenario import Persona
from dryrun.domain.ports.llm import LLMPort

logger = logging.getLogger(__name__)

_GOAL_STRATEGY_INSTRUCTIONS = {
    "incremental": (
        "Reveal information about your goal gradually, the way a real human user would. "
        "Do NOT state your full goal in your first message. "
        "Volunteer details only when they become relevant to the conversation."
    ),
    "upfront": "State your full goal in your first message.",
    "evasive": (
        "Only reveal goal details when the agent explicitly asks a clarifying question. "
        "Test the agent's willingness to ask."
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
            f"You are role-playing a user with the following profile:\n"
            f"Goal: {self._persona.goal}\n"
            f"Tone: {self._persona.tone}\n"
            f"Knowledge level: {self._persona.knowledge_level}\n"
            f"Background: {self._persona.background}\n\n"
            f"Goal-reveal strategy: {self._persona.goal_reveal_strategy}\n"
            f"  {strategy_instruction}\n\n"
            f"You are having a conversation with an AI agent to accomplish your goal.\n"
            f"Respond naturally as this person would. Stay in character.\n"
            f"You can ONLY see what the agent says to you. You cannot see its internal "
            f"reasoning, tool calls, or scratchpad. Respond only to what is visible.\n\n"
            f"When your goal is achieved, say exactly: GOAL_ACHIEVED\n"
            f"When you give up, say exactly: GOAL_ABANDONED\n"
            f"Never break character. Never acknowledge you are an AI."
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
                {"role": "system", "content": (
                    "Your previous response broke character. "
                    "Stay in character as the persona described above. Try again."
                )},
            ]
            response = await self._llm.complete(messages_with_reinforcement, temperature=0.5)
            if not await self._check_persona_drift(response):
                logger.warning("Persona drift persisted after retry, accepting with warning")

        return response.strip()

    async def _check_persona_drift(self, message: str) -> bool:
        check_messages = [
            {"role": "system", "content": (
                "You are a persona-consistency checker. "
                "Does the following message stay in character as a human user? "
                "Answer 'yes' or 'no' only."
            )},
            {"role": "user", "content": (
                f"Persona: {self._persona.tone} {self._persona.knowledge_level} user.\n"
                f"Message: {message}\n"
                f"Does this message stay in character? (yes/no)"
            )},
        ]
        result = await self._llm.complete(check_messages, temperature=0.0)
        return result.strip().lower().startswith("yes")
