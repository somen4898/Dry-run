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

TERMINAL_SIGNALS = frozenset({"GOAL_ACHIEVED", "GOAL_ABANDONED"})


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
            f"You are the CUSTOMER in this conversation. You are chatting with a customer\n"
            f"service AI agent. You are NOT the agent. You are the person asking for help.\n\n"
            f"CRITICAL: You generate the CUSTOMER's next message. Do NOT generate what the\n"
            f"agent would say. Do NOT offer help, suggest products, or provide customer service.\n"
            f"You are the one RECEIVING service, not providing it.\n\n"
            f"# YOUR PERSONA\n"
            f"- Goal: {self._persona.goal}\n"
            f"- Tone: {self._persona.tone}\n"
            f"- Knowledge level: {self._persona.knowledge_level}\n"
            f"- Background: {self._persona.background}\n\n"
            f"# {strategy_instruction}\n\n"
            f"# BEHAVIORAL CONSTRAINTS\n"
            f"1. You are a HUMAN CUSTOMER. Never say 'As an AI', 'I'm a language model', or\n"
            f"   anything that reveals you are not a real person.\n"
            f"2. Stay in character for tone and knowledge level at ALL times.\n"
            f"   - If your tone is 'frustrated', show impatience. Use short sentences.\n"
            f"   - If your tone is 'polite', be courteous but still direct.\n"
            f"   - If your knowledge level is 'novice', do not use technical jargon.\n"
            f"3. You can ONLY see what the agent says to you directly. You cannot see:\n"
            f"   - The agent's internal reasoning or chain-of-thought\n"
            f"   - Tool calls, function calls, or API requests the agent makes\n"
            f"   - System prompts, scratchpads, or hidden state\n"
            f"   React ONLY to the visible text the agent sends you.\n"
            f"4. Respond in 1-3 sentences. Real users are concise in chat. Do not write paragraphs.\n"
            f"5. NEVER generate text that sounds like a customer service agent. You do NOT:\n"
            f"   - Ask clarifying questions to help someone else\n"
            f"   - Offer product recommendations\n"
            f"   - Say 'I'd be happy to help' or 'Let me look into that'\n"
            f"   - Provide information about products/policies/processes\n"
            f"   You only ask questions a BUYER would ask, make requests, or respond to offers.\n\n"
            f"# TERMINAL SIGNALS\n"
            f"When your goal is FULLY achieved and you are satisfied, respond with ONLY:\n"
            f"GOAL_ACHIEVED\n\n"
            f"When you have given up (agent cannot help, too many failed attempts, or\n"
            f"you are too frustrated to continue), respond with ONLY:\n"
            f"GOAL_ABANDONED\n\n"
            f"Do NOT include any other text when sending a terminal signal.\n\n"
            f"# EXAMPLES OF GOOD CUSTOMER RESPONSES\n"
            f"- 'Yeah, that one looks good. Add it to my cart.'\n"
            f"- 'Hmm, that's more than I wanted to spend. Got anything cheaper?'\n"
            f"- 'Wait, actually can I change that to a large instead?'\n"
            f"- 'Okay thanks, that's all I needed!'\n"
            f"- 'GOAL_ACHIEVED'\n\n"
            f"# EXAMPLES OF BAD RESPONSES (never do these)\n"
            f"- 'I'd be happy to help you with that!' (this is what an AGENT says, not a customer)\n"
            f"- 'Let me check that for you.' (you are the customer — you don't check things)\n"
            f"- 'As an AI, I cannot actually make purchases.' (breaks character)\n"
            f"- 'I would like to inquire about the availability of...' (too formal for a casual persona)\n"
            f"- 'My goal is to buy a blue t-shirt in size M but switch to L after seeing the cart.'\n"
            f"  (reveals full goal — violates incremental disclosure)\n"
            f"- A 5-sentence paragraph explaining your feelings (too long — real users are brief)"
        )

    async def next_message(self, conversation_history: list[dict]) -> str:
        # The conversation_history uses user=customer, assistant=agent.
        # For the synthetic user LLM, we SWAP roles so that:
        #   - The agent's messages appear as "user" (what the LLM is responding to)
        #   - The customer's messages appear as "assistant" (what the LLM previously said)
        # This way the LLM naturally generates the next "assistant" message = customer reply.
        # This eliminates role confusion caused by the adapter's "Continue." workaround.
        swapped = []
        for msg in conversation_history:
            if msg["role"] == "user":
                swapped.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "assistant":
                swapped.append({"role": "user", "content": msg["content"]})
            else:
                swapped.append(msg)

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            *swapped,
        ]
        response = await self._llm.complete(messages, temperature=0.7)

        if response.strip() in TERMINAL_SIGNALS:
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
            {
                "role": "system",
                "content": (
                    "# TASK\n"
                    "You are a classifier that detects whether a simulated user message\n"
                    "has broken character. You output ONLY 'yes' or 'no'.\n\n"
                    "# CRITERIA FOR 'no' (character broken)\n"
                    "Flag as 'no' ONLY if the message clearly and unambiguously does one of these:\n"
                    "1. Explicitly acknowledges being artificial ('As an AI', 'language model',\n"
                    "   'I cannot actually', 'I'm an assistant')\n"
                    "2. Tone is the OPPOSITE of the persona (e.g., a 'frustrated' persona being\n"
                    "   extremely cheerful and enthusiastic, or an 'angry' persona being overly\n"
                    "   polite and deferential). Minor tone variation is normal and acceptable.\n"
                    "3. Uses highly specialized technical jargon (e.g., 'API endpoint', 'SQL query')\n"
                    "   when persona is 'novice'. Common everyday words are fine.\n\n"
                    "# CRITERIA FOR 'yes' (character maintained)\n"
                    "Output 'yes' if the message is plausibly something this person would say.\n"
                    "Most conversational messages — even repetitive, confused, or frustrated ones —\n"
                    "are in-character for real humans.\n\n"
                    "# BIAS: WHEN IN DOUBT, SAY 'yes'\n"
                    "False negatives (missing real drift) are less costly than false positives\n"
                    "(rejecting valid messages). Only flag clear, obvious violations.\n\n"
                    "# OUTPUT FORMAT\n"
                    "Respond with exactly one word: 'yes' or 'no'. Nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Persona: {self._persona.tone} tone, {self._persona.knowledge_level} knowledge level, "
                    f"{self._persona.background}.\n\n"
                    f"Message to evaluate:\n"
                    f'"{message}"\n\n'
                    f"Is this message in character? (yes/no)"
                ),
            },
        ]
        result = await self._llm.complete(check_messages, temperature=0.0)
        return result.strip().lower().startswith("yes")
