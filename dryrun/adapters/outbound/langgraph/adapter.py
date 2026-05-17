"""LangGraphAdapter — implements AgentPort for LangGraph compiled graphs."""

from __future__ import annotations
import asyncio
import logging
import time
import uuid
from typing import Any
from langchain_core.messages import HumanMessage
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.models.trace import AgentTurn, ToolCall

logger = logging.getLogger(__name__)


class LangGraphAdapter(AgentPort):
    def __init__(self, graph: Any):
        """graph: a compiled LangGraph StateGraph."""
        self._graph = graph

    def new_session(self) -> str:
        return str(uuid.uuid4())

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        config = {"configurable": {"thread_id": session_id}}

        # Capture state before
        try:
            state_before = dict(self._graph.get_state(config).values)
        except Exception:
            logger.debug("Could not capture state_before", exc_info=True)
            state_before = {}

        turn_number = len(state_before.get("messages", [])) // 2 + 1

        # Run the graph
        start = time.perf_counter()
        input_msg = {"messages": [HumanMessage(content=user_input)]}
        result = self._graph.invoke(input_msg, config)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Capture state after
        try:
            state_after = dict(self._graph.get_state(config).values)
        except Exception:
            logger.debug("Could not capture state_after", exc_info=True)
            state_after = {}

        # Extract output
        messages = result.get("messages", [])
        output_text, visible_output_text, tool_calls = self._extract_output(messages)

        # Estimate tokens
        tokens_used = self._estimate_tokens(messages)

        return AgentTurn(
            turn_number=turn_number,
            agent_id=state_after.get("current_agent", "unknown"),
            input_text=user_input,
            output_text=output_text,
            tool_calls=tool_calls,
            state_before=self._serialize_state(state_before),
            state_after=self._serialize_state(state_after),
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            visible_output_text=visible_output_text,
        )

    def get_state(self, session_id: str) -> dict:
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self._graph.get_state(config)
            return self._serialize_state(dict(state.values))
        except Exception:
            logger.debug("Could not capture state for session %s", session_id, exc_info=True)
            return {}

    def _extract_output(self, messages: list) -> tuple[str, str, list[ToolCall]]:
        """Extract output text and tool calls from message history."""
        full_parts: list[str] = []
        visible_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.get("name", "unknown"),
                            arguments=tc.get("args", {}),
                            output=None,
                            latency_ms=0,
                        )
                    )

        # The last non-tool AI message is the visible output
        # Skip routing instructions (e.g., "ROUTE:sales") which are internal agent signals
        for msg in reversed(messages):
            content = self._get_text_content(msg)
            if content:
                if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
                    # Filter out routing instructions from visible output
                    lines = content.split("\n")
                    visible_lines = [l for l in lines if not l.strip().startswith("ROUTE:")]
                    filtered = "\n".join(visible_lines).strip()
                    if filtered:
                        visible_parts.insert(0, filtered)
                        break

        # Full output = all AI message contents
        for msg in messages:
            content = self._get_text_content(msg)
            if content and getattr(msg, "type", None) == "ai":
                full_parts.append(content)

        output_text = "\n".join(full_parts) or ""
        visible_output_text = "\n".join(visible_parts) or output_text

        return output_text, visible_output_text, tool_calls

    @staticmethod
    def _get_text_content(msg) -> str:
        """Extract text from a message, handling both str and list content formats."""
        content = getattr(msg, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Anthropic returns content as a list of blocks: [{"type": "text", "text": "..."}]
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimate based on message content length."""
        total_chars = sum(len(self._get_text_content(msg)) for msg in messages)
        return total_chars // 4

    def _serialize_state(self, state: dict) -> dict:
        """Convert state to JSON-serializable dict."""
        result = {}
        for k, v in state.items():
            if k == "messages":
                result[k] = [
                    {"type": getattr(m, "type", "unknown"), "content": getattr(m, "content", "")}
                    for m in v
                ]
            else:
                result[k] = v
        return result
