"""Trace domain models — ToolCall, AgentTurn, Trace."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    output: Any
    latency_ms: int


class AgentTurn(BaseModel):
    turn_number: int
    agent_id: str
    input_text: str
    output_text: str
    tool_calls: list[ToolCall]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    latency_ms: int
    tokens_used: int
    visible_output_text: str


class Trace(BaseModel):
    scenario_id: str
    turns: list[AgentTurn]
    final_state: dict[str, Any]
    total_turns: int
    total_tokens: int
    total_latency_ms: int
    terminal_reason: str  # "goal_met" | "max_turns" | "timeout" | "error"
