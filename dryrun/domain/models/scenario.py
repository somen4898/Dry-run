"""Scenario domain models — Persona, Expectation, Constraints, Scenario."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Persona(BaseModel):
    goal: str
    tone: str
    knowledge_level: str
    background: str
    goal_reveal_strategy: str = "incremental"


class Expectation(BaseModel):
    required_tools: list[str]
    required_tool_args: dict[str, Any]
    terminal_state: str | None = None
    output_must_contain: list[str]


class Constraints(BaseModel):
    max_turns: int = 10
    timeout_seconds: int = 120
    max_tokens: int = 8000


class Scenario(BaseModel):
    id: str
    name: str
    description: str
    persona: Persona
    opening_input: str
    expectations: Expectation
    constraints: Constraints
    golden: bool = False
    tags: list[str] = []
