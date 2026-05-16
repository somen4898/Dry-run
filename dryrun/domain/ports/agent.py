"""AgentPort — the contract any agent framework adapter must implement."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dryrun.domain.models.trace import AgentTurn


class AgentPort(ABC):
    @abstractmethod
    def new_session(self) -> str:
        """Create a fresh, isolated session. State must not leak across sessions."""
        ...

    @abstractmethod
    def step(self, session_id: str, user_input: str) -> AgentTurn:
        """Run one agent turn. Must populate visible_output_text correctly."""
        ...

    @abstractmethod
    def get_state(self, session_id: str) -> dict:
        """Return a snapshot of the agent's current state for this session."""
        ...
