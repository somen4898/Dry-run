# Phase 1: Core Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dryrun run one_scenario.yaml` executes a full multi-turn conversation between the synthetic user and the sample LangGraph agent and prints the captured trace.

**Architecture:** Hexagonal (ports-and-adapters). Domain layer (models + ports + services) has zero external deps. Application layer orchestrates use cases. Adapters layer wires infrastructure. Dependencies point inward.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph, OpenAI API, Click, Rich, PyYAML, pytest

---

## Task Ordering Rationale

1. Scaffold first (pyproject.toml) so `pytest` and imports work from task 2 onward
2. Domain models next — zero dependencies, pure Pydantic, everything else depends on them
3. Ports next — ABCs that define the contracts for adapters
4. Contract tests with a stub adapter — proves the ABC is honest before any real adapter exists
5. Scenario YAML loading — the input format, needed by everything downstream
6. Synthetic user — depends on LLMPort and domain models
7. Runner — depends on AgentPort, SyntheticUser, Trace models
8. Sample LangGraph agent — the system under test, a fixture
9. LangGraphAdapter — real implementation of AgentPort
10. OpenAI LLM client — real implementation of LLMPort
11. Config loading — the `dryrun.yaml` config file
12. CLI wiring — connects everything
13. End-to-end smoke test — the exit criterion

---

### Task 1: Project scaffold and pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: all `__init__.py` files for package structure
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[project]
name = "dryrun-agents"
version = "0.1.0"
description = "Simulation-based testing harness for multi-agent LangGraph systems"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0,<3.0",
    "click>=8.0,<9.0",
    "rich>=13.0",
    "pyyaml>=6.0",
    "openai>=1.0",
    "langgraph>=0.2",
    "langchain-openai>=0.1",
    "langchain-core>=0.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.2",
    "mypy>=1.8",
    "ruff>=0.3",
]

[project.scripts]
dryrun = "dryrun.adapters.inbound.cli.commands:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
```

- [ ] **Step 2: Create directory structure with all `__init__.py` files**

```
dryrun/__init__.py
dryrun/domain/__init__.py
dryrun/domain/models/__init__.py
dryrun/domain/ports/__init__.py
dryrun/application/__init__.py
dryrun/adapters/__init__.py
dryrun/adapters/inbound/__init__.py
dryrun/adapters/inbound/cli/__init__.py
dryrun/adapters/outbound/__init__.py
dryrun/adapters/outbound/langgraph/__init__.py
dryrun/adapters/outbound/openai/__init__.py
example/__init__.py
example/agent/__init__.py
example/scenarios/          (directory only, no __init__.py)
tests/__init__.py
tests/unit/__init__.py
tests/unit/domain/__init__.py
tests/unit/application/__init__.py
tests/unit/adapters/__init__.py
tests/contract/__init__.py
tests/integration/__init__.py
```

- [ ] **Step 3: Create tests/conftest.py**

```python
"""Dry Run test configuration."""
```

- [ ] **Step 4: Install in editable mode and verify pytest runs**

Run: `pip install -e ".[dev]" && pytest --co -q`
Expected: 0 tests collected, exit code 0

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml dryrun/ example/ tests/
git commit -m "feat: scaffold project structure with pyproject.toml and package layout"
```

---

### Task 2: Domain models — Scenario, Persona, Expectation, Constraints

**Files:**
- Create: `dryrun/domain/models/scenario.py`
- Test: `tests/unit/domain/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for scenario domain models."""
import pytest
from dryrun.domain.models.scenario import Persona, Expectation, Constraints, Scenario


class TestPersona:
    def test_create_with_defaults(self):
        p = Persona(
            goal="Buy a laptop",
            tone="polite",
            knowledge_level="novice",
            background="College student on a budget",
        )
        assert p.goal_reveal_strategy == "incremental"

    def test_goal_reveal_strategy_values(self):
        for strategy in ("incremental", "upfront", "evasive"):
            p = Persona(
                goal="x", tone="x", knowledge_level="x",
                background="x", goal_reveal_strategy=strategy,
            )
            assert p.goal_reveal_strategy == strategy


class TestConstraints:
    def test_defaults(self):
        c = Constraints()
        assert c.max_turns == 10
        assert c.timeout_seconds == 120
        assert c.max_tokens == 8000


class TestScenario:
    def test_full_scenario(self):
        s = Scenario(
            id="test-001",
            name="Happy path",
            description="User buys a laptop",
            persona=Persona(
                goal="Buy a laptop", tone="polite",
                knowledge_level="novice", background="Student",
            ),
            opening_input="Hi, I need a laptop",
            expectations=Expectation(
                required_tools=["search_inventory"],
                required_tool_args={"search_inventory": {"query": "laptop"}},
                terminal_state=None,
                output_must_contain=["laptop"],
            ),
            constraints=Constraints(max_turns=5),
        )
        assert s.golden is False
        assert s.tags == []
        assert s.constraints.max_turns == 5

    def test_scenario_golden_flag(self):
        s = Scenario(
            id="g-001", name="Golden", description="x",
            persona=Persona(goal="x", tone="x", knowledge_level="x", background="x"),
            opening_input="x",
            expectations=Expectation(
                required_tools=[], required_tool_args={},
                terminal_state=None, output_must_contain=[],
            ),
            constraints=Constraints(),
            golden=True, tags=["smoke"],
        )
        assert s.golden is True
        assert "smoke" in s.tags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_models.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

`dryrun/domain/models/scenario.py`:
```python
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
    terminal_state: str | None
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/domain/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/domain/models/scenario.py tests/unit/domain/test_models.py
git commit -m "feat: add scenario domain models (Persona, Expectation, Constraints, Scenario)"
```

---

### Task 3: Domain models — Trace, AgentTurn, ToolCall

**Files:**
- Create: `dryrun/domain/models/trace.py`
- Modify: `tests/unit/domain/test_models.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/domain/test_models.py`:

```python
from dryrun.domain.models.trace import ToolCall, AgentTurn, Trace


class TestToolCall:
    def test_create(self):
        tc = ToolCall(
            tool_name="search_inventory",
            arguments={"query": "laptop"},
            output={"results": [{"name": "ThinkPad"}]},
            latency_ms=150,
        )
        assert tc.tool_name == "search_inventory"


class TestAgentTurn:
    def test_visible_output_separate_from_output(self):
        turn = AgentTurn(
            turn_number=1,
            agent_id="support",
            input_text="Hi",
            output_text="[internal reasoning] Hello! How can I help?",
            tool_calls=[],
            state_before={},
            state_after={"greeted": True},
            latency_ms=500,
            tokens_used=100,
            visible_output_text="Hello! How can I help?",
        )
        assert turn.visible_output_text != turn.output_text
        assert "[internal" not in turn.visible_output_text


class TestTrace:
    def test_empty_trace(self):
        t = Trace(
            scenario_id="test-001",
            turns=[],
            final_state={},
            total_turns=0,
            total_tokens=0,
            total_latency_ms=0,
            terminal_reason="max_turns",
        )
        assert t.terminal_reason == "max_turns"

    def test_trace_with_turns(self):
        turn = AgentTurn(
            turn_number=1, agent_id="support",
            input_text="Hi", output_text="Hello!",
            tool_calls=[], state_before={}, state_after={},
            latency_ms=200, tokens_used=50,
            visible_output_text="Hello!",
        )
        t = Trace(
            scenario_id="test-001",
            turns=[turn],
            final_state={"done": True},
            total_turns=1, total_tokens=50,
            total_latency_ms=200,
            terminal_reason="goal_met",
        )
        assert len(t.turns) == 1
        assert t.turns[0].agent_id == "support"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_models.py -v`
Expected: FAIL with ImportError on trace module

- [ ] **Step 3: Write implementation**

`dryrun/domain/models/trace.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/domain/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/domain/models/trace.py tests/unit/domain/test_models.py
git commit -m "feat: add trace domain models (ToolCall, AgentTurn, Trace)"
```

---

### Task 4: Port interfaces — AgentPort and LLMPort ABCs

**Files:**
- Create: `dryrun/domain/ports/agent.py`
- Create: `dryrun/domain/ports/llm.py`
- Test: `tests/unit/domain/test_ports.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests that port ABCs cannot be instantiated and enforce the contract."""
import pytest
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort


class TestAgentPortABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            AgentPort()  # type: ignore

    def test_has_required_methods(self):
        assert hasattr(AgentPort, "new_session")
        assert hasattr(AgentPort, "step")
        assert hasattr(AgentPort, "get_state")


class TestLLMPortABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMPort()  # type: ignore

    def test_has_complete_method(self):
        assert hasattr(LLMPort, "complete")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_ports.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

`dryrun/domain/ports/agent.py`:
```python
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
```

`dryrun/domain/ports/llm.py`:
```python
"""LLMPort — the contract for LLM completion providers."""
from __future__ import annotations
from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """Send messages to an LLM and return the completion text."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/domain/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/domain/ports/agent.py dryrun/domain/ports/llm.py tests/unit/domain/test_ports.py
git commit -m "feat: define AgentPort and LLMPort ABCs in domain ports"
```

---

### Task 5: Contract tests for AgentPort + stub adapter

**Files:**
- Create: `tests/contract/test_agent_port_contract.py`

- [ ] **Step 1: Write the contract test suite with stub**

```python
"""Contract tests for AgentPort — any implementation must pass these."""
import pytest
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.models.trace import AgentTurn, ToolCall


class StubAgentAdapter(AgentPort):
    """Minimal in-memory stub that proves the ABC works without LangGraph."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._counter = 0

    def new_session(self) -> str:
        self._counter += 1
        sid = f"stub-{self._counter}"
        self._sessions[sid] = []
        return sid

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        history = self._sessions[session_id]
        turn_number = len(history) + 1
        history.append({"input": user_input})
        return AgentTurn(
            turn_number=turn_number,
            agent_id="stub-agent",
            input_text=user_input,
            output_text=f"Stub response to: {user_input}",
            tool_calls=[],
            state_before={"turns": turn_number - 1},
            state_after={"turns": turn_number},
            latency_ms=1,
            tokens_used=10,
            visible_output_text=f"Stub response to: {user_input}",
        )

    def get_state(self, session_id: str) -> dict:
        return {"turns": len(self._sessions.get(session_id, []))}


class TestAgentPortContract:
    """Any AgentPort implementation must pass these tests."""

    @pytest.fixture
    def adapter(self) -> AgentPort:
        return StubAgentAdapter()

    def test_new_session_returns_string(self, adapter: AgentPort):
        sid = adapter.new_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_sessions_are_isolated(self, adapter: AgentPort):
        s1 = adapter.new_session()
        s2 = adapter.new_session()
        assert s1 != s2
        adapter.step(s1, "hello")
        state1 = adapter.get_state(s1)
        state2 = adapter.get_state(s2)
        assert state1 != state2

    def test_step_returns_agent_turn(self, adapter: AgentPort):
        sid = adapter.new_session()
        turn = adapter.step(sid, "test input")
        assert isinstance(turn, AgentTurn)
        assert turn.turn_number == 1
        assert turn.input_text == "test input"
        assert len(turn.visible_output_text) > 0

    def test_step_increments_turn_number(self, adapter: AgentPort):
        sid = adapter.new_session()
        t1 = adapter.step(sid, "first")
        t2 = adapter.step(sid, "second")
        assert t1.turn_number == 1
        assert t2.turn_number == 2

    def test_visible_output_text_is_populated(self, adapter: AgentPort):
        sid = adapter.new_session()
        turn = adapter.step(sid, "hi")
        assert turn.visible_output_text is not None
        assert isinstance(turn.visible_output_text, str)

    def test_get_state_returns_dict(self, adapter: AgentPort):
        sid = adapter.new_session()
        state = adapter.get_state(sid)
        assert isinstance(state, dict)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/contract/ -v`
Expected: All PASS (stub implements the ABC correctly)

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_agent_port_contract.py
git commit -m "test: add AgentPort contract tests with stub adapter"
```

---

### Task 6: Scenario YAML loading + example scenarios

**Files:**
- Create: `example/scenarios/happy_path.yaml`
- Create: `example/scenarios/mid_task_change.yaml`
- Test: `tests/unit/domain/test_scenario_loading.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for YAML scenario loading."""
import pytest
from pathlib import Path
import yaml
from dryrun.domain.models.scenario import Scenario


HAPPY_PATH_YAML = """\
id: happy-path-001
name: "Happy path - laptop purchase"
description: "User successfully purchases a laptop through the support agent"
persona:
  goal: "Buy a budget laptop for college"
  tone: "polite"
  knowledge_level: "novice"
  background: "College freshman, first time buying a laptop online"
  goal_reveal_strategy: "incremental"
opening_input: "Hi, I need help finding a laptop"
expectations:
  required_tools:
    - search_inventory
    - add_to_cart
    - process_checkout
  required_tool_args:
    search_inventory:
      query: "laptop"
  terminal_state: null
  output_must_contain:
    - "order"
constraints:
  max_turns: 8
  timeout_seconds: 120
  max_tokens: 8000
golden: true
tags:
  - smoke
  - purchase
"""


class TestScenarioFromYAML:
    def test_load_from_yaml_string(self):
        data = yaml.safe_load(HAPPY_PATH_YAML)
        scenario = Scenario(**data)
        assert scenario.id == "happy-path-001"
        assert scenario.persona.goal_reveal_strategy == "incremental"
        assert "search_inventory" in scenario.expectations.required_tools
        assert scenario.golden is True

    def test_load_from_yaml_file(self, tmp_path: Path):
        p = tmp_path / "test_scenario.yaml"
        p.write_text(HAPPY_PATH_YAML)
        data = yaml.safe_load(p.read_text())
        scenario = Scenario(**data)
        assert scenario.name == "Happy path - laptop purchase"

    def test_missing_required_field_raises(self):
        bad_yaml = "id: test\nname: test\n"
        data = yaml.safe_load(bad_yaml)
        with pytest.raises(Exception):
            Scenario(**data)

    def test_default_constraints(self):
        data = yaml.safe_load(HAPPY_PATH_YAML)
        data["constraints"] = {}
        scenario = Scenario(**data)
        assert scenario.constraints.max_turns == 10
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/unit/domain/test_scenario_loading.py -v`
Expected: All PASS (Pydantic + PyYAML do the work)

- [ ] **Step 3: Create example scenario YAML files**

`example/scenarios/happy_path.yaml`:
```yaml
id: happy-path-001
name: "Happy path - laptop purchase"
description: "User successfully purchases a laptop through the support agent"
persona:
  goal: "Buy a budget laptop for college"
  tone: "polite"
  knowledge_level: "novice"
  background: "College freshman, first time buying a laptop online"
  goal_reveal_strategy: "incremental"
opening_input: "Hi, I need help finding a laptop"
expectations:
  required_tools:
    - search_inventory
    - add_to_cart
    - process_checkout
  required_tool_args:
    search_inventory:
      query: "laptop"
  terminal_state: null
  output_must_contain:
    - "order"
constraints:
  max_turns: 8
  timeout_seconds: 120
  max_tokens: 8000
golden: true
tags:
  - smoke
  - purchase
```

`example/scenarios/mid_task_change.yaml`:
```yaml
id: mid-task-change-001
name: "Mid-task change - switch from laptop to tablet"
description: "User starts looking for a laptop, then changes mind to a tablet mid-conversation"
persona:
  goal: "Initially want a laptop, but switch to wanting a tablet after seeing prices"
  tone: "indecisive"
  knowledge_level: "novice"
  background: "Budget-conscious student who is easily swayed by price"
  goal_reveal_strategy: "incremental"
opening_input: "Hey, I'm looking for a laptop for school"
expectations:
  required_tools:
    - search_inventory
  required_tool_args:
    search_inventory:
      query: "laptop"
  terminal_state: null
  output_must_contain: []
constraints:
  max_turns: 10
  timeout_seconds: 120
  max_tokens: 8000
golden: false
tags:
  - edge-case
  - mid-task-change
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/domain/test_scenario_loading.py example/scenarios/
git commit -m "feat: add YAML scenario loading and example scenario files"
```

---

### Task 7: Synthetic user with goal-hiding and persona-drift check

**Files:**
- Create: `dryrun/application/synthetic_user.py`
- Test: `tests/unit/application/test_synthetic_user.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for SyntheticUser — uses a mock LLMPort."""
import pytest
import asyncio
from dryrun.domain.ports.llm import LLMPort
from dryrun.domain.models.scenario import Persona
from dryrun.application.synthetic_user import SyntheticUser


class MockLLMPort(LLMPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return next(self._responses)


class TestSyntheticUser:
    @pytest.fixture
    def persona(self) -> Persona:
        return Persona(
            goal="Buy a laptop",
            tone="polite",
            knowledge_level="novice",
            background="College student",
            goal_reveal_strategy="incremental",
        )

    def test_next_message_returns_string(self, persona):
        llm = MockLLMPort(["I'm looking for something affordable", "yes"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "How can I help?"}]
        result = asyncio.run(user.next_message(history))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_goal_achieved_signal(self, persona):
        llm = MockLLMPort(["GOAL_ACHIEVED"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "Your order is confirmed!"}]
        result = asyncio.run(user.next_message(history))
        assert result == "GOAL_ACHIEVED"

    def test_goal_abandoned_signal(self, persona):
        llm = MockLLMPort(["GOAL_ABANDONED"])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "Sorry, we can't help."}]
        result = asyncio.run(user.next_message(history))
        assert result == "GOAL_ABANDONED"

    def test_system_prompt_contains_goal_reveal_strategy(self, persona):
        llm = MockLLMPort(["response"])
        user = SyntheticUser(persona=persona, llm=llm)
        prompt = user._build_system_prompt()
        assert "incremental" in prompt.lower()
        assert persona.goal in prompt

    def test_system_prompt_evasive_strategy(self):
        persona = Persona(
            goal="Get a refund", tone="frustrated",
            knowledge_level="expert", background="Repeat customer",
            goal_reveal_strategy="evasive",
        )
        llm = MockLLMPort(["response"])
        user = SyntheticUser(persona=persona, llm=llm)
        prompt = user._build_system_prompt()
        assert "evasive" in prompt.lower()

    def test_persona_drift_check_passes_good_message(self, persona):
        llm = MockLLMPort(["yes"])
        user = SyntheticUser(persona=persona, llm=llm)
        result = asyncio.run(user._check_persona_drift("I'd like a budget laptop please"))
        assert result is True

    def test_persona_drift_check_fails_on_ai_reveal(self, persona):
        llm = MockLLMPort(["no"])
        user = SyntheticUser(persona=persona, llm=llm)
        result = asyncio.run(user._check_persona_drift(
            "As an AI language model, I cannot actually buy things"
        ))
        assert result is False

    def test_drift_retry_then_accept(self, persona):
        """On drift failure, regenerate once. On second failure, accept with warning."""
        llm = MockLLMPort([
            "As an AI, I can't buy things",  # first generation (bad)
            "no",                              # drift check fails
            "I'd like a laptop please",        # retry generation (good)
            "yes",                             # drift check passes
        ])
        user = SyntheticUser(persona=persona, llm=llm)
        history = [{"role": "assistant", "content": "How can I help?"}]
        result = asyncio.run(user.next_message(history))
        assert "AI" not in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_synthetic_user.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

`dryrun/application/synthetic_user.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/application/test_synthetic_user.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/application/synthetic_user.py tests/unit/application/test_synthetic_user.py
git commit -m "feat: implement SyntheticUser with goal-hiding and persona-drift check"
```

---

### Task 8: Runner (RunSuiteUseCase)

**Files:**
- Create: `dryrun/application/run_suite.py`
- Test: `tests/unit/application/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the runner — uses mock AgentPort and mock LLMPort."""
import pytest
import asyncio
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.domain.models.trace import AgentTurn, Trace
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.application.run_suite import RunSuiteUseCase


class MockAgentPort(AgentPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)
        self._sessions: dict[str, int] = {}
        self._counter = 0

    def new_session(self) -> str:
        self._counter += 1
        sid = f"mock-{self._counter}"
        self._sessions[sid] = 0
        return sid

    def step(self, session_id: str, user_input: str) -> AgentTurn:
        self._sessions[session_id] += 1
        turn_num = self._sessions[session_id]
        response = next(self._responses)
        return AgentTurn(
            turn_number=turn_num,
            agent_id="mock-agent",
            input_text=user_input,
            output_text=response,
            tool_calls=[],
            state_before={"turn": turn_num - 1},
            state_after={"turn": turn_num},
            latency_ms=100,
            tokens_used=50,
            visible_output_text=response,
        )

    def get_state(self, session_id: str) -> dict:
        return {"turn": self._sessions.get(session_id, 0)}


class MockLLMPort(LLMPort):
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    async def complete(self, messages, temperature=0.7, response_format=None) -> str:
        return next(self._responses)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="test-001",
        name="Test",
        description="Test scenario",
        persona=Persona(
            goal="Buy a laptop", tone="polite",
            knowledge_level="novice", background="Student",
        ),
        opening_input="Hi, I need a laptop",
        expectations=Expectation(
            required_tools=[], required_tool_args={},
            terminal_state=None, output_must_contain=[],
        ),
        constraints=Constraints(max_turns=5),
    )


class TestRunSuiteUseCase:
    def test_goal_met_terminates(self, scenario):
        agent = MockAgentPort(["Here's a laptop for you!", "Order confirmed!"])
        llm = MockLLMPort([
            "I'd like the ThinkPad",   # user turn 1
            "yes",                      # drift check
            "GOAL_ACHIEVED",            # user turn 2 (terminal)
        ])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert isinstance(trace, Trace)
        assert trace.terminal_reason == "goal_met"
        assert trace.total_turns == 2

    def test_max_turns_terminates(self, scenario):
        scenario.constraints.max_turns = 2
        agent = MockAgentPort(["response1", "response2"])
        llm = MockLLMPort([
            "user msg 1", "yes",  # turn 1 + drift check
            "user msg 2", "yes",  # turn 2 + drift check
        ])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.terminal_reason == "max_turns"

    def test_trace_captures_scenario_id(self, scenario):
        agent = MockAgentPort(["Hello!"])
        llm = MockLLMPort(["GOAL_ACHIEVED"])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.scenario_id == "test-001"

    def test_trace_accumulates_tokens(self, scenario):
        agent = MockAgentPort(["r1", "r2"])
        llm = MockLLMPort(["msg", "yes", "GOAL_ACHIEVED"])
        runner = RunSuiteUseCase(agent_port=agent, llm_port=llm)
        trace = asyncio.run(runner.run_scenario(scenario))
        assert trace.total_tokens == 100  # 50 per turn, 2 turns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_runner.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

`dryrun/application/run_suite.py`:
```python
"""RunSuiteUseCase — orchestrates scenario execution."""
from __future__ import annotations
import time
import logging
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.models.trace import Trace, AgentTurn
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.ports.llm import LLMPort
from dryrun.application.synthetic_user import SyntheticUser

logger = logging.getLogger(__name__)

_TERMINAL_SIGNALS = frozenset({"GOAL_ACHIEVED", "GOAL_ABANDONED"})


class RunSuiteUseCase:
    def __init__(self, agent_port: AgentPort, llm_port: LLMPort):
        self._agent = agent_port
        self._llm = llm_port

    async def run_scenario(self, scenario: Scenario) -> Trace:
        session_id = self._agent.new_session()
        user = SyntheticUser(persona=scenario.persona, llm=self._llm)

        turns: list[AgentTurn] = []
        history: list[dict] = []
        current_input = scenario.opening_input
        total_tokens = 0
        total_latency_ms = 0
        start_time = time.monotonic()

        for turn_idx in range(scenario.constraints.max_turns):
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed > scenario.constraints.timeout_seconds:
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, "timeout"
                )

            # Run agent step
            agent_turn = self._agent.step(session_id, current_input)
            turns.append(agent_turn)
            total_tokens += agent_turn.tokens_used
            total_latency_ms += agent_turn.latency_ms

            # IMPORTANT: pass ONLY visible_output_text to synthetic user
            history.append({"role": "assistant", "content": agent_turn.visible_output_text})

            # Check token budget
            if total_tokens > scenario.constraints.max_tokens:
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, "token_budget"
                )

            # Get synthetic user response
            next_input = await user.next_message(history)

            if next_input in _TERMINAL_SIGNALS:
                reason = "goal_met" if next_input == "GOAL_ACHIEVED" else "goal_abandoned"
                return self._build_trace(
                    scenario, turns, session_id, total_tokens, total_latency_ms, reason
                )

            history.append({"role": "user", "content": next_input})
            current_input = next_input

        return self._build_trace(
            scenario, turns, session_id, total_tokens, total_latency_ms, "max_turns"
        )

    def _build_trace(
        self,
        scenario: Scenario,
        turns: list[AgentTurn],
        session_id: str,
        total_tokens: int,
        total_latency_ms: int,
        terminal_reason: str,
    ) -> Trace:
        final_state = self._agent.get_state(session_id) if turns else {}
        return Trace(
            scenario_id=scenario.id,
            turns=turns,
            final_state=final_state,
            total_turns=len(turns),
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            terminal_reason=terminal_reason,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/application/test_runner.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/application/run_suite.py tests/unit/application/test_runner.py
git commit -m "feat: implement RunSuiteUseCase with tracer and constraint enforcement"
```

---

### Task 9: Sample 3-agent LangGraph customer support system

**Files:**
- Create: `example/agent/support_agent.py`

- [ ] **Step 1: Write the sample agent**

`example/agent/support_agent.py`:
```python
"""Sample 3-agent LangGraph customer support system.

This is a FIXTURE — the system under test for Dry Run's development.
Not part of the product. Not extensible. Time-boxed.

Agents:
  - triage_agent: routes to sales or support
  - sales_agent: handles inventory search, cart, checkout
  - support_agent: handles order status, refunds

Tools:
  - search_inventory(query: str) -> list[dict]
  - add_to_cart(item_id: str, quantity: int) -> dict
  - update_cart(item_id: str, quantity: int) -> dict
  - process_checkout(cart_id: str) -> dict
  - check_order_status(order_id: str) -> dict
  - initiate_refund(order_id: str, reason: str) -> dict
"""
from __future__ import annotations
import operator
from typing import Annotated, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# --- State ---

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str


# --- Tools ---

@tool
def search_inventory(query: str) -> list[dict]:
    """Search the product inventory."""
    return [
        {"id": "laptop-001", "name": "ThinkPad T14", "price": 899},
        {"id": "laptop-002", "name": "MacBook Air M3", "price": 1099},
        {"id": "tablet-001", "name": "iPad Air", "price": 599},
    ]


@tool
def add_to_cart(item_id: str, quantity: int = 1) -> dict:
    """Add an item to the cart."""
    return {"status": "added", "item_id": item_id, "quantity": quantity}


@tool
def update_cart(item_id: str, quantity: int) -> dict:
    """Update quantity of an item in the cart."""
    return {"status": "updated", "item_id": item_id, "quantity": quantity}


@tool
def process_checkout(cart_id: str = "default") -> dict:
    """Process checkout for the current cart."""
    return {"status": "confirmed", "order_id": "ORD-12345", "total": 899}


@tool
def check_order_status(order_id: str) -> dict:
    """Check the status of an order."""
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


@tool
def initiate_refund(order_id: str, reason: str) -> dict:
    """Initiate a refund for an order."""
    return {"order_id": order_id, "refund_status": "initiated", "reason": reason}


# --- Agent nodes ---

sales_tools = [search_inventory, add_to_cart, update_cart, process_checkout]
support_tools = [check_order_status, initiate_refund]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def triage_agent(state: AgentState) -> dict:
    """Route to sales or support based on user intent."""
    messages = state["messages"]
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a triage agent. Based on the user's message, decide:\n"
            "- If about purchasing, browsing, cart: respond with 'ROUTE:sales'\n"
            "- If about order status, refunds, issues: respond with 'ROUTE:support'\n"
            "- Otherwise: respond helpfully and ask for clarification."
        )},
        *messages,
    ])
    return {"messages": [response], "current_agent": "triage"}


def sales_agent(state: AgentState) -> dict:
    """Handle sales-related queries."""
    messages = state["messages"]
    response = llm.bind_tools(sales_tools).invoke([
        {"role": "system", "content": (
            "You are a sales agent. Help the customer find products, "
            "manage their cart, and complete purchases. Be helpful and concise."
        )},
        *messages,
    ])
    return {"messages": [response], "current_agent": "sales"}


def support_agent_node(state: AgentState) -> dict:
    """Handle support-related queries."""
    messages = state["messages"]
    response = llm.bind_tools(support_tools).invoke([
        {"role": "system", "content": (
            "You are a customer support agent. Help with order status, "
            "refunds, and issues. Be empathetic and efficient."
        )},
        *messages,
    ])
    return {"messages": [response], "current_agent": "support"}


# --- Routing ---

def route_from_triage(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    if "ROUTE:sales" in content:
        return "sales_agent"
    elif "ROUTE:support" in content:
        return "support_agent"
    return END


def should_use_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# --- Graph ---

sales_tool_node = ToolNode(sales_tools)
support_tool_node = ToolNode(support_tools)

graph = StateGraph(AgentState)

graph.add_node("triage", triage_agent)
graph.add_node("sales_agent", sales_agent)
graph.add_node("support_agent", support_agent_node)
graph.add_node("sales_tools", sales_tool_node)
graph.add_node("support_tools", support_tool_node)

graph.set_entry_point("triage")

graph.add_conditional_edges("triage", route_from_triage, {
    "sales_agent": "sales_agent",
    "support_agent": "support_agent",
    END: END,
})

graph.add_conditional_edges("sales_agent", should_use_tools, {
    "tools": "sales_tools",
    END: END,
})

graph.add_conditional_edges("support_agent", should_use_tools, {
    "tools": "support_tools",
    END: END,
})

graph.add_edge("sales_tools", "sales_agent")
graph.add_edge("support_tools", "support_agent")

# Compile with MemorySaver for state isolation via thread_id
compiled_graph = graph.compile(checkpointer=MemorySaver())
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from example.agent.support_agent import compiled_graph; print(type(compiled_graph))"`
Expected: Prints the compiled graph type (no crash)

- [ ] **Step 3: Commit**

```bash
git add example/agent/support_agent.py
git commit -m "feat: add sample 3-agent LangGraph customer support fixture"
```

---

### Task 10: LangGraphAdapter — implements AgentPort

**Files:**
- Create: `dryrun/adapters/outbound/langgraph/adapter.py`
- Test: `tests/contract/test_langgraph_adapter_contract.py`

- [ ] **Step 1: Write the contract test**

```python
"""Contract tests for LangGraphAdapter — must pass all AgentPort contracts."""
import os
import pytest
from tests.contract.test_agent_port_contract import TestAgentPortContract
from dryrun.adapters.outbound.langgraph.adapter import LangGraphAdapter


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for LangGraph adapter",
)
class TestLangGraphAdapterContract(TestAgentPortContract):
    """Runs the full AgentPort contract suite against LangGraphAdapter."""

    @pytest.fixture
    def adapter(self):
        from example.agent.support_agent import compiled_graph
        return LangGraphAdapter(compiled_graph)
```

- [ ] **Step 2: Write implementation**

`dryrun/adapters/outbound/langgraph/adapter.py`:
```python
"""LangGraphAdapter — implements AgentPort for LangGraph compiled graphs."""
from __future__ import annotations
import time
import uuid
from typing import Any
from langchain_core.messages import HumanMessage, BaseMessage
from dryrun.domain.ports.agent import AgentPort
from dryrun.domain.models.trace import AgentTurn, ToolCall


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
            return {}

    def _extract_output(self, messages: list) -> tuple[str, str, list[ToolCall]]:
        """Extract output text and tool calls from message history."""
        full_parts: list[str] = []
        visible_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(ToolCall(
                        tool_name=tc.get("name", "unknown"),
                        arguments=tc.get("args", {}),
                        output=None,
                        latency_ms=0,
                    ))

        # The last non-tool AI message is the visible output
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                if msg.type == "ai" and not getattr(msg, "tool_calls", None):
                    visible_parts.insert(0, msg.content)
                    break

        # Full output = all AI message contents
        for msg in messages:
            if hasattr(msg, "content") and msg.content and msg.type == "ai":
                full_parts.append(msg.content)

        output_text = "\n".join(full_parts) or ""
        visible_output_text = "\n".join(visible_parts) or output_text

        return output_text, visible_output_text, tool_calls

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimate based on message content length."""
        total_chars = sum(
            len(getattr(msg, "content", "") or "")
            for msg in messages
        )
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
                try:
                    result[k] = v
                except Exception:
                    result[k] = str(v)
        return result
```

- [ ] **Step 3: Run contract tests**

Run: `pytest tests/contract/ -v` (unit tests with stub pass without API key; LangGraph tests need `OPENAI_API_KEY`)
Expected: Stub tests PASS, LangGraph tests PASS (or SKIP if no API key)

- [ ] **Step 4: Commit**

```bash
git add dryrun/adapters/outbound/langgraph/adapter.py tests/contract/test_langgraph_adapter_contract.py
git commit -m "feat: implement LangGraphAdapter with state isolation and visible-output filtering"
```

---

### Task 11: OpenAI LLM client — implements LLMPort

**Files:**
- Create: `dryrun/adapters/outbound/openai/llm.py`
- Test: `tests/unit/adapters/test_openai_llm.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for OpenAIClient — implements LLMPort."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dryrun.adapters.outbound.openai.llm import OpenAIClient
from dryrun.domain.ports.llm import LLMPort


class TestOpenAIClient:
    def test_implements_llm_port(self):
        client = OpenAIClient(model="gpt-4o-mini")
        assert isinstance(client, LLMPort)

    @patch("dryrun.adapters.outbound.openai.llm.AsyncOpenAI")
    def test_complete_returns_string(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o-mini")
        result = asyncio.run(client.complete(
            [{"role": "user", "content": "Hi"}]
        ))
        assert result == "Hello!"

    @patch("dryrun.adapters.outbound.openai.llm.AsyncOpenAI")
    def test_temperature_passed_through(self, mock_openai_cls):
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(model="gpt-4o-mini")
        asyncio.run(client.complete(
            [{"role": "user", "content": "test"}],
            temperature=0.0,
        ))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/adapters/test_openai_llm.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write implementation**

`dryrun/adapters/outbound/openai/llm.py`:
```python
"""OpenAIClient — implements LLMPort using the OpenAI API."""
from __future__ import annotations
from openai import AsyncOpenAI
from dryrun.domain.ports.llm import LLMPort


class OpenAIClient(LLMPort):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/adapters/test_openai_llm.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add dryrun/adapters/outbound/openai/llm.py tests/unit/adapters/test_openai_llm.py
git commit -m "feat: implement OpenAIClient adapter for LLMPort"
```

---

### Task 12: Config loading + CLI wiring

**Files:**
- Create: `dryrun/config.py`
- Create: `example/dryrun.yaml`
- Create: `dryrun/adapters/inbound/cli/commands.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/integration/test_cli.py`

- [ ] **Step 1: Write the config test**

```python
"""Tests for dryrun config loading."""
import pytest
from pathlib import Path
from dryrun.config import DryRunConfig


CONFIG_YAML = """\
agent_module: "example.agent.support_agent"
agent_object: "compiled_graph"
scenarios_dir: "example/scenarios/"
models:
  synthetic_user: "gpt-4o-mini"
"""


class TestDryRunConfig:
    def test_load_from_yaml(self, tmp_path: Path):
        p = tmp_path / "dryrun.yaml"
        p.write_text(CONFIG_YAML)
        config = DryRunConfig.from_yaml(p)
        assert config.agent_module == "example.agent.support_agent"
        assert config.agent_object == "compiled_graph"
        assert config.models.synthetic_user == "gpt-4o-mini"

    def test_defaults(self):
        config = DryRunConfig(
            agent_module="example.agent.support_agent",
            agent_object="compiled_graph",
        )
        assert config.scenarios_dir == "scenarios/"
        assert config.models.synthetic_user == "gpt-4o-mini"
```

- [ ] **Step 2: Write config implementation**

`dryrun/config.py`:
```python
"""DryRunConfig — Pydantic config loaded from dryrun.yaml."""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    synthetic_user: str = "gpt-4o-mini"


class DryRunConfig(BaseModel):
    agent_module: str
    agent_object: str
    scenarios_dir: str = "scenarios/"
    models: ModelConfig = ModelConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> DryRunConfig:
        data = yaml.safe_load(path.read_text())
        return cls(**data)
```

- [ ] **Step 3: Run config test**

Run: `pytest tests/unit/test_config.py -v`
Expected: All PASS

- [ ] **Step 4: Create example/dryrun.yaml**

```yaml
agent_module: "example.agent.support_agent"
agent_object: "compiled_graph"
scenarios_dir: "example/scenarios/"
models:
  synthetic_user: "gpt-4o-mini"
```

- [ ] **Step 5: Write CLI integration test**

```python
"""Integration tests for the CLI."""
import pytest
from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower()

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "scenario" in result.output.lower()

    def test_run_missing_scenario_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
```

- [ ] **Step 6: Write CLI implementation**

`dryrun/adapters/inbound/cli/commands.py`:
```python
"""CLI commands — composition root. Wires adapters to use cases."""
from __future__ import annotations
import asyncio
import importlib
import sys
from pathlib import Path
import click
import yaml
from rich.console import Console
from rich.table import Table

from dryrun.config import DryRunConfig
from dryrun.domain.models.scenario import Scenario
from dryrun.application.run_suite import RunSuiteUseCase
from dryrun.adapters.outbound.langgraph.adapter import LangGraphAdapter
from dryrun.adapters.outbound.openai.llm import OpenAIClient

console = Console()


@click.group()
def cli():
    """Dry Run — simulation-based testing harness for LangGraph agents."""
    pass


@cli.command()
@click.argument("scenario_path", type=click.Path(exists=True))
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to dryrun.yaml config file")
def run(scenario_path: str, config_path: str | None):
    """Run a scenario against the agent and print the captured trace."""
    scenario_file = Path(scenario_path)

    # Load scenario
    with open(scenario_file) as f:
        scenario_data = yaml.safe_load(f)
    scenario = Scenario(**scenario_data)

    # Load config
    if config_path:
        config = DryRunConfig.from_yaml(Path(config_path))
    else:
        for candidate in [Path("dryrun.yaml"), Path("example/dryrun.yaml")]:
            if candidate.exists():
                config = DryRunConfig.from_yaml(candidate)
                break
        else:
            console.print("[red]No dryrun.yaml config found. Use --config.[/red]")
            sys.exit(1)

    # Load the agent
    try:
        module = importlib.import_module(config.agent_module)
        graph = getattr(module, config.agent_object)
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Failed to load agent: {e}[/red]")
        sys.exit(1)

    # Wire adapters
    agent_port = LangGraphAdapter(graph)
    llm_port = OpenAIClient(model=config.models.synthetic_user)
    runner = RunSuiteUseCase(agent_port=agent_port, llm_port=llm_port)

    # Run
    console.print(f"\n[bold]Running scenario:[/bold] {scenario.name}")
    console.print(f"[dim]ID: {scenario.id}[/dim]")
    console.print(f"[dim]Persona: {scenario.persona.tone} {scenario.persona.knowledge_level}[/dim]")
    console.print(f"[dim]Goal reveal: {scenario.persona.goal_reveal_strategy}[/dim]\n")

    trace = asyncio.run(runner.run_scenario(scenario))

    # Print trace
    _print_trace(trace)


def _print_trace(trace):
    """Pretty-print the trace using Rich."""
    console.print(f"\n[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold]Trace for scenario:[/bold] {trace.scenario_id}")
    console.print(f"[bold]Terminal reason:[/bold] {trace.terminal_reason}")
    console.print(f"[bold]Total turns:[/bold] {trace.total_turns}")
    console.print(f"[bold]Total tokens:[/bold] {trace.total_tokens}")
    console.print(f"[bold]Total latency:[/bold] {trace.total_latency_ms}ms")
    console.print(f"[bold green]{'=' * 60}[/bold green]\n")

    for turn in trace.turns:
        console.print(f"[bold cyan]--- Turn {turn.turn_number} (agent: {turn.agent_id}) ---[/bold cyan]")
        console.print(f"[dim]Input:[/dim] {turn.input_text}")
        console.print(f"[dim]Output:[/dim] {turn.output_text[:200]}")
        if turn.visible_output_text != turn.output_text:
            console.print(f"[dim]Visible:[/dim] {turn.visible_output_text[:200]}")
        if turn.tool_calls:
            table = Table(title="Tool Calls")
            table.add_column("Tool")
            table.add_column("Arguments")
            table.add_column("Output")
            for tc in turn.tool_calls:
                table.add_row(tc.tool_name, str(tc.arguments), str(tc.output)[:100])
            console.print(table)
        console.print(f"[dim]Latency: {turn.latency_ms}ms | Tokens: {turn.tokens_used}[/dim]\n")

    console.print(f"\n[bold]Final state:[/bold]")
    for k, v in trace.final_state.items():
        if k != "messages":
            console.print(f"  {k}: {v}")
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/unit/ tests/contract/ tests/integration/test_cli.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add dryrun/config.py example/dryrun.yaml dryrun/adapters/inbound/cli/commands.py tests/unit/test_config.py tests/integration/test_cli.py
git commit -m "feat: add config loading and CLI with 'dryrun run' command"
```

---

### Task 13: End-to-end smoke test

**Files:**
- Create: `tests/integration/test_e2e.py`

- [ ] **Step 1: Write the e2e test**

```python
"""End-to-end test: dryrun run one_scenario.yaml executes a full conversation."""
import os
import pytest
from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for e2e test",
)
class TestEndToEnd:
    def test_happy_path_scenario(self):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "example/scenarios/happy_path.yaml",
            "--config", "example/dryrun.yaml",
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "Trace for scenario" in result.output
        assert "Terminal reason" in result.output
        assert "happy-path-001" in result.output
```

- [ ] **Step 2: Run unit + contract tests (no API key needed)**

Run: `pytest tests/unit/ tests/contract/ -v`
Expected: All PASS

- [ ] **Step 3: Run e2e test (requires OPENAI_API_KEY)**

Run: `OPENAI_API_KEY=... pytest tests/integration/test_e2e.py -v -s`
Expected: PASS — full multi-turn conversation traced

- [ ] **Step 4: Verify the exit criterion manually**

Run: `dryrun run example/scenarios/happy_path.yaml --config example/dryrun.yaml`
Expected: Rich-formatted trace output showing turns, tool calls, terminal reason

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_e2e.py
git commit -m "test: add end-to-end smoke test for Phase 1 exit criterion"
```

---

## Task Summary

| Task | Description | Key Files | Test File |
|------|-------------|-----------|-----------|
| 1 | Project scaffold | `pyproject.toml`, `__init__.py` files | (pytest --co) |
| 2 | Scenario models | `domain/models/scenario.py` | `test_models.py` |
| 3 | Trace models | `domain/models/trace.py` | `test_models.py` |
| 4 | Port ABCs | `domain/ports/agent.py`, `domain/ports/llm.py` | `test_ports.py` |
| 5 | Contract tests + stub | — | `test_agent_port_contract.py` |
| 6 | YAML loading + examples | `example/scenarios/*.yaml` | `test_scenario_loading.py` |
| 7 | Synthetic user | `application/synthetic_user.py` | `test_synthetic_user.py` |
| 8 | Runner | `application/run_suite.py` | `test_runner.py` |
| 9 | Sample agent | `example/agent/support_agent.py` | (smoke check) |
| 10 | LangGraphAdapter | `adapters/outbound/langgraph/adapter.py` | `test_langgraph_adapter_contract.py` |
| 11 | OpenAI client | `adapters/outbound/openai/llm.py` | `test_openai_llm.py` |
| 12 | Config + CLI | `config.py`, `adapters/inbound/cli/commands.py` | `test_config.py`, `test_cli.py` |
| 13 | E2E smoke test | — | `test_e2e.py` |

## Known Implementation Notes

1. **Async pattern:** Runner is async internally, called with `asyncio.run()` from CLI. LLMPort is async (natural for HTTP). AgentPort is sync (LangGraph `invoke` is sync).
2. **MemorySaver required:** The sample agent MUST compile with `checkpointer=MemorySaver()` for `get_state()` to work in the LangGraphAdapter.
3. **Token estimation is approximate:** `chars / 4` in Phase 1. Proper callback-based tracking in Phase 2.
4. **Persona-drift check budget:** Exactly one retry. Prevents infinite loops on adversarial personas.
5. **visible_output_text heuristic:** Last AI message without tool_calls. Refined in Phase 2 if needed.
