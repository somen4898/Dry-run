# Dry Run — Hexagonal Architecture Design Spec

**Date:** 2026-05-15
**Status:** Approved
**Author:** Somen Bandishti

---

## Overview

Dry Run follows a hexagonal (ports-and-adapters) architecture. The system is organized into three concentric layers with a strict inward dependency rule:

- **Domain** — Core business logic, data models, and port interfaces. Zero external dependencies.
- **Application** — Use-case orchestration. Depends only on domain.
- **Adapters** — Infrastructure implementations. Inbound adapters drive the application; outbound adapters are driven by it.

Dependencies always point inward: adapters → application → domain. The domain never imports from application or adapters.

---

## Domain Layer (`dryrun/domain/`)

### Models (`domain/models/`)

All data models are Pydantic v2 BaseModels. No business logic, no I/O.

| File | Models | Purpose |
|---|---|---|
| `scenario.py` | `Scenario`, `Persona`, `Expectation`, `Constraints` | Test scenario definition. Persona includes `goal_reveal_strategy` for τ²-bench information asymmetry. |
| `trace.py` | `Trace`, `AgentTurn`, `ToolCall` | Execution trace capture. `AgentTurn` includes `visible_output_text` — the subset of output the synthetic user sees. |
| `evaluation.py` | `EvalResult`, `DimensionScore`, `RunResult` | Evaluation results. `EvalResult` includes `similar_past_failures` for UC-5. |

### Ports (`domain/ports/`)

Abstract Base Classes defining the contracts between layers. Each port has exactly one responsibility.

| Port | File | Methods | v1 Adapter |
|---|---|---|---|
| `AgentPort` | `agent.py` | `new_session()`, `step()`, `get_state()` | `LangGraphAdapter` |
| `StorePort` | `store.py` | `upsert_scenario()`, `get_golden_suite()`, `find_similar_failures()`, `save_run_result()`, `get_last_run_aggregate()`, `populate_similar_failures_into()` | `QdrantStore` |
| `LLMPort` | `llm.py` | `complete(messages, temperature, response_format)` | `OpenAIClient` |
| `EmbeddingPort` | `embedding.py` | `embed(text) → list[float]`, `embed_batch(texts) → list[list[float]]` | `OpenAIEmbeddingClient` |
| `GeneratorPort` | `generator.py` | `generate(agent_description, seeds, failure_patterns, scenario_type) → Scenario` | `DSPyGenerator` |
| `ReporterPort` | `reporter.py` | `report(run_result, diff, config)` | `TerminalReporter`, `PRCommentReporter` |

### Services (`domain/services/`)

Pure business logic. No I/O, no external dependencies. Functions take domain models in and return domain models out.

| File | Responsibility |
|---|---|
| `scoring.py` | Four deterministic evaluators: tool correctness (set intersection), argument correctness (exact + structural + semantic-tagged), step efficiency (loop/thrash/redundancy detection), constraint adherence (counting). |
| `diffing.py` | Compare two `RunResult` objects. Produce newly-passing, newly-failing, score delta per dimension. |
| `aggregation.py` | Weighted score aggregation across 7 dimensions. Threshold checking. Pass/fail verdict. Golden-suite enforcement. |

**Why deterministic evaluators are domain services:** They are pure functions — trace in, `DimensionScore` out. No LLM calls, no I/O. They belong in the innermost ring.

**Why LLM-judge evaluators are NOT domain services:** They require an LLM call (via `LLMPort`). The application layer orchestrates them by calling `LLMPort` with the judge prompts and parsing the structured JSON response.

---

## Application Layer (`dryrun/application/`)

Use-case orchestrators. Each file represents one user-facing capability. They receive port implementations via constructor injection and compose domain services with port calls.

| File | Use Case | Ports Used |
|---|---|---|
| `run_suite.py` | Run scenarios → evaluate → diff → report | `AgentPort`, `StorePort`, `LLMPort`, `EmbeddingPort`, `ReporterPort` |
| `generate.py` | Generate new scenarios from seeds | `GeneratorPort` |
| `capture.py` | Production trace → golden scenario YAML | `LLMPort` |
| `similar_lookup.py` | Ad-hoc semantic search (UC-5) | `StorePort` |
| `synthetic_user.py` | Persona role-play with drift check and goal-hiding | `LLMPort` |

### The Run Loop (in `run_suite.py`)

```
for each scenario:
    session_id = agent_port.new_session()
    synthetic_user = SyntheticUser(scenario.persona, llm_port)
    current_input = scenario.opening_input

    for turn in range(max_turns):
        agent_turn = agent_port.step(session_id, current_input)
        # CRITICAL: pass only visible_output_text to synthetic user
        next_input = synthetic_user.next_message(agent_turn.visible_output_text)
        if terminal_signal(next_input): break

    trace = build_trace(...)
    deterministic_scores = domain.services.scoring.evaluate(trace, scenario)
    llm_judge_scores = call_llm_judges(trace, scenario)  # via LLMPort
    eval_result = domain.services.aggregation.aggregate(deterministic_scores + llm_judge_scores)
    store_port.save_run_result(eval_result)

baseline = store_port.get_last_run_aggregate()
diff = domain.services.diffing.diff(current_run, baseline)
store_port.populate_similar_failures_into(failed_results)
reporter_port.report(run_result, diff)
```

### SyntheticUser (`synthetic_user.py`)

Application-level component, not a domain service (it does I/O via `LLMPort`), not an adapter (it's not wrapping infrastructure).

Responsibilities:
- Construct the persona system prompt with goal-reveal strategy
- Pass only `visible_output_text` to conversation history (enforced at call site, not just prompt)
- Persona-drift self-check: one check per message, one retry on failure, accept with warning on second failure
- Detect terminal signals: `GOAL_ACHIEVED`, `GOAL_ABANDONED`

---

## Adapters Layer (`dryrun/adapters/`)

### Inbound Adapters (`adapters/inbound/`)

These drive the application — they are how users interact with Dry Run.

**CLI (`inbound/cli/commands.py`):**
- Click-based CLI commands: `run`, `generate`, `capture`, `similar`, `report`, `diff`, `status`, `calibrate-judges`
- **Composition root:** reads `dryrun.yaml`, instantiates all outbound adapters, injects them into application use cases
- Maps `--golden-only`, `--pr-comment`, `--full` flags to use case parameters

**CI (`inbound/ci/github_actions.py`):**
- Exit code mapping (0 = pass, 1 = fail/regression)
- Summary artifact output
- Delegates to `ReporterPort` for formatting

### Outbound Adapters (`adapters/outbound/`)

These are driven by the application — they implement the port interfaces.

| Directory | Implements | External Dependency | Notes |
|---|---|---|---|
| `langgraph/adapter.py` | `AgentPort` | `langgraph`, `langchain` | Uses `RunnableConfig` with unique `thread_id` per session. Filters `visible_output_text` from full output. |
| `qdrant/store.py` | `StorePort` | `qdrant-client` | Two collections: `scenarios` (vector = embedded description), `run_results` (vector = embedded failure reason). |
| `openai/llm.py` | `LLMPort` | `openai` | Wraps chat completions. Configurable model per use case (judge=gpt-4o, synthetic-user=gpt-4o-mini). |
| `openai/embeddings.py` | `EmbeddingPort` | `openai` | `text-embedding-3-small`. Batch support. |
| `deepeval/judge.py` | (wraps `LLMPort`) | `deepeval` | GEval metric wrappers for the 3 orthogonal judge dimensions. Not its own port — implementation detail of how structured judge output is produced. |
| `dspy/generator.py` | `GeneratorPort` | `dspy-ai` | v1: `dspy.Predict(GenerateScenario)`. v1.1: MIPROv2 optimization. |
| `reporters/terminal.py` | `ReporterPort` | `rich` | Terminal output with colors, tables, pass/fail marks. |
| `reporters/pr_comment.py` | `ReporterPort` | GitHub API | Markdown PR comment with `<details>` blocks. Updated in-place per PR. |

---

## Dependency Rules

| Layer | Can import from | Cannot import from |
|---|---|---|
| `domain/` | Python stdlib, Pydantic | `application/`, `adapters/`, any external SDK |
| `application/` | `domain/` | `adapters/`, any external SDK |
| `adapters/` | `domain/`, `application/` | Other adapters (no cross-adapter imports) |

**Enforcement:** No tooling in v1 — enforced by code review and the directory structure. A `ruff` import-rule plugin or `import-linter` can be added later.

---

## Testing Strategy

| Test Type | Directory | What It Tests | Dependencies |
|---|---|---|---|
| Unit | `tests/unit/domain/` | Domain services (scoring, diffing, aggregation) | None — pure functions |
| Unit | `tests/unit/application/` | Use case orchestration | Mock ports |
| Integration | `tests/integration/adapters/` | Adapter implementations against real infrastructure | Qdrant (docker), OpenAI API |
| Contract | `tests/contract/` | Any `AgentPort` implementation passes the contract. Includes a stub non-LangGraph adapter. | None |

---

## TRD Component → Hexagonal Layer Mapping

| TRD Component | Hexagonal Location | Rationale |
|---|---|---|
| `adapter/base.py` (AgentAdapter ABC) | `domain/ports/agent.py` | It's a port interface — belongs in domain |
| `adapter/langgraph.py` | `adapters/outbound/langgraph/adapter.py` | Outbound adapter implementing AgentPort |
| `runner/runner.py` | `application/run_suite.py` | Use-case orchestration |
| `runner/user.py` | `application/synthetic_user.py` | Application component using LLMPort |
| `runner/tracer.py` | Absorbed into `AgentPort` contract + `LangGraphAdapter` | Trace capture is the adapter's responsibility |
| `eval/deterministic.py` | `domain/services/scoring.py` | Pure logic, no I/O |
| `eval/llm_judge.py` | `adapters/outbound/deepeval/judge.py` | Requires LLM I/O |
| `eval/evaluator.py` (orchestrator) | `application/run_suite.py` | Orchestration lives in application layer |
| `store/store.py` | `adapters/outbound/qdrant/store.py` | Infrastructure adapter |
| `generator/generator.py` | `adapters/outbound/dspy/generator.py` | Infrastructure adapter |
| `capture/capture.py` | `application/capture.py` | Use-case orchestration |
| `reporter/reporter.py` | `adapters/outbound/reporters/terminal.py` | Output adapter |
| `reporter/pr_comment.py` | `adapters/outbound/reporters/pr_comment.py` | Output adapter |
| `schema.py` (all data models) | `domain/models/` (split into 3 files) | Domain models |
| `config.py` | `dryrun/config.py` (top-level) | Cross-cutting concern, not layer-specific |
| `cli.py` | `adapters/inbound/cli/commands.py` | Inbound adapter — composition root |
