# Dry Run — Interview Deep-Dive Preparation

**Purpose:** Complete Q&A prep for walking an interviewer through the Dry Run project. Organized from high-level ("what is this") through architecture, trade-offs, failures, and extension questions. Every answer is phrased as you (Somen) speaking.

---

## Part 1: What Is This & Why Does It Exist

### Q1: Walk me through your most technically challenging project.

**A:** Dry Run is an open-source, simulation-based testing harness for multi-agent LangGraph systems. It runs agents against realistic, generated scenarios in an isolated sandbox, evaluates the full interaction trajectory across seven dimensions — four deterministic and three LLM-as-judge — and gates deployment in CI when agent behavior regresses.

The core insight is that agents are processes, not functions. They maintain state, call tools, make routing decisions, and interact across multiple turns. A single input-output assertion can't validate a process — it has to be simulated. Unit tests validate individual tools. Evals validate individual model calls. Neither validates the agent as a complete end-to-end system. Dry Run is that missing third layer.

### Q2: What problem were you solving? Who was the customer?

**A:** The customer is any AI engineer shipping a multi-agent LangGraph system to production. The problem is specific and well-documented: teams ship agents with no reliable way to verify the agent — as a complete process — actually does its job. An agent passes a developer's manual spot-check, ships to production, and fails on cases the developer never considered — a user changing their mind mid-task, an ambiguous refund request, a third escalation that should trigger a handoff.

The consequence is invisible until real users hit it. Dry Run catches these failures before deploy by simulating hundreds of scenarios you'd never write by hand.

### Q3: Why was this a priority? What motivated building it?

**A:** Three reasons. First, I was building multi-agent systems and personally experienced the gap — unit tests passed, evals passed, but the agent still broke in production on multi-turn interactions. Second, the competitive landscape has pieces of this (LangWatch Scenario, LangChain agentevals, DeepEval) but nobody ships the whole opinionated pipeline in one `pip install`, local-first, no SaaS. Third, I wanted a portfolio project that demonstrates the core skill of an AI engineer: wrapping unreliable AI components in deterministic engineering to produce reliable, reproducible system behavior.

### Q4: Isn't this just a wrapper around existing tools? What's actually new here?

**A:** I'm honest about this — simulation-based agent testing exists. LangWatch Scenario is the closest peer. What's different about Dry Run is the packaging: a single `pip install` that gives you the whole opinionated pipeline — adapter, runner, scorer, vector-backed store, regression diff, CI gate, and production-trace capture — wired together, local-first, no SaaS dependency. Existing tools give you the pieces; Dry Run gives you the harness.

The specific technical contributions that don't exist elsewhere in combination are: the τ²-bench information-asymmetry enforcement at the data model level, the orthogonal LLM-judge rubrics designed to avoid score correlation, the semantic failure lookup powered by Qdrant, and the single-command production-trace-to-golden-suite capture flow.

### Q5: Who are your competitors and when would someone choose them over you?

**A:**
- **LangWatch Scenario** — framework-agnostic, ships a red-team agent. Choose it if you want flexibility and are willing to assemble your own evaluation pipeline.
- **LangChain agentevals/openevals** — libraries of evaluators, not harnesses. Choose them if you already have a runner and just need scoring primitives.
- **LangSmith Multi-turn Evals** — LangChain's managed offering. Choose it if you're already paying for LangSmith and want online trajectory evaluation.
- **DeepEval / Confident AI** — broader hosted platform. Choose it if you want one vendor for evals + observability.

Choose Dry Run when you want batteries-included, fully local, no SaaS, with semantic failure lookup and CI gating out of the box. Trade-off: less flexibility than Scenario, no production observability like LangSmith, no managed UI.

---

## Part 2: Architecture & System Design

### Q6: Walk me through the architecture.

**A:** Dry Run follows hexagonal architecture — ports and adapters — with three concentric layers and a strict inward dependency rule.

The **domain layer** is the innermost ring. It has zero external dependencies — just Python stdlib and Pydantic. It contains three things: data models (Scenario, Trace, EvalResult), port interfaces (six ABCs that define contracts), and pure business logic services (deterministic scoring, diffing, aggregation).

The **application layer** is the middle ring. It orchestrates use cases — run a scenario suite, generate scenarios, capture production traces, look up similar failures. It depends only on domain. It receives port implementations via dependency injection.

The **adapters layer** is the outer ring. Inbound adapters drive the application (CLI via Click, CI via GitHub Actions). Outbound adapters are driven by the application (LangGraph, Qdrant, OpenAI, DeepEval, DSPy). Dependencies always point inward. Adapters never import from each other.

### Q7: Why hexagonal architecture? Isn't that overkill for a CLI tool?

**A:** Three reasons. First, the core value proposition of this project is pluggability — swapping LangGraph for CrewAI, or Qdrant for SQLite, or OpenAI for Anthropic should be a single-file change, not a rewrite. Hexagonal makes that structurally enforced, not just a good intention. Second, it makes the test strategy clean — domain services get pure unit tests with no mocks, application use cases get tests with mock ports, adapters get integration tests against real infrastructure. Third, the folder structure itself communicates architectural thinking — a recruiter opening the repo sees `domain/ports/`, `adapters/outbound/langgraph/` and immediately understands the design philosophy.

The cost is minimal — it's about 6 extra ABC files and a slightly deeper directory structure. The benefit is that every external dependency is quarantined.

### Q8: Explain the dependency rule. How do you enforce it?

**A:** Domain depends on nothing. Application depends only on domain. Adapters depend on both but never on each other. This means:
- `dryrun/domain/` never imports from `application/` or `adapters/`
- `dryrun/application/` never imports from `adapters/`
- No adapter imports another adapter

In v1, enforcement is by code review and the directory structure making violations obvious. You could add `import-linter` or a `ruff` plugin rule later, but the structure makes it hard to violate accidentally — if you're in `domain/services/scoring.py` and you try to import `openai`, it looks wrong.

### Q9: What are the six port interfaces and why those six?

**A:**

| Port | Abstracts | Why it's a port |
|---|---|---|
| **AgentPort** | Any agent framework (LangGraph, CrewAI, AutoGen) | The system under test changes per user — it must be pluggable |
| **StorePort** | Persistence + semantic search (Qdrant, SQLite, Postgres) | Storage backend is an infrastructure choice, not a domain concern |
| **LLMPort** | Any LLM provider (OpenAI, Anthropic, local) | The judge, synthetic user, and capture flow all need LLM calls — the provider shouldn't be hardcoded |
| **EmbeddingPort** | Any embedding model | Same reasoning — embedding provider is an infrastructure choice |
| **GeneratorPort** | Scenario generation strategy (DSPy, custom) | Generation approach will evolve (baseline → MIPROv2) — the application shouldn't know which |
| **ReporterPort** | Output format (terminal, PR comment, JSON) | How results are displayed is an output concern, not a domain concern |

The key insight: I drew the port boundaries at the points where I expect change. Agent frameworks change. LLM providers change. Storage backends change. The business logic of "run a scenario, score it, diff it" doesn't change.

### Q10: How does data flow through the system for a single scenario run?

**A:**
1. **CLI** (inbound adapter) reads the YAML scenario file, loads config, instantiates outbound adapters, injects them into `RunSuiteUseCase`
2. **RunSuiteUseCase** (application) creates a new session via `AgentPort.new_session()` — gets an isolated session ID
3. For each turn: inject user input via `AgentPort.step()` → get back an `AgentTurn` with `visible_output_text`
4. Pass ONLY `visible_output_text` to `SyntheticUser.next_message()` — the synthetic user never sees internal reasoning or tool calls
5. SyntheticUser (application) calls `LLMPort.complete()` to generate the next user message, runs persona-drift check
6. Loop until terminal signal (`GOAL_ACHIEVED`, `GOAL_ABANDONED`, `max_turns`, `timeout`, `token_budget`)
7. Build a `Trace` object from all collected `AgentTurn`s
8. (Phase 2+) Evaluate trace → diff against baseline → report via `ReporterPort`

### Q11: Why is the runner async internally but the CLI is sync?

**A:** The `LLMPort` interface is async because LLM API calls are I/O-bound — async is the natural fit for HTTP requests. The `AgentPort` is sync because LangGraph's `invoke()` is synchronous. The runner orchestrates both, so it's async to accommodate the LLM calls. The CLI uses `asyncio.run()` to bridge the sync Click command to the async runner. This is the standard pattern for CLI tools that need to call async code — no event loop complexity leaks into the user-facing layer.

### Q12: How do you handle state isolation between scenario runs?

**A:** Each scenario run gets its own session ID from `AgentPort.new_session()`. For the LangGraph adapter specifically, this maps to a unique `thread_id` in LangGraph's `RunnableConfig`, backed by `MemorySaver`. LangGraph's checkpointing system ensures complete state isolation — one scenario's conversation history, tool call state, and agent routing decisions never leak into another's. The contract tests verify this: `test_sessions_are_isolated` creates two sessions, advances one, and confirms the other's state is unaffected.

---

## Part 3: Key Design Decisions & Trade-offs

### Q13: What is the τ²-bench information-asymmetry principle and why does it matter?

**A:** This is borrowed from the τ²-bench paper. The core finding is: if a synthetic user can see the agent's internal reasoning, tool calls, and scratchpad, the synthetic user becomes unrealistically compliant — it essentially solves the agent's job for it, and your tests pass trivially.

In Dry Run, I enforce information asymmetry at two levels. First, at the **data model level**: `AgentTurn` has two separate fields — `output_text` (everything the agent produced) and `visible_output_text` (only what a real user would see). The adapter is responsible for filtering. Second, at the **call site level**: the runner passes only `visible_output_text` to the synthetic user's conversation history, never the full output. This is defense in depth — even if someone customizes the system prompt, the visibility constraint still holds.

Without this, every test looks like it passes because the synthetic user is too helpful.

### Q14: Why did you rewrite the LLM-judge rubrics from the original design?

**A:** The original three rubrics — Task Completion, Plan Adherence, Plan Quality — produced highly correlated scores on the same trace. When I tested them, a GPT-4o judge couldn't reliably distinguish between "did the agent complete the task" and "did the agent follow a good plan" — they're measuring almost the same thing from slightly different angles.

I redesigned them to be **genuinely orthogonal** — each measures something the other two cannot:

- **Goal Achievement**: outcome only — was the goal met? Doesn't care about path or style.
- **Trajectory Efficiency**: path cost only — how much waste? Doesn't care about outcome or style.
- **Persona Fit**: style only — appropriate tone and technical level? Doesn't care about outcome or path.

A high score on one tells you nothing about the others. I validate this with a calibration step: run all three judges against a labeled reference set and check pairwise correlation. If any pair correlates above 0.85, the prompts need rewriting.

### Q15: Why 4 deterministic + 3 LLM-judge? Why not all LLM?

**A:** Three reasons. First, **cost** — deterministic evaluators are free. No API calls. 57% of the evaluation costs nothing. Second, **reproducibility** — deterministic evaluators produce identical results on identical traces across runs. LLM judges have variance even at temperature 0. Third, **speed** — the four deterministic dimensions evaluate in microseconds. The three LLM judges need API round-trips.

The split maps naturally to what each approach is good at. Tool correctness is a set intersection — did the agent call the expected tools? That's trivially deterministic. Persona fit — was the tone appropriate for a frustrated novice user? That requires judgment. Use the right tool for the job.

### Q16: Why Qdrant? Why not SQLite or just files?

**A:** Qdrant earns its place through one specific use case: **semantic failure lookup** (UC-5). When a scenario fails in CI, Dry Run surfaces the top-3 most semantically similar past failures from the scenario store. This tells the developer whether this is a known pattern or a new one — before they start debugging.

You can't do this with SQLite or files without building your own embedding + similarity search layer. Qdrant gives you vector search out of the box, runs locally via docker-compose, and handles the scenario/result persistence that I need anyway.

The trade-off: it's heavier than SQLite. You need Docker. For v1, I accept this because the semantic search justifies it. If it turned out nobody used UC-5, I'd consider dropping to SQLite in v2.

### Q17: Why DSPy for scenario generation instead of just prompting an LLM directly?

**A:** DSPy gives you typed signatures — structured input/output contracts for LLM programs. Instead of writing a long prompt and hoping the LLM returns the right format, I define a `GenerateScenario` signature with typed input fields (agent_description, seed_scenarios, failure_patterns, scenario_type) and typed output fields (persona_goal, persona_tone, opening_input, expected_tools).

More importantly, DSPy's architecture supports optimization. In v1, I ship a baseline `dspy.Predict` — no optimization, just the typed signature. But in v1.1, once I have ~200+ failure traces, I can plug in `MIPROv2` to bias generation toward failure-exposing patterns. The same code path, just a different optimizer. I couldn't do that gracefully with raw prompts.

### Q18: Why did you defer MIPROv2 optimization to v1.1?

**A:** MIPROv2 needs training signal — it optimizes by learning which generated scenarios historically exposed failures. In an 8-week build against one sample agent, I won't have enough failure data for MIPROv2 to outperform a well-prompted baseline `dspy.Predict`. Shipping it anyway would mean either (a) the optimizer does nothing useful, or (b) I waste Phase 3 debugging an optimization algorithm instead of shipping the core features.

The baseline generator works regardless of training-data volume. It produces varied scenarios from seeds using the typed signature. Once enough failure traces accumulate (~200+), MIPROv2 activates. This is a deliberate scoping decision, not a deferral out of laziness.

### Q19: Why Click for the CLI instead of argparse or Typer?

**A:** Click gives composable command groups (`dryrun run`, `dryrun generate`, `dryrun capture`, `dryrun similar`, `dryrun calibrate-judges`) with clean auto-generated help text. Argparse can do this but requires more boilerplate. Typer is built on Click anyway and adds a type-annotation layer I don't need since I'm already using Pydantic for validation. Click is the sweet spot — mature, composable, no extra abstraction.

### Q20: Why DeepEval specifically for the LLM-judge dimensions?

**A:** DeepEval's `GEval` metric gives me structured JSON output with explicit rubrics. I define the rubric prompt, GEval calls the judge LLM, enforces JSON schema on the response, and handles retry on invalid JSON. I could build this myself — it's maybe 50 lines — but DeepEval also gives me access to their other metrics if I need them in later phases, and it's an active open-source project with good community adoption. The trade-off is an extra dependency, but it's lightweight and well-tested.

---

## Part 4: The AgentAdapter Abstraction

### Q21: Walk me through the AgentPort ABC. Why three methods?

**A:** Three methods is the minimum viable contract for driving any agent framework turn by turn:

- `new_session() → str`: Create a fresh, isolated session. Returns a session ID. This is how I guarantee no state leakage between scenario runs.
- `step(session_id, user_input) → AgentTurn`: Inject one user message, run one agent cycle, return the captured turn. The caller (runner) controls the loop — the adapter just does one step.
- `get_state(session_id) → dict`: Snapshot the agent's current state. Used for the trace and for state-transition analysis in the step efficiency evaluator.

I considered adding `reset()` or `close()` methods but they're not needed — `new_session()` creates fresh state, and Python garbage collection handles cleanup. Fewer methods = easier to implement for new frameworks.

### Q22: How does the LangGraphAdapter implement visible_output_text filtering?

**A:** The heuristic is: take the last AI message that has no `tool_calls` attached. In a typical LangGraph execution, the agent might produce several messages — internal routing decisions, tool call requests, tool responses, intermediate reasoning. The final message addressed to the user is the one without tool_calls.

This is the Phase 1 heuristic. It handles the common case well. For agents that produce multiple user-facing messages in one step, or that embed reasoning in the same message as the user response, the adapter would need a more sophisticated filter. That's a refinement for later phases, driven by actual failure cases rather than speculation.

### Q23: Why did you include contract tests with a stub adapter?

**A:** The contract tests serve two purposes. First, they prove the ABC is honest — a non-LangGraph stub can implement it. If the contract tests only passed with LangGraph-specific behavior, the abstraction would be fake. Second, they define the behavioral expectations that any adapter must meet: sessions are isolated, step returns an AgentTurn, turn numbers increment, visible_output_text is populated.

When someone adds a CrewAI adapter in v1.1, they inherit the contract test suite by overriding one fixture. If their adapter passes all contract tests, it works with the rest of the system. That's the whole point of the port pattern.

---

## Part 5: The Synthetic User

### Q24: How does goal-hiding work?

**A:** Each persona has a `goal_reveal_strategy` field — `incremental`, `upfront`, or `evasive`. This is embedded directly in the synthetic user's system prompt:

- **Incremental** (default): "Reveal information about your goal gradually, the way a real human user would. Do NOT state your full goal in your first message."
- **Upfront**: "State your full goal in your first message." (For transactional one-shot scenarios.)
- **Evasive**: "Only reveal goal details when the agent explicitly asks a clarifying question." (Tests whether the agent asks good clarifying questions.)

Without this, the LLM-powered synthetic user dumps the entire goal in turn 1 — "I want to buy a blue t-shirt in size M but I might change to size L after reviewing the cart." A real user would never say that. The agent's job becomes trivial, and the test is worthless.

### Q25: How does the persona-drift check work? Isn't it expensive?

**A:** After generating each synthetic user message, a separate LLM call checks: "Does this message stay in character as a human user?" The check prompt is tiny — the persona description plus the generated message, asking for a yes/no answer. Temperature 0. It costs maybe $0.0001 per check.

The retry budget is exactly one. If the check fails, the message is regenerated with a stronger persona reinforcement prompt at lower temperature (0.5 → 0.5). If the second check also fails, the message is accepted with a warning logged. This prevents infinite loops on adversarial persona prompts while catching the most common drift mode: "As an AI language model, I cannot..."

Two LLM calls per message (generation + drift check) in the normal case. Three in the drift-detected case. Cheap insurance against the best-documented failure mode in simulated-user systems.

### Q26: Why not just use a better system prompt instead of a separate drift check?

**A:** Because prompt engineering alone doesn't reliably prevent drift in long conversations. The synthetic user's system prompt already says "Never break character. Never acknowledge you are an AI." But after 8+ turns of complex conversation, the underlying LLM's safety training can override the persona instructions — especially when the conversation touches topics the LLM was trained to be cautious about.

The drift check is defense in depth. The system prompt is the first line. The drift check is the second. It's the same engineering principle as having both input validation and database constraints — either alone is insufficient.

---

## Part 6: Evaluation System

### Q27: Walk me through how tool correctness scoring works.

**A:** It's a set intersection. The scenario YAML specifies `required_tools: [search_inventory, add_to_cart, process_checkout]`. After the agent runs, I collect all tool names that were actually called from the trace: `{tc.tool_name for turn in trace.turns for tc in turn.tool_calls}`. The score is `|called ∩ required| / |required|`. If the agent called all three, score is 1.0. If it missed one, score is 0.67.

This is deliberately simple. It doesn't check call order (that's step efficiency's job) or argument values (that's argument correctness's job). Each dimension measures exactly one thing.

### Q28: How does argument correctness handle semantic arguments?

**A:** There are three levels. For arguments with exact expected values (like `size: "L"`), it's an exact match. For arguments where only structure matters (like "the search query should be a string"), it checks key presence and type. For arguments explicitly tagged as `semantic` in the scenario YAML, it uses embedding similarity with `text-embedding-3-small`.

The key design decision: embeddings are called ONLY for arguments explicitly tagged as semantic. The default for any argument is exact match with type tolerance. I made this opt-in because embedding every argument is expensive and usually unnecessary. A search query like "laptop" vs "laptops" needs semantic matching. A cart item ID like "laptop-001" does not.

### Q29: How does step efficiency detect problems?

**A:** Three detectors, all pure graph-path analysis — no LLM needed:

1. **Loop detection**: Count visits per graph node. If any node is visited more than 3 times, that's a loop. The agent is stuck.
2. **Thrashing detection**: Check for A→B→A→B oscillation patterns in state transitions. The agent can't decide what to do.
3. **Redundancy detection**: Consecutive identical tool calls with identical arguments. The agent forgot it already did this.

Score is `1.0 - penalty`, where penalty accumulates per detected issue. It's fast, deterministic, and catches the three most common multi-agent failure modes I've seen in production.

### Q30: How do you ensure the three LLM judges are actually orthogonal?

**A:** By design and by measurement. By design: each judge prompt explicitly says what to evaluate AND what to ignore. Goal Achievement says "Do NOT consider how many turns it took, how the agent communicated, whether the path was efficient. Only the outcome." Trajectory Efficiency says "A trajectory with zero waste scores 1.0 regardless of whether the goal was ultimately reached."

By measurement: in week 4, I run a calibration step — all three judges evaluate a fixed reference set of 30 labeled traces. I compute pairwise Pearson correlation between the three dimension scores. If any pair correlates above 0.85, the prompts need rewriting. This is a `dryrun calibrate-judges` CLI command, not a one-off script — it's repeatable.

---

## Part 7: CI/CD & Production Concerns

### Q31: How does the CI gate work?

**A:** Two GitHub Actions jobs. On every PR: `dryrun-golden` runs only the golden suite — the must-always-pass scenarios. Fast (~30 seconds, ~$0.01). On merge to main: `dryrun-full` runs every scenario (~5-10 min, ~$0.50). Both use Qdrant as a service container.

The gate logic: fail the build (exit code 1) if the aggregate score drops below the configured threshold OR any golden-suite scenario fails. Pass (exit code 0) otherwise. There's also a configurable `regression_delta` — if the aggregate score drops more than 5% from the last known-good baseline, that's a regression even if the absolute score is still above threshold.

### Q32: Why two CI jobs instead of one?

**A:** Cost. 140 scenarios × 3 LLM-judge calls × 5 turns each is real money on every PR. The golden-only job runs maybe 10-15 scenarios that represent critical paths — happy path, key edge cases, known past failures. It catches obvious regressions fast and cheap. The full suite catches subtle regressions but only runs on merge to main when the cost is justified.

This is a deliberate operational trade-off, documented in the README.

### Q33: How does the production trace capture flow work?

**A:** One command: `dryrun capture --from-trace path/to/trace.json --output scenarios/golden/`. An LLM reads the raw production trace and extracts: what the user was trying to do, what persona they displayed, where the agent failed. It constructs a `Scenario` object with `golden=True` and serializes it to YAML.

Critically, the developer reviews and commits the scenario — capture does NOT auto-add to the golden suite. This is a deliberate design decision. Auto-adding could introduce flaky scenarios from noisy production traces. The human reviews the extracted scenario, validates it makes sense, and commits it. From that point, it's regression-tested forever.

### Q34: How do you handle API cost transparency?

**A:** Cost is reported before each run (estimated based on scenario count, max turns, and model pricing) and after (actual based on token counts). The reporter shows "Token cost: 18,420 tokens (~$0.09)" in the run summary. This is a non-functional requirement from the PRD — teams need to budget for agent testing just like they budget for compute.

---

## Part 8: Testing Strategy

### Q35: Walk me through your test pyramid.

**A:**
- **Unit tests (domain)**: Pure logic — scoring, diffing, aggregation. No mocks needed because domain services have no dependencies. These are fast, deterministic, and comprehensive.
- **Unit tests (application)**: Use case orchestration with mock ports. Test that the runner terminates on GOAL_ACHIEVED, respects max_turns, passes only visible_output_text to the synthetic user. All behavior verified without touching any infrastructure.
- **Contract tests**: The AgentPort contract suite — any adapter implementation must pass these. Includes a stub non-LangGraph adapter to prove the abstraction is honest. This is the key test that prevents the interface from quietly degrading into a LangGraph-only contract.
- **Integration tests**: Real adapters against real infrastructure. LangGraphAdapter against the sample agent (needs OpenAI API key). CLI tests via Click's `CliRunner`. E2E smoke test that validates the Phase 1 exit criterion.

The split matters: I can run 90% of tests instantly with no API keys. Only integration tests need infrastructure. CI runs everything.

### Q36: Why contract tests? Isn't that unusual for a Python project?

**A:** Contract tests are common in systems with pluggable implementations — which is exactly what hexagonal architecture produces. The contract test suite defines the behavioral expectations of the `AgentPort`: sessions are isolated, step returns an `AgentTurn`, turn numbers increment, `visible_output_text` is populated, `get_state` returns a dict.

Any new adapter inherits the suite by overriding one fixture. If the CrewAI adapter passes all contract tests, it works with the entire system without any integration testing against every possible scenario. This is cheaper and more reliable than testing every adapter-scenario combination.

---

## Part 9: What Went Wrong & What I'd Change

### Q37: What was harder than expected?

**A:** Three things.

First, the **visible_output_text extraction heuristic** in the LangGraphAdapter. LangGraph produces a stream of messages — human, AI, tool calls, tool responses, routing decisions. Figuring out which message is "the one addressed to the user" is non-trivial. My Phase 1 heuristic (last AI message without tool_calls) works for the common case but breaks for agents that produce multiple user-facing messages in one step. It's the area most likely to need refinement.

Second, the **persona drift problem** was worse than I expected. In long conversations (8+ turns), even GPT-4o-mini breaks character. The drift check catches it, but I initially underestimated how often it fires — about 15% of messages in my testing. The retry budget of one is enough to fix most cases, but it means every 7th or 8th synthetic user message costs an extra LLM call.

Third, the **MIPROv2 scoping decision** was hard emotionally. I wanted to ship the optimization layer — it's the more impressive technical contribution. But honestly, 8 weeks against one sample agent doesn't produce enough failure data. Shipping a broken optimizer would be worse than shipping no optimizer. I had to choose shipping reliable software over shipping impressive-looking software.

### Q38: What would you do differently if you started over?

**A:** Two things. First, I'd define the `AgentPort` even more minimally — `step()` returns a raw string instead of an `AgentTurn`, and the tracer lives in the application layer rather than being the adapter's responsibility. Right now the adapter does too much work constructing the `AgentTurn` with all its fields. It makes adapter implementations heavier than they need to be.

Second, I'd consider whether Qdrant is truly necessary for v1. The semantic failure lookup (UC-5) justifies it architecturally, but operationally it means every user needs Docker. An in-memory FAISS index with file-backed persistence might have been lighter for v1, with Qdrant as an upgrade path.

### Q39: What's the biggest risk to this project?

**A:** LLM-judge reliability. The three judge dimensions are the core of the evaluation, and LLMs are inherently non-deterministic. Even at temperature 0, the same trace can score differently on consecutive runs. I mitigate this with the orthogonal rubric design, the calibration step, and by making 57% of the evaluation deterministic. But the fundamental risk remains: if users can't trust the judge scores, they can't trust the regression detection.

The mitigation path is: track judge score variance across runs on identical traces, and flag dimensions with high variance for prompt refinement.

---

## Part 10: Extension Questions (The "Can You Improvise" Test)

### Q40: How would you add CrewAI support?

**A:** One new file: `adapters/outbound/crewai/adapter.py` that implements `AgentPort`. The `new_session()` method creates a fresh CrewAI crew instance. The `step()` method feeds user input to the crew and captures the output. The `get_state()` method snapshots the crew's state. Add it to the adapter factory in the CLI. Zero changes to domain or application layers. This is exactly what the hexagonal architecture is designed for.

### Q41: How would you add parallel scenario execution?

**A:** The runner currently loops through scenarios sequentially. For parallel execution, I'd use `asyncio.gather` to run multiple scenarios concurrently, each with its own session ID (isolation is already guaranteed by `new_session()`). The rate-limiting constraint is LLM API calls — I'd add a semaphore to cap concurrent API requests. The `RunSuiteUseCase` interface doesn't change; the orchestration inside changes from a for-loop to a gather.

### Q42: How would you handle multi-modal agents (images, audio)?

**A:** The `AgentTurn` model would need a `visible_output_content` field that's a union type — text, image, audio. The `LLMPort` interface would need multimodal message support. The LLM-judge prompts would need to handle non-text content. The deterministic evaluators would largely be unaffected — tool correctness, argument correctness, step efficiency, and constraint adherence don't care about modality.

The biggest architectural impact: the synthetic user would need a multimodal LLM to "see" images the agent sends. This is a v2.0 feature at minimum.

### Q43: How would you scale this to 10,000 scenarios?

**A:** Three bottlenecks. First, **LLM API calls** — the synthetic user and three judges make API calls per scenario. At 10K scenarios × ~10 calls each, that's 100K API calls. Solution: async with rate-limiting, batch where possible, use the cheapest model that maintains quality (gpt-4o-mini for synthetic user, gpt-4o only for judges).

Second, **Qdrant storage** — 10K scenarios with embeddings is trivially within Qdrant's capacity, but the `populate_similar_failures_into` batch call would need to be optimized for 10K results.

Third, **CI runtime** — 10K scenarios can't run on every PR. The golden-only/full split becomes critical. The golden suite stays at ~50 scenarios. The full 10K suite runs nightly or on merge to main only.

### Q44: What if an interviewer asks: "How would you make this work for our specific use case?"

**A:** I'd ask three questions: What agent framework are you using? What does a "failure" look like for your agent? What's your current testing approach? Then I'd map their answers onto the architecture:
- Their framework → which `AgentPort` adapter to implement
- Their failure modes → what scenario YAML expectations to define
- Their current testing → where Dry Run fills the gap (likely the multi-turn simulation layer)

This is the power of the hexagonal architecture — I can discuss their specific case in terms of the port they'd implement, without changing the core system.

---

## Part 11: The "Show You're a Senior Engineer" Questions

### Q45: What does this project tell an interviewer about your engineering ability?

**A:** Three things. First, I can **design systems** — hexagonal architecture with dependency rules, port abstractions at the right boundaries, a clean test strategy. Second, I can **make real engineering trade-offs** — deferring MIPROv2 because the training data doesn't support it, choosing orthogonal rubrics over the obvious correlated ones, accepting the Qdrant operational cost because the semantic search justifies it. Third, I can **wrap unreliable AI in deterministic engineering** — the 4+3 evaluator split, the information-asymmetry enforcement, the persona-drift check. That's the core skill of an AI engineer.

### Q46: What's the engineering principle that ties the whole project together?

**A:** "Agents are processes, not functions. A process cannot be validated by input-output assertions — it must be simulated." Every design decision flows from this principle. The multi-turn runner exists because processes have state. The synthetic user exists because processes interact with users. The seven evaluation dimensions exist because process quality is multi-dimensional. The regression diff exists because process behavior changes over time.

### Q47: If you had one more week, what would you add?

**A:** The calibration pipeline for the LLM judges. Right now the orthogonal rubric design is based on reasoning about what should be independent. With one more week, I'd label 30 traces by hand, run all three judges, compute pairwise correlation, and either validate the design or identify which prompt needs refinement. Data beats theory.

### Q48: What's the most non-obvious thing you learned building this?

**A:** That the synthetic user is the hardest component, not the evaluator. Everyone thinks the hard part of agent testing is scoring — did the agent do a good job? But actually, the hard part is simulating a realistic user. Real users don't dump their entire goal in turn 1. Real users change their minds. Real users get frustrated. Real users don't say "As an AI language model." Getting the synthetic user right — goal-hiding, information asymmetry, persona drift — is what makes the simulation realistic. Without it, every test passes trivially and the whole system is worthless.
