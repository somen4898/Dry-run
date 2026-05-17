# Phase 3 & 4 Design — Storage, Generation, Diff, CI Gate, Release

> **Scope:** Covers Phase 3 (Storage, Generation, Diff) and Phase 4 (CI Gate, Capture, Release) as a single spec. Both share the Qdrant storage infrastructure.

---

## 1. Architecture Overview

Layered build with a `StorePort` abstraction. Each piece is testable independently via an in-memory store adapter. Qdrant is the production adapter.

**Dependency order:** StorePort → Embeddings → Generation → Diff → Similar-Failure Lookup → CI Gate → Capture → Package

**Key decisions:**
- All models default to `claude-haiku-4-5` (cost control)
- Local embeddings via `sentence-transformers/all-MiniLM-L6-v2` (free, no API)
- DSPy baseline generator (`dspy.Predict`, no MIPROv2 optimization)
- Qdrant for vector search + metadata, InMemoryStore for tests
- GitHub Actions template + generic exit codes for CI portability

---

## 2. Storage Layer

### StorePort (domain port)

```python
class StorePort(ABC):
    # Scenarios
    async def upsert_scenario(self, scenario: Scenario, embedding: list[float]) -> None
    async def find_similar_scenarios(self, embedding: list[float], top_k: int = 5) -> list[Scenario]
    async def is_near_duplicate(self, embedding: list[float], threshold: float = 0.92) -> bool

    # Run results
    async def save_run(self, result: RunResult) -> str  # returns run_id
    async def get_run(self, run_id: str) -> RunResult | None
    async def get_latest_run(self) -> RunResult | None

    # Golden suite
    async def get_golden_scenarios(self) -> list[Scenario]
    async def mark_golden(self, scenario_id: str) -> None

    # Similar failures
    async def find_similar_failures(self, embedding: list[float], top_k: int = 3) -> list[FailureMatch]
```

### Adapters

| Adapter | Purpose | Where |
|---------|---------|-------|
| `QdrantAdapter` | Production — vector search + metadata filtering | `dryrun/adapters/outbound/qdrant/store.py` |
| `InMemoryStoreAdapter` | Tests — dicts + cosine similarity | `dryrun/adapters/outbound/memory/store.py` |

### Qdrant Collections

| Collection | Purpose | Embedding? |
|---|---|---|
| `scenarios` | All scenarios with YAML + embedding | Yes (384-dim, MiniLM) |
| `run_results` | Per-run metadata + per-scenario eval results | No (metadata only) |

### Embedding

- Model: `sentence-transformers/all-MiniLM-L6-v2` (local, free, 384-dim)
- Input: scenario `description + persona.goal + str(expectations.required_tools)`
- Runs locally, no API cost

---

## 3. Scenario Generation

### CLI Command

```bash
dryrun generate --seeds example/scenarios/ --count 5 --output generated/
```

### DSPy Signature

```python
class GenerateScenario(dspy.Signature):
    """Generate a new, diverse test scenario for an AI agent based on seed examples."""
    seed_scenarios: str = dspy.InputField(desc="2-3 example scenarios as YAML")
    variation_strategy: str = dspy.InputField(desc="How to vary: tone, goal, edge-case, persona")
    new_scenario: str = dspy.OutputField(desc="A complete scenario YAML with unique id, persona, expectations")
```

### Variation Strategies (rotated automatically)

- `tone_shift` — same goal, different persona tone
- `edge_case` — add constraints that stress the agent (low turns, evasive user)
- `goal_variation` — same domain, different goal
- `persona_swap` — different knowledge level + background

### Flow

1. Load seed scenarios from directory
2. Embed each seed, check Qdrant for near-duplicates (skip if >0.92 similarity)
3. Call DSPy `dspy.Predict` with signature + randomly selected variation strategy
4. Validate generated YAML against `Scenario` Pydantic model
5. Embed + dedup check against existing scenarios
6. Store in Qdrant, write to output directory

### Dedup Gate

Before writing, embed the generated scenario and check `is_near_duplicate()`. If too similar to existing, retry with a different variation strategy (max 2 retries, then skip).

### Model

Configurable via `models.generator` in dryrun.yaml. Defaults to `claude-haiku-4-5`.

---

## 4. Diff Reporting

### CLI Flag

```bash
dryrun run scenarios/ --diff
```

### Flow

1. After suite run completes, save `RunResult` to store via StorePort
2. If `--diff` flag, fetch previous run via `get_latest_run()`
3. Compute diff: per-scenario comparison by scenario_id

### Domain Model

```python
class ScenarioDelta(BaseModel):
    scenario_id: str
    previous_score: float
    current_score: float
    delta: float
    dimension_deltas: dict[str, float]  # per-dimension changes

class RunDiff(BaseModel):
    previous_run_id: str
    current_run_id: str
    score_delta: float
    newly_failing: list[ScenarioDelta]
    newly_passing: list[ScenarioDelta]
    stable_pass: int
    stable_fail: int
```

### Reporter Output

```
Score: 0.72 -> 0.68 (-0.04)

  Newly failing (2):
    x refund-001: goal_achievement 0.9 -> 0.4 (-0.5)
    x expert-001: tool_correctness 0.8 -> 0.5 (-0.3)

  Newly passing (1):
    v ambiguous-001: 0.45 -> 0.78 (+0.33)

  Stable (7): 5 pass, 2 fail
```

### Golden Suite Enforcement

If any scenario with `golden: true` fails, the run is marked as blocked regardless of aggregate score. The reporter shows these prominently.

---

## 5. Similar-Failure Lookup (UC-5)

### When It Fires

For each failed scenario after evaluation.

### Flow

1. Embed the failed scenario (description + goal + failed dimension names)
2. Query `find_similar_failures()` — filter by `passed=False` in stored results
3. Return top-3 matches with their failure reasons and run context

### Reporter Output

```
x refund-001 (score: 0.42)
  tool_correctness: 0.50 — Missing: [initiate_refund]
  goal_achievement: 0.40 — Agent acknowledged but never processed refund

  Similar past failures:
    -> refund-003 (run 2026-05-15): same tool_correctness failure
    -> order-cancel-001 (run 2026-05-14): similar goal pattern
```

### Domain Model

```python
class FailureMatch(BaseModel):
    scenario_id: str
    run_id: str
    run_timestamp: str
    similarity_score: float
    failed_dimensions: list[str]
    failure_reasons: list[str]
```

---

## 6. CI Gate

### CLI Command

```bash
dryrun gate scenarios/ --config dryrun.yaml
# or with options:
dryrun run scenarios/ --diff --golden-only
```

### Exit Codes

- `0` — all pass, no golden failures, no regression beyond threshold
- `1` — any golden failure, or aggregate regression > threshold

### Gate Logic

```python
def compute_gate_result(run: RunResult, diff: RunDiff | None, config: GateConfig) -> int:
    # Golden failures always block
    if any golden scenario failed:
        return 1
    # Regression check (only if diff available)
    if diff and diff.score_delta < -config.regression_threshold:
        return 1
    return 0
```

### JSON Summary Artifact

Outputs `dryrun-report.json` for CI tooling to consume:
```json
{
  "passed": true,
  "total": 10,
  "pass_count": 8,
  "fail_count": 2,
  "aggregate_score": 0.72,
  "golden_failures": [],
  "regression": -0.02
}
```

### GitHub Actions Template

```yaml
- uses: actions/setup-python@v5
- run: pip install dryrun-agents
- run: dryrun run scenarios/ --diff --golden-only
```

### Optional PR Comment

Script that posts summary table to PR via `gh pr comment`. Not a full Action — just a shell script in the repo.

---

## 7. Capture Flow

### CLI Command

```bash
dryrun capture trace.json --output scenarios/new_scenario.yaml
```

### Flow

1. Read production trace JSON (from observability tooling — LangSmith, LangFuse, etc.)
2. LLM (haiku) extracts: persona goal, tone, knowledge level, required tools, constraints
3. Generates a `Scenario` YAML
4. Validates against Pydantic model
5. Embeds + dedup check
6. Writes to output path

### Trace Input Format

Flexible — supports a minimal contract:
```json
{
  "messages": [{"role": "user/assistant", "content": "..."}],
  "tool_calls": [{"name": "...", "args": {...}}],
  "metadata": {"session_id": "...", "timestamp": "..."}
}
```

---

## 8. Config Changes

```yaml
models:
  provider: "anthropic"
  synthetic_user: "claude-haiku-4-5"
  agent: "claude-haiku-4-5"
  judge: "claude-haiku-4-5"
  generator: "claude-haiku-4-5"

store:
  provider: "qdrant"       # or "memory" for tests
  url: "http://localhost:6333"
  collection_prefix: "dryrun_"

gate:
  regression_threshold: 0.05
  golden_must_pass: true
```

### New Config Models

```python
class StoreConfig(BaseModel):
    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    collection_prefix: str = "dryrun_"

class GateConfig(BaseModel):
    regression_threshold: float = 0.05
    golden_must_pass: bool = True
```

---

## 9. Package & Release

- **Package name:** `dryrun-agents` (PyPI)
- **Entry point:** `dryrun = dryrun.adapters.inbound.cli.commands:cli`
- **Docker:** `docker-compose.yml` with Qdrant + dryrun services
- **README:** Problem statement, architecture diagram, quick-start (15 min), competitive positioning (LangWatch, DeepEval, LangSmith), worked example
- **CI:** pytest + ruff (already have pre-push hook)

---

## 10. New File Structure

```
dryrun/
  domain/
    ports/
      store.py              # StorePort ABC
    models/
      diff.py               # RunDiff, ScenarioDelta, FailureMatch
    services/
      embedding.py          # sentence-transformers wrapper
      diff.py               # compute_diff logic
  application/
    generator.py            # DSPy-based scenario generator
    capture.py              # Production trace -> scenario YAML
    gate.py                 # CI gate logic (exit code computation)
  adapters/
    outbound/
      qdrant/
        store.py            # QdrantAdapter(StorePort)
      memory/
        store.py            # InMemoryStoreAdapter(StorePort)
    inbound/
      cli/
        commands.py         # Add: generate, capture, gate commands
  config.py                 # Add: StoreConfig, GateConfig
```

---

## 11. Dependencies Added

```toml
[project.dependencies]
# Existing...
qdrant-client = ">=1.9"
sentence-transformers = ">=3.0"
dspy = ">=2.5"
```

---

## 12. Exit Criteria

**Phase 3:**
- `dryrun generate` produces valid new scenarios from seeds
- A second suite run produces a correct diff showing score delta
- Similar-failure lookup returns relevant past failures for failed scenarios

**Phase 4:**
- `pip install dryrun-agents` installs cleanly
- Quick-start works in under 15 minutes
- GitHub Actions CI gate returns correct exit codes
- `dryrun capture trace.json` produces valid scenario YAML
- Docker-compose runs Qdrant + dryrun locally
