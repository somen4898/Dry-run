"""DSPy-based scenario generator — produces varied scenarios from seeds."""

from __future__ import annotations
import logging
import random
import yaml
import dspy
from dryrun.domain.models.scenario import Scenario
from dryrun.domain.ports.store import StorePort
from dryrun.domain.services.embedding import embed_scenario

logger = logging.getLogger(__name__)

VARIATION_STRATEGIES = [
    "tone_shift: Keep the same goal but change persona tone (e.g., polite→frustrated, casual→direct)",
    "edge_case: Add constraints that stress the agent (fewer max_turns, evasive goal reveal, impatient user)",
    "goal_variation: Same domain but a different user goal (e.g., buy→return, search→compare)",
    "persona_swap: Different knowledge level and background (e.g., novice→expert, student→professional)",
]


class GenerateScenario(dspy.Signature):
    """Generate a new, diverse test scenario for an AI agent based on seed examples.
    Output ONLY valid YAML for a scenario with fields: id, name, description, persona (goal, tone, knowledge_level, background), opening_input, expectations (required_tools, required_tool_args, output_must_contain), constraints (max_turns)."""

    seed_scenarios: str = dspy.InputField(desc="2-3 example scenarios as YAML for reference")
    variation_strategy: str = dspy.InputField(desc="How to make the new scenario different from seeds")
    new_scenario: str = dspy.OutputField(desc="Complete scenario as valid YAML")


class ScenarioGenerator:
    def __init__(self, store: StorePort, model: str = "claude-haiku-4-5"):
        self._store = store
        self._model = model

    async def generate(
        self, seeds: list[Scenario], count: int = 5, max_retries: int = 2
    ) -> list[Scenario]:
        """Generate new scenarios from seeds, dedup against store."""
        generated: list[Scenario] = []

        for i in range(count):
            strategy = random.choice(VARIATION_STRATEGIES)
            seed_yaml = self._format_seeds(random.sample(seeds, min(2, len(seeds))))

            for attempt in range(max_retries + 1):
                raw_yaml = await self._call_dspy(seed_yaml, strategy)
                scenario = self._parse_and_validate(raw_yaml, i)
                if scenario is None:
                    logger.warning("Generated invalid YAML (attempt %d)", attempt + 1)
                    continue

                # Dedup check
                embedding = embed_scenario(scenario)
                if await self._store.is_near_duplicate(embedding, threshold=0.92):
                    logger.info("Skipping near-duplicate: %s", scenario.id)
                    if attempt < max_retries:
                        strategy = random.choice(VARIATION_STRATEGIES)
                        continue
                    break

                # Store and collect
                await self._store.upsert_scenario(scenario, embedding)
                generated.append(scenario)
                break

        return generated

    async def _call_dspy(self, seed_yaml: str, strategy: str) -> str:
        """Call DSPy predictor. Override in tests."""
        lm = dspy.LM(f"anthropic/{self._model}")
        with dspy.context(lm=lm):
            predictor = dspy.Predict(GenerateScenario)
            result = predictor(seed_scenarios=seed_yaml, variation_strategy=strategy)
            return result.new_scenario

    def _format_seeds(self, seeds: list[Scenario]) -> str:
        """Format seed scenarios as YAML string."""
        return "\n---\n".join(
            yaml.dump(s.model_dump(exclude_none=True), default_flow_style=False)
            for s in seeds
        )

    def _parse_and_validate(self, raw_yaml: str, index: int) -> Scenario | None:
        """Parse YAML and validate as Scenario. Returns None if invalid."""
        try:
            cleaned = raw_yaml.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])

            data = yaml.safe_load(cleaned)
            if not isinstance(data, dict):
                return None
            return Scenario(**data)
        except Exception as e:
            logger.debug("Validation failed: %s", e)
            return None
