"""Tests for DSPy-based scenario generator."""

import pytest
from unittest.mock import patch
from dryrun.application.generator import ScenarioGenerator
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints
from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter


def _seed_scenario() -> Scenario:
    return Scenario(
        id="seed-001", name="Seed", description="Buy a laptop for college",
        persona=Persona(goal="Buy budget laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi, I need a laptop",
        expectations=Expectation(required_tools=["search_inventory", "add_to_cart"], required_tool_args={}, output_must_contain=["order"], terminal_state=None),
        constraints=Constraints(max_turns=8),
    )


MOCK_GENERATED_YAML = """
id: "gen-001"
name: "Generated Scenario"
description: "Buy headphones as a gift"
persona:
  goal: "Find wireless headphones under $50"
  tone: "casual"
  knowledge_level: "novice"
  background: "College student buying a birthday gift"
opening_input: "Hey, looking for some headphones"
expectations:
  required_tools: ["search_inventory"]
  required_tool_args: {}
  output_must_contain: []
  terminal_state: null
constraints:
  max_turns: 8
"""


class TestScenarioGenerator:
    @pytest.mark.asyncio
    async def test_generate_returns_scenarios(self):
        store = InMemoryStoreAdapter()
        generator = ScenarioGenerator(store=store)

        with patch.object(generator, "_call_dspy", return_value=MOCK_GENERATED_YAML):
            results = await generator.generate(seeds=[_seed_scenario()], count=1)

        assert len(results) == 1
        assert results[0].id == "gen-001"
        assert results[0].persona.tone == "casual"

    @pytest.mark.asyncio
    async def test_skips_near_duplicates(self):
        store = InMemoryStoreAdapter()
        existing = _seed_scenario()
        await store.upsert_scenario(existing, [1.0] * 384)

        generator = ScenarioGenerator(store=store)

        with patch.object(generator, "_call_dspy", return_value=MOCK_GENERATED_YAML):
            with patch("dryrun.application.generator.embed_scenario", return_value=[1.0] * 384):
                results = await generator.generate(seeds=[_seed_scenario()], count=1)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_validates_against_pydantic(self):
        store = InMemoryStoreAdapter()
        generator = ScenarioGenerator(store=store)

        invalid_yaml = "id: missing_fields\nname: Bad"

        with patch.object(generator, "_call_dspy", return_value=invalid_yaml):
            results = await generator.generate(seeds=[_seed_scenario()], count=1)

        assert len(results) == 0
