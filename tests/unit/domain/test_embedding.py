"""Tests for embedding service."""

import pytest
from dryrun.domain.services.embedding import embed_scenario, embed_failure
from dryrun.domain.models.scenario import Scenario, Persona, Expectation, Constraints


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="s1", name="Test", description="Buy a laptop for college",
        persona=Persona(goal="Buy budget laptop", tone="polite", knowledge_level="novice", background="Student"),
        opening_input="Hi",
        expectations=Expectation(required_tools=["search", "add_to_cart"], required_tool_args={}, output_must_contain=[], terminal_state=None),
        constraints=Constraints(),
    )


class TestEmbedScenario:
    def test_returns_list_of_floats(self, scenario):
        embedding = embed_scenario(scenario)
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_similar_scenarios_have_high_similarity(self):
        s1 = Scenario(
            id="s1", name="A", description="Buy a laptop",
            persona=Persona(goal="Buy laptop", tone="polite", knowledge_level="novice", background="Student"),
            opening_input="Hi",
            expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[], terminal_state=None),
            constraints=Constraints(),
        )
        s2 = Scenario(
            id="s2", name="B", description="Purchase a laptop computer",
            persona=Persona(goal="Purchase laptop", tone="polite", knowledge_level="novice", background="Student"),
            opening_input="Hello",
            expectations=Expectation(required_tools=["search"], required_tool_args={}, output_must_contain=[], terminal_state=None),
            constraints=Constraints(),
        )
        e1 = embed_scenario(s1)
        e2 = embed_scenario(s2)
        dot = sum(a * b for a, b in zip(e1, e2))
        assert dot > 0.8  # Normalized vectors, so dot product = cosine similarity


class TestEmbedFailure:
    def test_returns_list_of_floats(self, scenario):
        embedding = embed_failure(scenario, ["tool_correctness", "goal_achievement"])
        assert isinstance(embedding, list)
        assert len(embedding) == 384
