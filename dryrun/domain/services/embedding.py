"""Embedding service — local sentence-transformers for scenario vectors."""

from __future__ import annotations
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from dryrun.domain.models.scenario import Scenario


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (cached singleton)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_scenario(scenario: Scenario) -> list[float]:
    """Embed a scenario using description + goal + required tools."""
    text = (
        f"{scenario.description}. "
        f"Goal: {scenario.persona.goal}. "
        f"Tools: {', '.join(scenario.expectations.required_tools)}"
    )
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_failure(scenario: Scenario, failed_dimensions: list[str]) -> list[float]:
    """Embed a failure context for similar-failure lookup."""
    text = (
        f"{scenario.description}. "
        f"Goal: {scenario.persona.goal}. "
        f"Failed: {', '.join(failed_dimensions)}"
    )
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
