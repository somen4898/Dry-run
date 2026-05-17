"""Store factory — creates the right StorePort based on config."""

from __future__ import annotations
from dryrun.config import StoreConfig
from dryrun.domain.ports.store import StorePort


def create_store(config: StoreConfig) -> StorePort:
    """Create a StorePort implementation based on config."""
    if config.provider == "memory":
        from dryrun.adapters.outbound.memory.store import InMemoryStoreAdapter
        return InMemoryStoreAdapter()
    elif config.provider == "qdrant":
        from dryrun.adapters.outbound.qdrant.store import QdrantAdapter
        return QdrantAdapter(url=config.url, prefix=config.collection_prefix)
    else:
        raise ValueError(f"Unknown store provider: '{config.provider}'")
