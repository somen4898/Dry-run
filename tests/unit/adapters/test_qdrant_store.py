"""Tests for QdrantAdapter — uses mocks, no live Qdrant required."""

from unittest.mock import patch
from dryrun.adapters.outbound.qdrant.store import QdrantAdapter
from dryrun.domain.ports.store import StorePort


class TestQdrantAdapter:
    def test_implements_store_port(self):
        with patch("dryrun.adapters.outbound.qdrant.store.QdrantAsyncClient"):
            adapter = QdrantAdapter(url="http://localhost:6333", prefix="test_")
            assert isinstance(adapter, StorePort)
