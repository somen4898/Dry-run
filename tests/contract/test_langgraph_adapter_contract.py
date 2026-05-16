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
