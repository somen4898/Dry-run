"""Tests for dryrun config loading."""

from pathlib import Path
from dryrun.config import DryRunConfig


CONFIG_YAML = """\
agent_module: "example.agent.support_agent"
agent_object: "compiled_graph"
scenarios_dir: "example/scenarios/"
models:
  synthetic_user: "gpt-4o-mini"
"""


class TestDryRunConfig:
    def test_load_from_yaml(self, tmp_path: Path):
        p = tmp_path / "dryrun.yaml"
        p.write_text(CONFIG_YAML)
        config = DryRunConfig.from_yaml(p)
        assert config.agent_module == "example.agent.support_agent"
        assert config.agent_object == "compiled_graph"
        assert config.models.synthetic_user == "gpt-4o-mini"

    def test_defaults(self):
        config = DryRunConfig(
            agent_module="example.agent.support_agent",
            agent_object="compiled_graph",
        )
        assert config.scenarios_dir == "scenarios/"
        assert config.models.provider == "anthropic"
        assert config.models.synthetic_user == "claude-sonnet-4-6"
        assert config.models.agent == "claude-sonnet-4-6"
