"""Tests for dryrun config loading."""

from pathlib import Path
from dryrun.config import DryRunConfig, ThresholdsConfig


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


class TestThresholdsConfig:
    def test_defaults(self):
        t = ThresholdsConfig()
        assert t.aggregate == 0.70
        assert t.tool_correctness == 0.80
        assert t.argument_correctness == 0.75
        assert t.step_efficiency == 0.70
        assert t.constraint_adherence == 0.90
        assert t.goal_achievement == 0.70
        assert t.trajectory_efficiency == 0.65
        assert t.persona_fit == 0.70

    def test_config_includes_thresholds(self):
        cfg = DryRunConfig(agent_module="x", agent_object="y")
        assert cfg.thresholds.aggregate == 0.70

    def test_override_from_dict(self):
        cfg = DryRunConfig(
            agent_module="x",
            agent_object="y",
            thresholds={"aggregate": 0.85, "persona_fit": 0.60},
        )
        assert cfg.thresholds.aggregate == 0.85
        assert cfg.thresholds.persona_fit == 0.60
        assert cfg.thresholds.tool_correctness == 0.80  # default preserved
