"""End-to-end test: dryrun run one_scenario.yaml executes a full conversation."""

import os
import pytest
from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for e2e test",
)
class TestEndToEnd:
    def test_happy_path_scenario(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "example/scenarios/happy_path.yaml",
                "--config",
                "example/dryrun.yaml",
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert "Trace for scenario" in result.output
        assert "Terminal reason" in result.output
        assert "happy-path-001" in result.output
