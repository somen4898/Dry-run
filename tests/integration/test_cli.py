"""Integration tests for the CLI."""

from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


class TestCLI:
    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output.lower()

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_missing_scenario_file(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
