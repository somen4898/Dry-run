"""Tests for CLI generate command."""

from click.testing import CliRunner
from dryrun.adapters.inbound.cli.commands import cli


class TestGenerateCommand:
    def test_generate_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--seeds" in result.output
        assert "--count" in result.output
        assert "--output" in result.output
