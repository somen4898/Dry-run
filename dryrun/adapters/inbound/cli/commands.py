"""CLI commands — composition root. Wires adapters to use cases."""

from __future__ import annotations
import asyncio
import importlib
import sys
from pathlib import Path
import click
import yaml
from rich.console import Console
from rich.table import Table

from dryrun.config import DryRunConfig
from dryrun.domain.models.scenario import Scenario
from dryrun.application.run_suite import RunSuiteUseCase
from dryrun.adapters.outbound.langgraph.adapter import LangGraphAdapter
from dryrun.adapters.outbound.llm_factory import create_llm

console = Console()


@click.group()
def cli():
    """Dry Run — simulation-based testing harness for LangGraph agents."""
    pass


@cli.command()
@click.argument("scenario_path", type=click.Path(exists=True))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to dryrun.yaml config file",
)
@click.option(
    "--concurrency",
    "max_concurrent",
    type=int,
    default=5,
    help="Max parallel scenario executions (default: 5)",
)
def run(scenario_path: str, config_path: str | None, max_concurrent: int):
    """Run a scenario (file) or suite (directory) against the agent."""
    target = Path(scenario_path)

    # Load config
    if config_path:
        config = DryRunConfig.from_yaml(Path(config_path))
    else:
        for candidate in [Path("dryrun.yaml"), Path("example/dryrun.yaml")]:
            if candidate.exists():
                config = DryRunConfig.from_yaml(candidate)
                break
        else:
            console.print("[red]No dryrun.yaml config found. Use --config.[/red]")
            sys.exit(1)

    # Load the agent
    try:
        module = importlib.import_module(config.agent_module)
        graph = getattr(module, config.agent_object)
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Failed to load agent: {e}[/red]")
        sys.exit(1)

    # Wire adapters — provider comes from config, not hardcoded
    agent_port = LangGraphAdapter(graph)
    llm_port = create_llm(config.models, purpose="synthetic_user")
    runner = RunSuiteUseCase(agent_port=agent_port, llm_port=llm_port, config=config)

    # Set env vars so the sample agent uses the same provider
    import os

    os.environ.setdefault("DRYRUN_LLM_PROVIDER", config.models.provider)
    os.environ.setdefault("DRYRUN_AGENT_MODEL", config.models.agent)

    if target.is_dir():
        # Suite mode — run all scenarios in directory (parallel)
        console.print(f"\n[bold]Running suite:[/bold] {target} (concurrency: {max_concurrent})")
        run_result = asyncio.run(runner.run_suite(target, max_concurrent=max_concurrent))
        _print_run_result(run_result)
    else:
        # Single scenario mode
        scenario_data = yaml.safe_load(target.read_text())
        scenario = Scenario(**scenario_data)

        console.print(f"\n[bold]Running scenario:[/bold] {scenario.name}")
        console.print(f"[dim]ID: {scenario.id}[/dim]")
        console.print(f"[dim]Persona: {scenario.persona.tone} {scenario.persona.knowledge_level}[/dim]")
        console.print(f"[dim]Goal reveal: {scenario.persona.goal_reveal_strategy}[/dim]\n")

        trace = asyncio.run(runner.run_scenario(scenario))
        _print_trace(trace)


def _print_run_result(result):
    """Pretty-print a RunResult summary using Rich."""
    console.print(f"\n[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold]Suite Run:[/bold] {result.run_id}")
    console.print(f"[bold]Timestamp:[/bold] {result.timestamp}")
    console.print(f"[bold]Total scenarios:[/bold] {result.total_scenarios}")
    console.print(f"[bold green]Passed:[/bold green] {result.passed}")
    console.print(f"[bold red]Failed:[/bold red] {result.failed}")
    console.print(f"[bold]Aggregate score:[/bold] {result.aggregate_score:.2f}")
    console.print(f"[bold]Token cost:[/bold] {result.token_cost_actual}")
    console.print(f"[bold green]{'=' * 60}[/bold green]\n")

    if result.per_dimension_scores:
        table = Table(title="Per-Dimension Averages")
        table.add_column("Dimension")
        table.add_column("Score")
        for dim, score in result.per_dimension_scores.items():
            table.add_row(dim, f"{score:.3f}")
        console.print(table)


def _print_trace(trace):
    """Pretty-print the trace using Rich."""
    console.print(f"\n[bold green]{'=' * 60}[/bold green]")
    console.print(f"[bold]Trace for scenario:[/bold] {trace.scenario_id}")
    console.print(f"[bold]Terminal reason:[/bold] {trace.terminal_reason}")
    console.print(f"[bold]Total turns:[/bold] {trace.total_turns}")
    console.print(f"[bold]Total tokens:[/bold] {trace.total_tokens}")
    console.print(f"[bold]Total latency:[/bold] {trace.total_latency_ms}ms")
    console.print(f"[bold green]{'=' * 60}[/bold green]\n")

    for turn in trace.turns:
        console.print(
            f"[bold cyan]--- Turn {turn.turn_number} (agent: {turn.agent_id}) ---[/bold cyan]"
        )
        console.print(f"[dim]Input:[/dim] {turn.input_text}")
        console.print(f"[dim]Output:[/dim] {turn.output_text[:200]}")
        if turn.visible_output_text != turn.output_text:
            console.print(f"[dim]Visible:[/dim] {turn.visible_output_text[:200]}")
        if turn.tool_calls:
            table = Table(title="Tool Calls")
            table.add_column("Tool")
            table.add_column("Arguments")
            table.add_column("Output")
            for tc in turn.tool_calls:
                table.add_row(tc.tool_name, str(tc.arguments), str(tc.output)[:100])
            console.print(table)
        console.print(f"[dim]Latency: {turn.latency_ms}ms | Tokens: {turn.tokens_used}[/dim]\n")

    console.print("\n[bold]Final state:[/bold]")
    for k, v in trace.final_state.items():
        if k != "messages":
            console.print(f"  {k}: {v}")
