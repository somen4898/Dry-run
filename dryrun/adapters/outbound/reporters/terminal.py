"""Terminal reporter — Rich-based evaluation result display."""

from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dryrun.domain.models.evaluation import EvalResult, RunResult
from dryrun.domain.ports.reporter import ReporterPort


class TerminalReporter(ReporterPort):
    def __init__(self, console: Console | None = None):
        self._console = console or Console()

    def report_scenario(self, result: EvalResult) -> None:
        """Print a single scenario result as a compact table."""
        status = "[bold green]PASS[/]" if result.passed else "[bold red]FAIL[/]"
        self._console.print(
            f"\n{status} {result.scenario_id} (score: {result.aggregate_score:.2f})"
        )

        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Dimension", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Pass", justify="center")
        table.add_column("Reason")

        for dim in result.dimensions:
            pass_mark = "[green]✓[/]" if dim.passed else "[red]✗[/]"
            score_style = "green" if dim.passed else "red"
            table.add_row(
                dim.dimension,
                f"[{score_style}]{dim.score:.2f}[/]",
                pass_mark,
                dim.reason[:60] if dim.reason else "",
            )

        self._console.print(table)

    def report_suite(self, result: RunResult) -> None:
        """Print the full suite summary."""
        self._console.print("\n")

        # Header
        overall = (
            "[bold green]SUITE PASSED[/]" if result.failed == 0 else "[bold red]SUITE FAILED[/]"
        )
        self._console.print(
            Panel(
                f"{overall}\n"
                f"Scenarios: {result.total_scenarios} total, "
                f"[green]{result.passed} passed[/], [red]{result.failed} failed[/]\n"
                f"Aggregate Score: {result.aggregate_score:.2f}\n"
                f"Tokens Used: {result.token_cost_actual:,}",
                title="Evaluation Report",
                border_style="blue",
            )
        )

        # Per-dimension averages
        if result.per_dimension_scores:
            table = Table(title="Per-Dimension Averages", show_header=True, header_style="bold")
            table.add_column("Dimension", style="cyan")
            table.add_column("Avg Score", justify="right")

            for dim, score in sorted(result.per_dimension_scores.items()):
                style = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                table.add_row(dim, f"[{style}]{score:.2f}[/]")

            self._console.print(table)

        # Failed scenarios detail
        failed_results = [r for r in result.eval_results if not r.passed]
        if failed_results:
            self._console.print("\n[bold red]Failed Scenarios:[/]")
            for r in failed_results:
                self._console.print(f"  [red]✗[/] {r.scenario_id} (score: {r.aggregate_score:.2f})")
                failed_dims = [d for d in r.dimensions if not d.passed]
                for d in failed_dims:
                    self._console.print(f"    └─ {d.dimension}: {d.score:.2f} — {d.reason[:80]}")
