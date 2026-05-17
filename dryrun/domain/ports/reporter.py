"""Reporter port — defines the contract for evaluation result reporting."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dryrun.domain.models.evaluation import EvalResult, RunResult


class ReporterPort(ABC):
    @abstractmethod
    def report_scenario(self, result: EvalResult) -> None:
        """Report a single scenario evaluation result."""
        ...

    @abstractmethod
    def report_suite(self, result: RunResult) -> None:
        """Report the full suite run result."""
        ...
