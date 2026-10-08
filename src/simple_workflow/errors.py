"""Error hierarchy for the workflow engine."""

from __future__ import annotations

from collections.abc import Iterable


class WorkflowError(Exception):
    """Base class for every error raised by the engine."""


class ScriptSyntaxError(WorkflowError):
    """The pipeline script could not be parsed."""


class CompileError(WorkflowError):
    """The pipeline graph is invalid (bad types, unknown tools, cycles, ...)."""

    def __init__(self, errors: Iterable[str]):
        self.errors: list[str] = [str(e) for e in errors]
        super().__init__("\n".join(self.errors) if self.errors else "compile error")


class ToolExecutionError(WorkflowError):
    """A tool failed while the workflow was running."""


class InputError(WorkflowError):
    """Runtime inputs do not match what the compiled plan expects."""
