"""Tool base classes."""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any

from ..signature import Signature


def timeout_from_env(default: float | None = None) -> float | None:
    """Per-node timeout in seconds, from ``SW_TOOL_TIMEOUT``.

    Returns ``default`` (unlimited by default) when the variable is unset or
    blank, so a workflow is not cut off by an arbitrary upper bound.
    """
    raw = os.environ.get("SW_TOOL_TIMEOUT")
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass(slots=True)
class RunContext:
    """Per-run configuration handed to every tool invocation.

    ``endpoints`` maps a tool name to its full URL and takes precedence over the
    environment.  When a tool is not listed there, its endpoint is read from
    ``TOOLS_<NAME>_ENDPOINT`` (e.g. ``TOOLS_MSST_ENDPOINT``).

    ``timeout`` bounds a single node's tool call, in seconds.  It defaults to
    ``None`` (no limit), so long-running tools are not cut off unless a timeout
    is explicitly configured.
    """

    endpoints: dict[str, str] = field(default_factory=dict)
    api_keys: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def endpoint_for(self, tool_name: str) -> str | None:
        if tool_name in self.endpoints:
            return self.endpoints[tool_name]
        return os.environ.get(f"TOOLS_{tool_name.upper()}_ENDPOINT")

    def api_key_for(self, tool_name: str) -> str | None:
        if tool_name in self.api_keys:
            return self.api_keys[tool_name]
        return os.environ.get(f"TOOLS_{tool_name.upper()}_API_KEY")


class Tool(abc.ABC):
    """A callable node in the workflow graph.

    ``run`` receives already-ordered positional parameters: the literal
    arguments from the script first, followed by the values that arrived over
    incoming edges.  It returns a single value.
    """

    name: str
    signature: Signature

    def __init__(self, name: str, signature: Signature) -> None:
        self.name = name
        self.signature = signature

    @abc.abstractmethod
    async def run(self, params: list[Any], ctx: RunContext) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}{self.signature.describe()}>"
