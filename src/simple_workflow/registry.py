"""Tool registry."""

from __future__ import annotations

from collections.abc import Iterator

from .errors import WorkflowError
from .tools.api import make_api_tools
from .tools.base import Tool
from .tools.local import local_tools


class ToolRegistry:
    """Maps tool names to :class:`Tool` implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "<none>"
            raise WorkflowError(f"unknown tool {name!r}; available tools: {available}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def names(self) -> list[str]:
        return sorted(self._tools)


def build_registry() -> ToolRegistry:
    """Build the default registry: local tools + the three API tools.

    API tool endpoints are resolved per call from ``TOOLS_<NAME>_ENDPOINT``.
    """
    registry = ToolRegistry()
    for tool in local_tools():
        registry.register(tool)
    for tool in make_api_tools():
        registry.register(tool)
    return registry
