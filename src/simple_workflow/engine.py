"""Workflow execution engine.

Runs a compiled :class:`~simple_workflow.compiler.Plan` on a DAG scheduler:
nodes whose dependencies are satisfied execute concurrently, so independent
branches of the pipeline overlap.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .compiler import Node, Plan
from .errors import InputError, ToolExecutionError
from .registry import ToolRegistry
from .tools.base import RunContext
from .types import ValueType, normalize, type_name


@dataclass(slots=True)
class InputValue:
    """A concrete runtime input: its type and its already-decoded value."""

    type: ValueType
    data: Any


@dataclass(slots=True)
class NodeResult:
    node_id: str
    value: Any
    seconds: float


class Executor:
    def __init__(self, plan: Plan, registry: ToolRegistry) -> None:
        self.plan = plan
        self.registry = registry

    async def run(
        self,
        inputs: dict[str, InputValue],
        ctx: RunContext | None = None,
    ) -> tuple[dict[str, Any], list[NodeResult]]:
        ctx = ctx or RunContext()
        values: dict[str, list[Any]] = {}
        timings: list[NodeResult] = []

        indegree = {node_id: 0 for node_id in self.plan.order}
        dependents: dict[str, list[str]] = {node_id: [] for node_id in self.plan.order}
        for node_id, node in self.plan.nodes.items():
            for producer in node.incoming:
                indegree[node_id] += 1
                dependents[producer].append(node_id)

        running: dict[asyncio.Task[NodeResult], str] = {}

        def schedule(node_id: str) -> None:
            task = asyncio.create_task(self._run_node(node_id, inputs, ctx, values))
            running[task] = node_id

        for node_id in self.plan.order:
            if indegree[node_id] == 0:
                schedule(node_id)

        try:
            while running:
                done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    node_id = running.pop(task)
                    result = task.result()
                    values[node_id] = result.value
                    timings.append(result)
                    for consumer in dependents[node_id]:
                        indegree[consumer] -= 1
                        if indegree[consumer] == 0:
                            schedule(consumer)
        except BaseException:
            for task in running:
                task.cancel()
            raise

        outputs = {
            name: values[node_id][0] for name, node_id in self.plan.sink_nodes.items()
        }
        return outputs, timings

    async def _run_node(
        self,
        node_id: str,
        inputs: dict[str, InputValue],
        ctx: RunContext,
        values: dict[str, list[Any]],
    ) -> NodeResult:
        loop = asyncio.get_running_loop()
        started = loop.time()
        node = self.plan.nodes[node_id]
        incoming = [
            value for producer in node.incoming for value in values[producer]
        ]
        result = await self._execute(node, incoming, inputs, ctx)
        return NodeResult(node_id=node_id, value=result, seconds=loop.time() - started)

    async def _execute(
        self,
        node: Node,
        incoming: list[Any],
        inputs: dict[str, InputValue],
        ctx: RunContext,
    ) -> list[Any]:
        if node.kind == "input":
            return [self._read_input(node, inputs)]
        if node.kind == "ref":
            return incoming
        if node.kind == "sink":
            return incoming
        if node.kind == "tool":
            return [await self._run_tool(node, incoming, ctx)]
        raise ToolExecutionError(f"unknown node kind {node.kind!r}")  # pragma: no cover

    def _read_input(self, node: Node, inputs: dict[str, InputValue]) -> Any:
        name = node.input_name or "input"
        if name not in inputs:
            raise InputError(f"missing required input {name!r}")
        supplied = inputs[name]
        expected = node.output_types[0] if node.output_types else None
        if expected is not None and expected != supplied.type:
            raise InputError(
                f"input {name!r} expects {type_name(expected)} but got "
                f"{type_name(supplied.type)}"
            )
        return normalize(supplied.data, expected or supplied.type)

    async def _run_tool(self, node: Node, incoming: list[Any], ctx: RunContext) -> Any:
        tool = self.registry.get(node.tool_name or "")
        signature = tool.signature
        params: list[Any] = []
        for index, argument in enumerate(node.args):
            if signature.params[index].type is ValueType.NUMBER:
                try:
                    params.append(float(argument))
                except ValueError as exc:
                    raise InputError(
                        f"{node.raw!r}: argument {argument!r} is not a number"
                    ) from exc
            else:
                params.append(argument)
        params.extend(incoming)
        return await tool.run(params, ctx)
