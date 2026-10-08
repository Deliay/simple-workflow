"""Compile a parsed script into a validated, executable :class:`Plan`.

The pipeline is built in stages:

1. **Build**      - turn every element into a node and wire the ``->`` chain
   edges.
2. **Resolve**    - replace ``[var:x]`` references with edges to the node that
   produced ``x`` (forward references are allowed).
3. **Prune**      - drop every node that cannot reach a ``[final]``/``[output]``
   sink (reverse reachability from the sinks).
4. **Validate**   - walk the surviving graph in topological order, checking that
   each tool receives the right number of inputs with the right types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .errors import CompileError, WorkflowError
from .parser import Element, parse_script
from .registry import ToolRegistry
from .signature import Signature
from .types import MaybeType, ValueType, type_matches, type_name

_WILDCARD = None
_TEXT = ValueType.TEXT


@dataclass(slots=True)
class Node:
    id: str
    seq: int
    kind: str  # "tool" | "ref" | "input" | "sink"
    raw: str
    name: str | None = None
    tool_name: str | None = None
    args: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    input_name: str | None = None
    sink_name: str | None = None
    sink_kind: str | None = None
    incoming: list[str] = field(default_factory=list)
    output_types: tuple[MaybeType, ...] = ()
    errors: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return len(self.output_types)


@dataclass(slots=True)
class Plan:
    nodes: dict[str, Node]
    order: list[str]
    outputs: dict[str, MaybeType]
    sink_nodes: dict[str, str]
    input_types: dict[str, MaybeType]

    def to_dict(self) -> dict[str, object]:
        nodes = []
        for node_id in self.order:
            node = self.nodes[node_id]
            nodes.append(
                {
                    "id": node_id,
                    "kind": node.kind,
                    "name": node.name,
                    "tool": node.tool_name,
                    "args": node.args,
                    "refs": node.refs,
                    "inputs": node.incoming,
                    "output_types": [type_name(t) for t in node.output_types],
                }
            )
        return {
            "nodes": nodes,
            "outputs": {name: type_name(t) for name, t in self.outputs.items()},
            "inputs": {name: type_name(t) for name, t in self.input_types.items()},
        }


class _Builder:
    def __init__(
        self, registry: ToolRegistry, input_types: Mapping[str, ValueType | None] | None
    ):
        self.registry = registry
        self.input_types: dict[str, ValueType | None] = dict(input_types or {})
        self.nodes: dict[str, Node] = {}
        self.symbols: dict[str, str] = {}
        self.creation_order: list[str] = []
        self._counter = 0

    # -- construction -----------------------------------------------------
    def _new_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}_{self._counter}"

    def _add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        self.creation_order.append(node.id)
        return node

    def build(self, statements: list[list[Element]]) -> None:
        for statement in statements:
            self._build_statement(statement)
        self._resolve_refs()

    def _build_statement(self, elements: list[Element]) -> None:
        ids: list[str] = []
        for index, element in enumerate(elements):
            if index > 0 and element.kind in ("ref", "input"):
                raise WorkflowError(
                    f"{element.raw!r} cannot consume an upstream value; only a "
                    "tool call or a [final]/[output] sink may appear after '->'"
                )
            node = self._make_node(element)
            ids.append(node.id)
        for producer, consumer in zip(ids, ids[1:], strict=False):
            self.nodes[consumer].incoming.append(producer)

    def _make_node(self, element: Element) -> Node:
        seq = self._counter
        if element.kind == "tool":
            node = Node(
                id=self._new_id("tool"),
                seq=seq,
                kind="tool",
                raw=element.raw,
                name=element.name,
                tool_name=element.tool,
                args=list(element.args),
            )
        elif element.kind == "ref":
            node = Node(
                id=self._new_id("ref"),
                seq=seq,
                kind="ref",
                raw=element.raw,
                name=element.name,
                refs=list(element.refs),
            )
        elif element.kind == "input":
            node = Node(
                id=self._new_id("input"),
                seq=seq,
                kind="input",
                raw=element.raw,
                name=element.name,
                input_name=element.input_name,
            )
        elif element.kind == "sink":
            node = Node(
                id=self._new_id("sink"),
                seq=seq,
                kind="sink",
                raw=element.raw,
                name=element.name,
                sink_name=element.sink_name,
                sink_kind=element.sink_kind,
            )
        else:  # pragma: no cover - parser only emits the four kinds
            raise WorkflowError(f"unknown element kind: {element.kind!r}")

        self._add(node)
        if element.name and element.kind in ("tool", "input"):
            if element.name in self.symbols:
                raise WorkflowError(f"variable {element.name!r} is assigned more than once")
            self.symbols[element.name] = node.id
        return node

    def _resolve_refs(self) -> None:
        for node in self.nodes.values():
            if node.kind != "ref":
                continue
            for ref in node.refs:
                producer = self.symbols.get(ref)
                if producer is None:
                    node.errors.append(f"undefined variable {ref!r}")
                    continue
                node.incoming.append(producer)


def compile_script(
    script: str,
    registry: ToolRegistry,
    input_types: Mapping[str, ValueType | None] | None = None,
) -> Plan:
    statements = parse_script(script)
    builder = _Builder(registry, input_types)
    builder.build(statements)

    sinks = [n.id for n in builder.nodes.values() if n.kind == "sink"]
    if not sinks:
        raise CompileError(["the script has no [final] or [output] node"])

    kept = _reverse_reachable(builder.nodes, sinks)
    order = _toposort(builder.nodes, kept)
    _validate(builder, kept, order)

    outputs, sink_nodes = _name_outputs(builder.nodes, order)
    input_types_used: dict[str, MaybeType] = {}
    for node_id in order:
        node = builder.nodes[node_id]
        if node.kind == "input" and node.input_name:
            input_types_used[node.input_name] = builder.input_types.get(node.input_name)

    return Plan(
        nodes={nid: builder.nodes[nid] for nid in order},
        order=order,
        outputs=outputs,
        sink_nodes=sink_nodes,
        input_types=input_types_used,
    )


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _reverse_reachable(nodes: dict[str, Node], sinks: Iterable[str]) -> set[str]:
    keep: set[str] = set()
    stack = list(sinks)
    while stack:
        node_id = stack.pop()
        if node_id in keep:
            continue
        keep.add(node_id)
        stack.extend(nodes[node_id].incoming)
    return keep


def _toposort(nodes: dict[str, Node], kept: set[str]) -> list[str]:
    indegree = {node_id: 0 for node_id in kept}
    for node_id in kept:
        for producer in nodes[node_id].incoming:
            if producer in kept:
                indegree[node_id] += 1

    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=lambda nid: nodes[nid].seq,
    )
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for consumer in _consumers(nodes, node_id, kept):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
        ready.sort(key=lambda nid: nodes[nid].seq)

    if len(order) != len(kept):
        stuck = sorted(node_id for node_id in kept if node_id not in set(order))
        raise CompileError(
            ["the pipeline contains a cycle involving: " + ", ".join(stuck)]
        )
    return order


def _consumers(nodes: dict[str, Node], node_id: str, kept: set[str]) -> list[str]:
    return [
        other
        for other in kept
        if node_id in nodes[other].incoming
    ]


# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------


def _validate(builder: _Builder, kept: set[str], order: list[str]) -> None:
    """Type-check the surviving graph.

    Two passes are used so that the type of an ``[input]`` that was not
    declared up-front can be *inferred* from the context it is used in.  The
    first pass records those constraints; the second validates against them.
    """

    errors, constraints = _check_pass(builder, kept, order, collect_constraints=True)
    _apply_constraints(builder, constraints)
    errors2, _ = _check_pass(builder, kept, order, collect_constraints=False)
    combined = list(dict.fromkeys([*errors, *errors2]))
    if combined:
        raise CompileError(combined)


def _check_pass(
    builder: _Builder,
    kept: set[str],
    order: list[str],
    *,
    collect_constraints: bool,
) -> tuple[list[str], dict[str, ValueType]]:
    nodes = builder.nodes
    errors: list[str] = []
    constraints: dict[str, ValueType] = {}

    for node_id in order:
        node = nodes[node_id]
        errors.extend(f"{node.raw!r}: {message}" for message in node.errors)

        # Flattened (producer node, output slot, type) triples, in edge order.
        incoming: list[tuple[str, int, MaybeType]] = [
            (producer, slot, value_type)
            for producer in node.incoming
            if producer in kept
            for slot, value_type in enumerate(nodes[producer].output_types)
        ]

        if node.kind == "input":
            node.output_types = (builder.input_types.get(node.input_name or "input"),)
        elif node.kind == "ref":
            node.output_types = tuple(value_type for _, _, value_type in incoming)
        elif node.kind == "tool":
            node.output_types = _check_tool(
                builder,
                node,
                incoming,
                errors,
                constraints if collect_constraints else None,
            )
        elif node.kind == "sink":
            if len(incoming) != 1:
                errors.append(
                    f"{node.raw!r}: a sink expects exactly one value, got {len(incoming)}"
                )
                node.output_types = (_WILDCARD,)
            else:
                node.output_types = (incoming[0][2],)

    return errors, constraints


def _check_tool(
    builder: _Builder,
    node: Node,
    incoming: list[tuple[str, int, MaybeType]],
    errors: list[str],
    constraints: dict[str, ValueType] | None,
) -> tuple[MaybeType, ...]:
    try:
        tool = builder.registry.get(node.tool_name or "")
    except WorkflowError as exc:
        errors.append(f"{node.raw!r}: {exc}")
        return (_WILDCARD,)

    signature: Signature = tool.signature
    literals = len(node.args)
    if literals > len(signature.params):
        errors.append(
            f"{node.raw!r}: tool {tool.name!r} takes at most "
            f"{len(signature.params)} literal argument(s), got {literals}"
        )
        return (signature.output,)

    for param in signature.params[:literals]:
        if param.type not in (_TEXT, ValueType.NUMBER):
            errors.append(
                f"{node.raw!r}: literal argument cannot fill {type_name(param.type)} "
                f"parameter {param.name!r} of tool {tool.name!r}"
            )

    # Expected (type, name) for each upstream value.
    expected: list[tuple[MaybeType, str]] = [
        (param.type, param.name) for param in signature.params[literals:]
    ]
    minimum = len(expected)
    if signature.variadic is not None:
        expected.extend(
            [(signature.variadic.type, signature.variadic.name)] * max(0, len(incoming) - minimum)
        )
    elif len(incoming) < minimum:
        errors.append(
            f"{node.raw!r}: tool {tool.name!r} expects at least {minimum} "
            f"upstream value(s) but received {len(incoming)}"
        )
    elif len(incoming) > minimum:
        errors.append(
            f"{node.raw!r}: tool {tool.name!r} expects exactly {minimum} "
            f"upstream value(s) but received {len(incoming)}"
        )

    for index, (producer, slot, actual) in enumerate(incoming):
        if index >= len(expected):
            break
        expected_type, param_name = expected[index]
        if not type_matches(expected_type, actual):
            errors.append(
                f"{node.raw!r}: argument {param_name!r} of tool {tool.name!r} expects "
                f"{type_name(expected_type)} but the upstream output is {type_name(actual)}"
            )
        elif constraints is not None and actual is None and expected_type is not None:
            _constrain(builder.nodes, producer, slot, expected_type, constraints, errors, set())

    return (signature.output,)


def _constrain(
    nodes: dict[str, Node],
    node_id: str,
    slot: int,
    value_type: ValueType,
    constraints: dict[str, ValueType],
    errors: list[str],
    seen: set[str],
) -> None:
    if node_id in seen:
        return
    seen.add(node_id)
    node = nodes[node_id]
    if node.kind == "input":
        name = node.input_name or "input"
        previous = constraints.get(name)
        if previous is None:
            constraints[name] = value_type
        elif previous is not value_type:
            errors.append(
                f"input {name!r} is used as both {type_name(previous)} and "
                f"{type_name(value_type)}"
            )
        return
    if node.kind == "ref":
        index = slot
        for producer in node.incoming:
            width = len(nodes[producer].output_types)
            if index < width:
                _constrain(nodes, producer, index, value_type, constraints, errors, seen)
                return
            index -= width
        return
    # A tool's output type is already concrete; nothing to infer.


def _apply_constraints(builder: _Builder, constraints: dict[str, ValueType]) -> None:
    for name, value_type in constraints.items():
        if builder.input_types.get(name) is None:
            builder.input_types[name] = value_type


def _name_outputs(
    nodes: dict[str, Node], order: list[str]
) -> tuple[dict[str, MaybeType], dict[str, str]]:
    outputs: dict[str, MaybeType] = {}
    sink_nodes: dict[str, str] = {}
    for node_id in order:
        node = nodes[node_id]
        if node.kind != "sink":
            continue
        base = node.sink_name or node.sink_kind or "output"
        name = base
        suffix = 2
        while name in outputs:
            name = f"{base}_{suffix}"
            suffix += 1
        outputs[name] = node.output_types[0] if node.output_types else _WILDCARD
        sink_nodes[name] = node_id
    return outputs, sink_nodes
