"""Parser for the pipeline DSL.

A script is a sequence of statements.  Each statement is a *chain* of elements
joined by ``->``.  Physical lines may be continued either by starting the next
line with ``->`` or by ending the current line with ``->``.

Element shapes recognised inside a chain::

    [bv:BV1TqaR67E7f]                 # anonymous tool call
    band-roformer[msst:model.ckpt]    # tool call whose output is bound to a var
    [var:band-roformer]               # reference to a previously bound variable
    [var:a, b, c]                     # bundle of references (multi-input)
    [input] / [input:name]            # read one of the API inputs
    [final] / [output] / [output:name]# terminal sink (workflow result)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import ScriptSyntaxError

_ANON = re.compile(r"^\[\s*(.*?)\s*\]$", re.DOTALL)
_NAMED = re.compile(r"^([^\[\]]+?)\s*\[(.*)\]$", re.DOTALL)

#: Bracketed heads that are *not* tool calls.
RESERVED = {"var", "input", "final", "output"}


@dataclass(slots=True)
class Element:
    kind: str  # "tool" | "ref" | "input" | "sink"
    raw: str
    name: str | None = None
    tool: str | None = None
    args: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    input_name: str | None = None
    sink_name: str | None = None
    sink_kind: str | None = None


def parse_script(text: str) -> list[list[Element]]:
    """Parse a script into a list of statements (each a list of elements)."""
    statements: list[list[Element]] = []
    for line in _logical_lines(text):
        parts = [part.strip() for part in line.split("->")]
        if len(parts) == 1:
            statements.append([parse_element(parts[0])])
            continue
        if any(part == "" for part in parts):
            raise ScriptSyntaxError(f"empty element in chain: {line!r}")
        statements.append([parse_element(part) for part in parts])
    return statements


def parse_element(text: str) -> Element:
    text = text.strip()
    if not text:
        raise ScriptSyntaxError("empty element")

    match = _ANON.match(text)
    if match:
        return _parse_bracket(match.group(1).strip(), name=None, raw=text)

    match = _NAMED.match(text)
    if match:
        name = match.group(1).strip()
        if not name:
            raise ScriptSyntaxError(f"missing node name in {text!r}")
        return _parse_bracket(match.group(2).strip(), name=name, raw=text)

    # Bare identifier: shorthand for a variable reference.
    return Element(kind="ref", raw=text, refs=[text])


def _parse_bracket(inner: str, *, name: str | None, raw: str) -> Element:
    if not inner:
        raise ScriptSyntaxError(f"empty element: {raw!r}")
    head, _, rest = inner.partition(":")
    head = head.strip()
    rest = rest.strip()

    if head == "var":
        refs = [_strip_var(item) for item in _split_commas(rest)]
        if not refs:
            raise ScriptSyntaxError(f"empty [var] reference: {raw!r}")
        return Element(kind="ref", raw=raw, name=name, refs=refs)
    if head == "input":
        return Element(kind="input", raw=raw, name=name, input_name=rest or "input")
    if head in ("final", "output"):
        return Element(
            kind="sink",
            raw=raw,
            name=name,
            sink_kind=head,
            sink_name=rest or None,
        )

    # Anything else is a tool call.
    return Element(kind="tool", raw=raw, name=name, tool=head, args=_split_commas(rest))


def _strip_var(item: str) -> str:
    item = item.strip()
    if item.startswith("var:"):
        item = item[4:].strip()
    return item


def _split_commas(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            if current:
                lines.append(current)
                current = ""
            continue
        if not current:
            current = stripped
        elif current.endswith("->") or stripped.startswith("->"):
            current = f"{current} {stripped}"
        else:
            lines.append(current)
            current = stripped
    if current:
        lines.append(current)
    return lines
