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
    name[tool:arg]{file=x.mp4}        # ``{key=value}`` tags on a value producer
    [!audio:mix]                      # a leading ``!`` marks the node as
                                      # cache-bypassing (it and its downstream
                                      # subtree are always re-executed)

Tags are written after the node that produces the value: ``file=NAME`` names it
for ``zip`` and ``mix=ROLE`` assigns its role (e.g. ``vocal``) for ``mix``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import ScriptSyntaxError

_ANON = re.compile(r"^\[\s*(.*?)\s*\]$", re.DOTALL)
_NAMED = re.compile(r"^([^\[\]]+?)\s*\[(.*)\]$", re.DOTALL)
_TAGS = re.compile(r"^(?P<core>.*)\{(?P<tags>[^{}]*)\}\s*$", re.DOTALL)

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
    no_cache: bool = False
    tags: dict[str, str] = field(default_factory=dict)


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

    original = text
    core, tags = _split_tags(text)
    if not core:
        raise ScriptSyntaxError(f"missing element before tags in {original!r}")

    match = _ANON.match(core)
    if match:
        element = _parse_bracket(match.group(1).strip(), name=None, raw=original)
    else:
        match = _NAMED.match(core)
        if match:
            name = match.group(1).strip()
            if not name:
                raise ScriptSyntaxError(f"missing node name in {original!r}")
            element = _parse_bracket(match.group(2).strip(), name=name, raw=original)
        else:
            # Bare identifier: shorthand for a variable reference.
            element = Element(kind="ref", raw=original, refs=[core])
    element.tags = tags
    return element


def _split_tags(text: str) -> tuple[str, dict[str, str]]:
    match = _TAGS.match(text)
    if not match:
        return text, {}
    return match.group("core").strip(), _parse_tags(match.group("tags"))


def _parse_tags(inner: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in inner.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ScriptSyntaxError(f"invalid tag {item!r}; expected key=value")
        if key in tags:
            raise ScriptSyntaxError(f"duplicate tag {key!r}")
        tags[key] = value
    return tags


def _parse_bracket(inner: str, *, name: str | None, raw: str) -> Element:
    no_cache = False
    if inner.startswith("!"):
        no_cache = True
        inner = inner[1:].strip()
    if not inner:
        raise ScriptSyntaxError(f"empty element: {raw!r}")
    head, _, rest = inner.partition(":")
    head = head.strip()
    rest = rest.strip()

    if head == "var":
        refs = [_strip_var(item) for item in _split_commas(rest)]
        if not refs:
            raise ScriptSyntaxError(f"empty [var] reference: {raw!r}")
        return Element(kind="ref", raw=raw, name=name, refs=refs, no_cache=no_cache)
    if head == "input":
        return Element(
            kind="input", raw=raw, name=name, input_name=rest or "input", no_cache=no_cache
        )
    if head in ("final", "output"):
        return Element(
            kind="sink",
            raw=raw,
            name=name,
            sink_kind=head,
            sink_name=rest or None,
            no_cache=no_cache,
        )
    if head in ("file", "mix"):
        raise ScriptSyntaxError(
            f"[{head}:...] is no longer valid; write {{{head}=...}} after the node"
        )

    # Anything else is a tool call.
    return Element(
        kind="tool", raw=raw, name=name, tool=head, args=_split_commas(rest),
        no_cache=no_cache,
    )


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
