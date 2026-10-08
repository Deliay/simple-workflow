"""Tool signatures: the static contract every tool exposes."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ValueType


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    type: ValueType


@dataclass(frozen=True, slots=True)
class Signature:
    """A tool's input/output contract.

    ``params`` are the fixed leading parameters.  ``variadic`` optionally
    describes a trailing, unbounded parameter (e.g. ``audio`` accepts any number
    of binary streams).  Literal arguments written in the script fill the
    leading params; values arriving over an edge fill the remaining ones.
    """

    params: tuple[Param, ...] = ()
    variadic: Param | None = None
    output: ValueType = ValueType.BINARY

    def describe(self) -> str:
        args = [f"{p.name}: {p.type}" for p in self.params]
        if self.variadic is not None:
            args.append(f"...{self.variadic.name}: {self.variadic.type}")
        return f"({', '.join(args)}) -> {self.output}"
