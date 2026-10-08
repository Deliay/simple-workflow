"""The three value types understood by the engine.

The whole point of the system is that every edge in the graph carries a value
whose type is known *before* execution:

* ``TEXT``   -> :class:`str`
* ``NUMBER`` -> :class:`float`
* ``BINARY`` -> :class:`bytes`

``None`` is used as a wildcard ("any") type.  Wildcards appear when the type of
an ``[input]`` node is not declared up-front; they are resolved at runtime and
type-checked when the concrete value arrives.
"""

from __future__ import annotations

import base64
import enum
import math
from typing import Any, TypeAlias

from .errors import InputError, WorkflowError


class ValueType(enum.StrEnum):
    TEXT = "text"
    NUMBER = "number"
    BINARY = "binary"


#: ``None`` means "any" / "unknown at compile time".
MaybeType: TypeAlias = ValueType | None


def type_name(t: MaybeType) -> str:
    return t.value if t is not None else "any"


def type_matches(expected: MaybeType, actual: MaybeType) -> bool:
    """A wildcard matches anything, otherwise types must be identical."""
    if expected is None or actual is None:
        return True
    return expected is actual or expected == actual


def infer_type(value: Any) -> ValueType:
    if isinstance(value, bool):
        raise WorkflowError("booleans are not a supported value type")
    if isinstance(value, (int, float)):
        return ValueType.NUMBER
    if isinstance(value, str):
        return ValueType.TEXT
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ValueType.BINARY
    raise WorkflowError(f"cannot infer a value type for {type(value).__name__!r}")


def normalize(value: Any, expected: ValueType) -> Any:
    """Coerce/validate a runtime value against an expected concrete type."""
    actual = infer_type(value)
    if not type_matches(expected, actual):
        raise InputError(
            f"expected {type_name(expected)} value, got {type_name(actual)}"
        )
    if expected is ValueType.NUMBER:
        number = float(value)
        if not math.isfinite(number):
            raise InputError("number inputs must be finite")
        return number
    if expected is ValueType.TEXT:
        return str(value)
    return bytes(value)


# ---------------------------------------------------------------------------
# JSON (de)serialization helpers used by the HTTP layer.
# ---------------------------------------------------------------------------


def encode_value(value: Any, value_type: ValueType) -> dict[str, Any]:
    if value_type is ValueType.BINARY:
        return {"type": "binary", "data": base64.b64encode(value).decode("ascii")}
    if value_type is ValueType.NUMBER:
        return {"type": "number", "data": float(value)}
    return {"type": "text", "data": str(value)}


def decode_value(value_type: ValueType, data: Any) -> Any:
    if value_type is ValueType.BINARY:
        if not isinstance(data, str):
            raise InputError("binary input data must be a base64 string")
        try:
            return base64.b64decode(data, validate=True)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise InputError(f"binary input is not valid base64: {exc}") from exc
    if value_type is ValueType.NUMBER:
        try:
            return float(data)
        except (TypeError, ValueError) as exc:
            raise InputError(f"number input is not numeric: {data!r}") from exc
    if value_type is ValueType.TEXT:
        if not isinstance(data, str):
            raise InputError("text input data must be a string")
        return data
    raise InputError(f"unsupported value type: {value_type!r}")
