"""Simple Workflow: a strongly-typed, minimal pipeline engine."""

from __future__ import annotations

from .compiler import Plan, compile_script
from .engine import Executor, InputValue
from .errors import (
    CompileError,
    InputError,
    ScriptSyntaxError,
    ToolExecutionError,
    WorkflowError,
)
from .parser import parse_script
from .registry import ToolRegistry, build_registry
from .signature import Param, Signature
from .tools import RunContext, Tool
from .types import ValueType, decode_value, encode_value, infer_type, normalize

__version__ = "0.1.0"

__all__ = [
    "CompileError",
    "Executor",
    "InputError",
    "InputValue",
    "Param",
    "Plan",
    "RunContext",
    "ScriptSyntaxError",
    "Signature",
    "Tool",
    "ToolExecutionError",
    "ToolRegistry",
    "ValueType",
    "WorkflowError",
    "__version__",
    "build_registry",
    "compile_script",
    "decode_value",
    "encode_value",
    "infer_type",
    "normalize",
    "parse_script",
]
