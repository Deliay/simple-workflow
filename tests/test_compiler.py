"""Compiler / type-checker tests."""

from __future__ import annotations

import pytest

from simple_workflow import ValueType, compile_script
from simple_workflow.errors import CompileError, WorkflowError

from .conftest import LONG_SCRIPT, SHORT_SCRIPT, SHORT_SCRIPT_SPEC


def test_long_script_prunes_dead_nodes(registry) -> None:  # noqa: ANN001
    plan = compile_script(LONG_SCRIPT, registry)
    names = {node.name for node in plan.nodes.values()}
    # reverb-viridis-vocal never reaches [final] and must be discarded.
    assert "reverb-viridis-vocal" not in names
    assert "result" in names
    assert "viridis-vocal" in names
    assert list(plan.outputs) == ["final"]
    assert plan.outputs["final"] == ValueType.BINARY


def test_short_and_long_are_equivalent(registry) -> None:  # noqa: ANN001
    long_plan = compile_script(LONG_SCRIPT, registry)
    short_plan = compile_script(SHORT_SCRIPT, registry)
    assert set(long_plan.outputs) == set(short_plan.outputs)
    assert long_plan.outputs == short_plan.outputs


def test_spec_shorthand_reports_unbound_variable(registry) -> None:  # noqa: ANN001
    with pytest.raises(CompileError, match="undefined variable 'viridis-vocal'"):
        compile_script(SHORT_SCRIPT_SPEC, registry)


def test_unknown_tool_is_rejected(registry) -> None:  # noqa: ANN001
    with pytest.raises(CompileError, match="unknown tool"):
        compile_script("[nope:x] -> [final]", registry)


def test_missing_upstream_input_is_rejected(registry) -> None:  # noqa: ANN001
    with pytest.raises(CompileError, match="at least 1"):
        compile_script("only[msst:model] -> [final]", registry)


def test_type_mismatch_is_rejected(registry) -> None:  # noqa: ANN001
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    with pytest.raises(CompileError, match="expects binary"):
        compile_script(script, registry, {"data": ValueType.TEXT})


def test_input_type_is_inferred(registry) -> None:  # noqa: ANN001
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    plan = compile_script(script, registry)
    # `data` feeds unzip's binary archive parameter, so it is inferred binary.
    assert plan.input_types == {"data": ValueType.BINARY}


def test_cycle_is_rejected(registry) -> None:  # noqa: ANN001
    script = """
    [var:y] -> x[msst:m]
    [var:x] -> y[msst:m]
    [var:y] -> [final]
    """
    with pytest.raises(CompileError, match="cycle"):
        compile_script(script, registry)


def test_duplicate_variable_is_rejected(registry) -> None:  # noqa: ANN001
    script = "[bv:x] -> d[unzip:a]\n[bv:y] -> d[unzip:b]\n[var:d] -> [final]"
    with pytest.raises(WorkflowError, match="assigned more than once"):
        compile_script(script, registry)
