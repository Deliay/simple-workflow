"""Parser tests."""

from __future__ import annotations

import pytest

from simple_workflow import parse_script
from simple_workflow.errors import ScriptSyntaxError

from .conftest import LONG_SCRIPT, SHORT_SCRIPT


def test_long_script_statements() -> None:
    statements = parse_script(LONG_SCRIPT)
    assert len(statements) == 13
    first = statements[0]
    assert [element.kind for element in first] == ["tool", "tool"]
    assert first[0].tool == "bv"
    assert first[0].args == ["BV1TqaR67E7f"]
    assert first[1].name == "band-roformer"
    assert first[1].tool == "msst"


def test_short_script_joins_continuations() -> None:
    statements = parse_script(SHORT_SCRIPT)
    # First chain: bv -> msst -> unzip -> msst -> unzip -> audio -> msst ->
    #              unzip -> rvc -> audio  (10 elements).
    assert statements[0][0].tool == "bv"
    assert len(statements[0]) == 10
    assert statements[0][1].tool == "msst"
    assert statements[0][2].tool == "unzip"


def test_bundle_references() -> None:
    statements = parse_script(LONG_SCRIPT)
    bundle = next(
        element
        for statement in statements
        for element in statement
        if element.kind == "ref" and len(element.refs) > 1
    )
    assert bundle.refs == ["viridis-vocal", "bands", "harmony"]


def test_sink_and_input() -> None:
    assert parse_script("[input:data] -> [output:result]")[0][0].input_name == "data"
    assert parse_script("[input:data] -> [output:result]")[0][1].sink_kind == "output"


def test_empty_chain_element_rejected() -> None:
    with pytest.raises(ScriptSyntaxError):
        parse_script("[bv:x] -> -> [final]")
