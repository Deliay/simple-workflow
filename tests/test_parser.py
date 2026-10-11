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


def test_bang_marks_node_as_cache_bypassing() -> None:
    element = parse_script("[!audio:mix] -> [final]")[0][0]
    assert element.kind == "tool"
    assert element.tool == "audio"
    assert element.args == ["mix"]
    assert element.no_cache is True
    # A leading bang on a named node is also recognised.
    named = parse_script("out[!audio:mp3] -> [final]")[0][0]
    assert named.name == "out"
    assert named.no_cache is True


def test_tags_are_parsed_after_the_node() -> None:
    element = parse_script("a[bv:x]{file=yyy,mix=vocal}")[0][0]
    assert element.name == "a"
    assert element.tool == "bv"
    assert element.kind == "tool"
    assert element.tags == {"file": "yyy", "mix": "vocal"}


def test_anonymous_node_can_carry_tags() -> None:
    element = parse_script("[bv:x]{file=zzz}")[0][0]
    assert element.kind == "tool"
    assert element.tags == {"file": "zzz"}


def test_tag_requires_key_value() -> None:
    with pytest.raises(ScriptSyntaxError, match="key=value"):
        parse_script("a[bv:x]{file}")


def test_old_tag_syntax_is_rejected() -> None:
    for script in ("[var:x] -> [file:y]", "[var:x] -> [mix:vocal]"):
        with pytest.raises(ScriptSyntaxError, match="no longer valid"):
            parse_script(script)
