"""Execution engine tests."""

from __future__ import annotations

import asyncio

import pytest

from simple_workflow import Executor, InputValue, ValueType, compile_script

from .conftest import LONG_SCRIPT, SHORT_SCRIPT, make_zip


def _run(script: str, registry, inputs=None):  # noqa: ANN001
    plan = compile_script(script, registry)
    executor = Executor(plan, registry)
    return asyncio.run(executor.run(inputs or {}))


def test_long_pipeline_result(registry) -> None:  # noqa: ANN001
    outputs, _ = _run(LONG_SCRIPT, registry)
    assert outputs["final"] == b"mix:VIRIDIS|O|I"


def test_short_pipeline_result(registry) -> None:  # noqa: ANN001
    outputs, _ = _run(SHORT_SCRIPT, registry)
    assert outputs["final"] == b"mix:VIRIDIS|O|I"


def test_shared_nodes_execute_once(registry) -> None:  # noqa: ANN001
    _run(LONG_SCRIPT, registry)
    msst = registry.get("msst")
    # band-roformer, karaoke-roformer and dereverb-vocal
    assert len(msst.calls) == 3
    rvc = registry.get("rvc")
    assert len(rvc.calls) == 1


def test_input_node(registry) -> None:  # noqa: ANN001
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    archive = make_zip(**{"vocals.wav": b"hello"})
    outputs, _ = _run(
        script,
        registry,
        {"data": InputValue(ValueType.BINARY, archive)},
    )
    assert outputs["final"] == b"hello"


def test_unzip_matches_member_by_stem(registry) -> None:  # noqa: ANN001
    # The pipeline asks for "noreverb" while the archive member is "noreverb.wav".
    script = "[input:data] -> out[unzip:noreverb]\n[var:out] -> [final]"
    archive = make_zip(**{"noreverb.wav": b"NR"})
    outputs, _ = _run(script, registry, {"data": InputValue(ValueType.BINARY, archive)})
    assert outputs["final"] == b"NR"


def test_missing_input_raises(registry) -> None:  # noqa: ANN001
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    from simple_workflow.errors import InputError

    with pytest.raises(InputError, match="missing required input"):
        _run(script, registry)


def test_input_type_mismatch_at_runtime(registry) -> None:  # noqa: ANN001
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    from simple_workflow.errors import InputError

    with pytest.raises(InputError, match="expects binary"):
        _run(script, registry, {"data": InputValue(ValueType.TEXT, "nope")})
