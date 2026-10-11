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


def test_zip_packs_tagged_variables_by_filename(registry) -> None:  # noqa: ANN001
    import io
    import zipfile

    script = (
        "a[bv:test]{file=测试.mp4}\n"
        "b[bv:test2]{file=测试2.mp4}\n"
        "[var:a, var:b] -> [zip] -> [final]"
    )
    outputs, _ = _run(script, registry)
    with zipfile.ZipFile(io.BytesIO(outputs["final"])) as archive:
        assert archive.namelist() == ["测试.mp4", "测试2.mp4"]
        assert archive.read("测试.mp4") == b"raw:test"
        assert archive.read("测试2.mp4") == b"raw:test2"


def test_zip_without_a_file_tag_raises(registry) -> None:  # noqa: ANN001
    from simple_workflow.errors import ToolExecutionError

    script = "a[bv:test]\nb[bv:test2]\n[var:a, var:b] -> [zip] -> [final]"
    with pytest.raises(ToolExecutionError, match=r"file=\.\.\."):
        _run(script, registry)


def test_inline_bundle_feeds_zip(registry) -> None:  # noqa: ANN001
    import io
    import zipfile

    script = "[[bv:1]{file=1.mp4}, [bv:2]{file=2.mp3}] -> [zip] -> [final]"
    outputs, _ = _run(script, registry)
    with zipfile.ZipFile(io.BytesIO(outputs["final"])) as archive:
        assert archive.namelist() == ["1.mp4", "2.mp3"]
        assert archive.read("1.mp4") == b"raw:1"
        assert archive.read("2.mp3") == b"raw:2"


def test_inline_bundle_can_mix_refs_and_producers(registry) -> None:  # noqa: ANN001
    import io
    import zipfile

    script = (
        "old[bv:old]{file=old.mp4}\n"
        "[[bv:new]{file=new.mp4}, [var:old]] -> [zip] -> [final]"
    )
    outputs, _ = _run(script, registry)
    with zipfile.ZipFile(io.BytesIO(outputs["final"])) as archive:
        assert archive.namelist() == ["new.mp4", "old.mp4"]


def test_bundle_cannot_consume_upstream(registry) -> None:  # noqa: ANN001
    from simple_workflow.errors import WorkflowError

    with pytest.raises(WorkflowError, match="cannot consume"):
        compile_script("[bv:x] -> [[bv:y], [bv:z]] -> [final]", registry)


def test_mix_vocal_tag_reaches_the_tool() -> None:
    from simple_workflow import Param, Signature, Tool, ToolRegistry

    seen: dict[str, object] = {}

    class Spy(Tool):
        def __init__(self) -> None:
            super().__init__(
                "audio",
                Signature(
                    params=(Param("op", ValueType.TEXT),),
                    variadic=Param("inputs", ValueType.BINARY),
                    output=ValueType.BINARY,
                ),
            )

        async def run(self, params, ctx):  # noqa: ANN001, ANN201
            seen["roles"] = [getattr(p, "role", None) for p in params[1:]]
            return b"ok"

    spy_registry = ToolRegistry()
    spy_registry.register(Spy())
    script = "v[input:vocal]{mix=vocal}\n[var:v] -> [audio:mix] -> [final]"
    _run(script, spy_registry, {"vocal": InputValue(ValueType.BINARY, b"\x00")})
    assert seen["roles"] == ["vocal"]
