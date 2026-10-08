"""API tool request-shaping tests (endpoint resolution and form fields)."""

from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pytest

from simple_workflow import Param, Signature, ValueType
from simple_workflow.tools.api import ApiTool, make_api_tools
from simple_workflow.tools.base import RunContext

from .test_audio import make_wav

TEXT = ValueType.TEXT
BINARY = ValueType.BINARY

MONO_WAV = make_wav(np.zeros((100, 1), dtype=np.float32))


def msst_tool() -> ApiTool:
    return next(tool for tool in make_api_tools() if tool.name == "msst")


def test_msst_uses_named_fields_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(200, content=b"PK-zip-bytes")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("simple_workflow.tools.api.httpx.AsyncClient", factory)

    tool = msst_tool()
    ctx = RunContext(endpoints={"msst": "http://host/api/msst/inference"})
    result = asyncio.run(tool.run(["model.ckpt", MONO_WAV], ctx))

    assert result == b"PK-zip-bytes"
    assert captured["url"] == "http://host/api/msst/inference"
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b'name="model"' in body
    assert b"model.ckpt" in body
    assert b'name="audio"' in body
    assert b'name="response_format"' in body
    assert b"zip" in body


def test_rvc_uses_named_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        return httpx.Response(200, content=b"WAV")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("simple_workflow.tools.api.httpx.AsyncClient", factory)

    tool = next(tool for tool in make_api_tools() if tool.name == "rvc")
    ctx = RunContext(endpoints={"rvc": "http://host/infer"})
    result = asyncio.run(tool.run(["voice.pth", MONO_WAV], ctx))

    assert result == b"WAV"
    assert captured["url"] == "http://host/infer"
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b'name="model"' in body
    assert b"voice.pth" in body
    assert b'name="audio"' in body
    assert b'name="output_format"' in body
    assert b"wav" in body


def test_api_key_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("simple_workflow.tools.api.httpx.AsyncClient", factory)
    tool = ApiTool(
        "rvc", Signature(params=(Param("model", TEXT),), output=BINARY), api_key="static"
    )
    ctx = RunContext(endpoints={"rvc": "http://host/infer"}, api_keys={"rvc": "from-request"})
    asyncio.run(tool.run(["m"], ctx))
    assert captured["api_key"] == "from-request"


def test_bv_get_path_template(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"AUDIO-BYTES")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("simple_workflow.tools.api.httpx.AsyncClient", factory)

    tool = next(tool for tool in make_api_tools() if tool.name == "bv")
    template = "http://host/api/b/video/{bvid}/download?kind=audio"
    result = asyncio.run(tool.run(["BV1TqaR67E7f"], RunContext(endpoints={"bv": template})))

    assert result == b"AUDIO-BYTES"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://host/api/b/video/BV1TqaR67E7f/download?kind=audio"


def test_unknown_placeholder_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from simple_workflow.errors import ToolExecutionError

    tool = next(tool for tool in make_api_tools() if tool.name == "bv")
    ctx = RunContext(endpoints={"bv": "http://host/video/{nope}/x"})
    with pytest.raises(ToolExecutionError, match="placeholder"):
        asyncio.run(tool.run(["BV1"], ctx))


def test_endpoint_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOOLS_MSST_ENDPOINT", "http://env-host/api/msst/inference")
    tool = msst_tool()
    assert tool._resolve_url(RunContext()) == "http://env-host/api/msst/inference"


def test_missing_endpoint_raises() -> None:
    from simple_workflow.errors import ToolExecutionError

    tool = ApiTool("ghost", Signature(params=(Param("x", TEXT),), output=BINARY))
    with pytest.raises(ToolExecutionError, match="ghost"):
        asyncio.run(tool.run(["x"], RunContext()))


def test_generic_tools_use_positional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(200, content=b"ok")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(*args, transport=transport, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("simple_workflow.tools.api.httpx.AsyncClient", factory)
    tool = ApiTool("bv", Signature(params=(Param("id", TEXT),), output=BINARY))
    asyncio.run(tool.run(["BV1"], RunContext(endpoints={"bv": "http://host/bv"})))
    # No binary params -> urlencoded form body.
    assert captured["body"] == b"a0=BV1"
