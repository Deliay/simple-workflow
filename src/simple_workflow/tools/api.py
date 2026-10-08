"""HTTP-backed ("API class") tools.

Two request styles are supported:

**POST form** (default) — parameters are written to a ``multipart/form-data``
body.  By default the fields are named ``a0``, ``a1``, ... but a tool may map
them to concrete names via ``field_names`` (e.g. the MSST API expects ``model``
and ``audio``).  Text/number parameters become plain fields, binary parameters
become file parts.  ``default_fields`` are always added (e.g.
``response_format=zip``).

**GET with a URL template** — the resolved endpoint may contain ``{name}``
placeholders where ``name`` is a signature parameter.  The placeholder is
substituted with the (percent-encoded) argument and that argument is *not* sent
again as a query/body parameter.  Parameters left over from the template are
sent as query parameters.

Examples::

    # MSST
    TOOLS_MSST_ENDPOINT=http://xxx.xxx/api/msst/inference
    # -> POST, fields: model, audio, response_format=zip

    # Applio RVC
    TOOLS_RVC_ENDPOINT=http://xxx.xxx:7870/infer
    # -> POST, fields: model, audio, output_format=wav

    # Bilibili download (bv)
    TOOLS_BV_ENDPOINT=http://xxx.xxx/api/b/video/{bvid}/download?kind=audio
    # -> GET http://xxx.xxx/api/b/video/BV1.../download?kind=audio

The endpoint is a *full URL* resolved per tool, in this order:

1. ``RunContext.endpoints[tool_name]`` (per-request override),
2. the tool's own ``endpoint`` (if constructed with one),
3. the ``TOOLS_<NAME>_ENDPOINT`` environment variable.

An optional API key (``RunContext.api_keys`` / ``TOOLS_<NAME>_API_KEY``) is sent
as ``X-API-Key`` (configurable via ``api_key_header``).

The response body is the return value: ``number`` -> float, ``text`` -> str,
``binary`` -> raw bytes.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import httpx

from ..errors import ToolExecutionError
from ..media import ensure_wav
from ..signature import Param, Signature
from ..types import ValueType
from .base import RunContext, Tool

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ApiTool(Tool):
    def __init__(
        self,
        name: str,
        signature: Signature,
        *,
        endpoint: str | None = None,
        method: str = "POST",
        field_names: Sequence[str] | None = None,
        default_fields: dict[str, str] | None = None,
        transcode_audio: bool = False,
        transcode_channels: int | None = None,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",
    ) -> None:
        super().__init__(name, signature)
        self.endpoint = endpoint
        self.method = method.upper()
        self.field_names = tuple(field_names) if field_names else None
        self.default_fields = dict(default_fields or {})
        self.transcode_audio = transcode_audio
        self.transcode_channels = transcode_channels
        self.api_key = api_key
        self.api_key_header = api_key_header

    @property
    def env_var(self) -> str:
        return f"TOOLS_{self.name.upper()}_ENDPOINT"

    @property
    def api_key_env_var(self) -> str:
        return f"TOOLS_{self.name.upper()}_API_KEY"

    def configured_endpoint(self, ctx: RunContext | None = None) -> str | None:
        if ctx is not None and (resolved := ctx.endpoint_for(self.name)):
            return resolved
        return self.endpoint or os.environ.get(self.env_var)

    def _resolve_url(self, ctx: RunContext) -> str:
        url = ctx.endpoint_for(self.name) or self.endpoint
        if not url:
            raise ToolExecutionError(
                f"API tool {self.name!r} has no endpoint; set {self.env_var} "
                f"or pass tool_endpoints on the request"
            )
        return url

    def _resolve_api_key(self, ctx: RunContext) -> str | None:
        return ctx.api_key_for(self.name) or self.api_key

    def _param_name(self, index: int) -> str:
        if index < len(self.signature.params):
            return self.signature.params[index].name
        return f"a{index}"

    def _form_field(self, index: int) -> str:
        if self.field_names is not None and index < len(self.field_names):
            return self.field_names[index]
        return f"a{index}"

    async def run(self, params: list[Any], ctx: RunContext) -> Any:  # noqa: ANN401
        by_name = {
            self._param_name(index): value for index, value in enumerate(params)
        }
        url, consumed = _substitute(self.name, self._resolve_url(ctx), by_name)
        remaining = [
            (index, value)
            for index, value in enumerate(params)
            if self._param_name(index) not in consumed
        ]

        headers: dict[str, str] = {}
        if api_key := self._resolve_api_key(ctx):
            headers[self.api_key_header] = api_key

        try:
            async with httpx.AsyncClient(timeout=ctx.timeout) as client:
                if self.method == "GET":
                    query: dict[str, str] = {}
                    for index, value in remaining:
                        if isinstance(value, (bytes, bytearray, memoryview)):
                            raise ToolExecutionError(
                                f"API tool {self.name!r}: binary argument "
                                f"{self._param_name(index)!r} cannot be sent as a GET query"
                            )
                        query[self._param_name(index)] = _format_scalar(value)
                    response = await client.get(url, params=query or None, headers=headers or None)
                else:
                    data: dict[str, str] = dict(self.default_fields)
                    files: dict[str, tuple[str, bytes, str]] = {}
                    for index, value in remaining:
                        key = self._form_field(index)
                        if isinstance(value, (bytes, bytearray, memoryview)):
                            payload = bytes(value)
                            if self.transcode_audio:
                                payload = ensure_wav(
                                    payload, channels=self.transcode_channels
                                )
                            filename = f"{key}.wav" if payload[:4] == b"RIFF" else f"{key}.bin"
                            files[key] = (filename, payload, "application/octet-stream")
                        else:
                            data[key] = _format_scalar(value)
                    response = await client.post(
                        url, data=data, files=files or None, headers=headers or None
                    )
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"API tool {self.name!r} request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ToolExecutionError(
                f"API tool {self.name!r} returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        return _parse_response(self.name, self.signature.output, response)


def _substitute(
    tool_name: str, url: str, by_name: dict[str, Any]
) -> tuple[str, set[str]]:
    consumed: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in by_name:
            raise ToolExecutionError(
                f"API tool {tool_name!r}: endpoint placeholder {{{name}}} does not "
                f"match any parameter (available: {', '.join(by_name) or '<none>'})"
            )
        consumed.add(name)
        return quote(str(by_name[name]), safe="")

    return _PLACEHOLDER.sub(replace, url), consumed


def _format_scalar(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_response(name: str, output: ValueType, response: httpx.Response) -> Any:  # noqa: ANN401
    if output is ValueType.BINARY:
        return response.content
    if output is ValueType.TEXT:
        return response.text
    if output is ValueType.NUMBER:
        text = response.text.strip()
        try:
            return float(text)
        except ValueError as exc:
            raise ToolExecutionError(
                f"API tool {name!r} expected a number but received {text[:200]!r}"
            ) from exc
    raise ToolExecutionError(f"unsupported output type {output!r}")  # pragma: no cover


def make_api_tools() -> list[ApiTool]:
    """The three API tools described by the pipeline spec.

    Endpoints are not baked in; each is resolved from ``TOOLS_<NAME>_ENDPOINT``
    at run time (or from the request's ``tool_endpoints``).
    """
    text = ValueType.TEXT
    binary = ValueType.BINARY
    return [
        # GET http://.../api/b/video/{bvid}/download?kind=audio
        ApiTool(
            "bv",
            Signature(params=(Param("bvid", text),), output=binary),
            method="GET",
        ),
        # POST /api/msst/inference with model + audio (+ response_format=zip)
        ApiTool(
            "msst",
            Signature(params=(Param("model", text), Param("audio", binary)), output=binary),
            field_names=("model", "audio"),
            default_fields={"response_format": "zip"},
            transcode_audio=True,
            transcode_channels=1,
        ),
        # POST /infer with model + audio (+ output_format=wav)
        ApiTool(
            "rvc",
            Signature(params=(Param("model", text), Param("audio", binary)), output=binary),
            field_names=("model", "audio"),
            default_fields={"output_format": "wav"},
            transcode_audio=True,
        ),
    ]
