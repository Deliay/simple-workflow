"""HTTP API for the workflow engine."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .cache import CacheStore, build_cache, cleanup_interval
from .compiler import Plan, compile_script
from .engine import Executor, InputValue
from .errors import CompileError, ScriptSyntaxError, WorkflowError
from .registry import ToolRegistry, build_registry
from .tools.api import ApiTool
from .tools.base import RunContext
from .types import (
    ValueType,
    decode_value,
    encode_value,
    infer_type,
)

logger = logging.getLogger(__name__)

TypeName = Literal["text", "number", "binary"]


class ValueSpec(BaseModel):
    type: TypeName
    data: Any = Field(..., description="text/number value, or base64 for binary")


class CompileRequest(BaseModel):
    script: str
    input_types: dict[str, TypeName] | None = None


class CompileResponse(BaseModel):
    plan: dict[str, Any]


class RunRequest(BaseModel):
    script: str
    inputs: dict[str, ValueSpec] = Field(default_factory=dict)
    input_types: dict[str, TypeName] | None = None
    tool_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Per-tool endpoint overrides, e.g. {'msst': 'http://host/api/msst/inference'}",
    )
    tool_api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Per-tool API keys, sent as X-API-Key, e.g. {'rvc': 'secret'}",
    )
    timeout: float | None = Field(default=None, gt=0, le=3600)


class TimingOut(BaseModel):
    node: str
    tool: str | None = None
    seconds: float
    cached: bool = False


class RunResponse(BaseModel):
    outputs: dict[str, ValueSpec]
    elapsed: float
    timings: list[TimingOut]


class CacheStatusOut(BaseModel):
    enabled: bool
    cleared: int


def _to_value_types(spec: Mapping[str, str] | None) -> dict[str, ValueType] | None:
    if spec is None:
        return None
    return {name: ValueType(value) for name, value in spec.items()}


def _to_inputs(inputs: dict[str, ValueSpec]) -> dict[str, InputValue]:
    result: dict[str, InputValue] = {}
    for name, spec in inputs.items():
        value_type = ValueType(spec.type)
        result[name] = InputValue(value_type, decode_value(value_type, spec.data))
    return result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.registry = build_registry()
    app.state.cache = build_cache()
    cleanup_task: asyncio.Task[None] | None = None
    if app.state.cache.enabled:
        cleanup_task = asyncio.create_task(_cache_cleanup_loop(app.state.cache))
    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task


async def _cache_cleanup_loop(cache: CacheStore) -> None:
    """Periodically evict entries past their TTL."""
    interval = cleanup_interval()
    while True:
        await asyncio.sleep(interval)
        try:
            removed = await cache.purge_expired()
        except Exception:  # noqa: BLE001 - the loop must survive any error
            logger.exception("cache cleanup failed")
            continue
        if removed:
            logger.info("cache cleanup removed %d expired entrie(s)", removed)


app = FastAPI(
    title="Simple Workflow",
    version="0.1.0",
    summary="A strongly-typed, minimal pipeline engine.",
    lifespan=lifespan,
)


def _registry(request: Request) -> ToolRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def _cache(request: Request) -> CacheStore:
    return request.app.state.cache  # type: ignore[no-any-return]


def _compile(request: Request, body: CompileRequest) -> Plan:
    try:
        return compile_script(
            body.script, _registry(request), _to_value_types(body.input_types)
        )
    except (ScriptSyntaxError, CompileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/tools")
async def list_tools(request: Request) -> dict[str, Any]:
    registry = _registry(request)
    return {
        "tools": [
            {
                "name": tool.name,
                "signature": tool.signature.describe(),
                "endpoint": _tool_endpoint(tool),
            }
            for tool in registry
        ]
    }


def _tool_endpoint(tool: object) -> str | None:
    if isinstance(tool, ApiTool):
        return tool.configured_endpoint() or tool.env_var
    return None


@app.post("/v1/compile", response_model=CompileResponse)
async def compile_endpoint(request: Request, body: CompileRequest) -> CompileResponse:
    plan = _compile(request, body)
    return CompileResponse(plan=plan.to_dict())


@app.post("/v1/run", response_model=RunResponse)
async def run_endpoint(request: Request, body: RunRequest) -> RunResponse:
    plan = _compile(request, CompileRequest(script=body.script, input_types=body.input_types))
    registry = _registry(request)

    default_timeout = float(os.environ.get("SW_TOOL_TIMEOUT", "300"))
    ctx = RunContext(
        endpoints=body.tool_endpoints,
        api_keys=body.tool_api_keys,
        timeout=body.timeout or default_timeout,
    )

    executor = Executor(plan, registry, cache=_cache(request))
    started = time.perf_counter()
    try:
        outputs, timings = await executor.run(_to_inputs(body.inputs), ctx)
    except WorkflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed = time.perf_counter() - started

    encoded: dict[str, ValueSpec] = {}
    for name, value in outputs.items():
        value_type = infer_type(value)
        encoded[name] = ValueSpec(**encode_value(value, value_type))

    return RunResponse(
        outputs=encoded,
        elapsed=elapsed,
        timings=[
            TimingOut(
                node=result.node_id,
                tool=plan.nodes[result.node_id].tool_name,
                seconds=result.seconds,
                cached=result.cached,
            )
            for result in timings
        ],
    )


@app.delete("/v1/cache", response_model=CacheStatusOut)
async def clear_cache(request: Request) -> CacheStatusOut:
    """Delete every cached node output."""
    store = _cache(request)
    removed = await store.clear()
    return CacheStatusOut(enabled=store.enabled, cleared=removed)


__all__ = ["app", "ValueSpec", "RunRequest", "CompileRequest", "CacheStatusOut"]
