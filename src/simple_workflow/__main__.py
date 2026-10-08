"""Command line entry point: ``simple-workflow``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .compiler import compile_script
from .engine import Executor, InputValue
from .errors import WorkflowError
from .registry import build_registry
from .tools.base import RunContext
from .types import ValueType, encode_value


def _read_script(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_kv(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not name or not _:
            raise SystemExit(f"invalid input {item!r}; expected name=value")
        result[name] = value
    return result


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "simple_workflow.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    registry = build_registry()
    plan = compile_script(_read_script(args.script), registry)
    print(json.dumps(plan.to_dict(), indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    registry = build_registry()
    plan = compile_script(_read_script(args.script), registry)

    inputs: dict[str, InputValue] = {}
    for name, value in _parse_kv(args.text).items():
        inputs[name] = InputValue(ValueType.TEXT, value)
    for name, value in _parse_kv(args.number).items():
        inputs[name] = InputValue(ValueType.NUMBER, float(value))
    for name, path in _parse_kv(args.binary).items():
        inputs[name] = InputValue(ValueType.BINARY, Path(path).read_bytes())

    ctx = RunContext(
        endpoints=_parse_kv(args.endpoint),
        api_keys=_parse_kv(args.api_key),
        timeout=float(os.environ.get("SW_TOOL_TIMEOUT", "300")),
    )
    outputs, _ = asyncio.run(Executor(plan, registry).run(inputs, ctx))

    if args.output_dir:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in outputs.items():
            if isinstance(value, (bytes, bytearray)):
                (directory / f"{name}.bin").write_bytes(bytes(value))
            else:
                (directory / f"{name}.txt").write_text(str(value), encoding="utf-8")
        print(f"wrote {len(outputs)} output(s) to {directory}")
        return 0

    for name, value in outputs.items():
        if isinstance(value, (bytes, bytearray)):
            encoded = encode_value(bytes(value), ValueType.BINARY)
            print(f"{name}: binary ({len(value)} bytes, base64={encoded['data'][:40]}...)")
        else:
            print(f"{name}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simple-workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default=os.environ.get("SW_HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=int(os.environ.get("SW_PORT", "8000")))
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    validate = sub.add_parser("validate", help="compile and print the graph")
    validate.add_argument("script")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="execute a script locally")
    run.add_argument("script")
    run.add_argument("--text", action="append", metavar="NAME=VALUE")
    run.add_argument("--number", action="append", metavar="NAME=NUMBER")
    run.add_argument("--binary", action="append", metavar="NAME=PATH")
    run.add_argument(
        "--endpoint",
        action="append",
        metavar="TOOL=URL",
        help="override an API tool endpoint, e.g. msst=http://host/api/msst/inference",
    )
    run.add_argument(
        "--api-key",
        action="append",
        metavar="TOOL=KEY",
        help="API key for a tool, sent as X-API-Key, e.g. rvc=secret",
    )
    run.add_argument("--output-dir", default=None)
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkflowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
