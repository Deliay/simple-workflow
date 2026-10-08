"""HTTP API tests."""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from simple_workflow.api import app

from .conftest import LONG_SCRIPT, make_zip


def test_health_and_tools() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        tools = {tool["name"] for tool in client.get("/v1/tools").json()["tools"]}
        assert {"bv", "msst", "rvc", "unzip", "audio"} <= tools


def test_compile_long_script() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/compile", json={"script": LONG_SCRIPT})
        assert response.status_code == 200
        plan = response.json()["plan"]
        assert plan["outputs"] == {"final": "binary"}


def test_compile_reports_errors() -> None:
    with TestClient(app) as client:
        response = client.post("/v1/compile", json={"script": "[nope:x] -> [final]"})
        assert response.status_code == 400
        assert "unknown tool" in response.json()["detail"]


def test_run_local_workflow() -> None:
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    archive = make_zip(**{"vocals.wav": b"payload"})
    body = {
        "script": script,
        "inputs": {
            "data": {"type": "binary", "data": base64.b64encode(archive).decode()}
        },
    }
    with TestClient(app) as client:
        response = client.post("/v1/run", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert base64.b64decode(payload["outputs"]["final"]["data"]) == b"payload"


def test_api_tool_without_endpoint_fails() -> None:
    body = {"script": "[bv:BV1] -> [final]"}
    with TestClient(app) as client:
        response = client.post("/v1/run", json=body)
    assert response.status_code == 400
    assert "endpoint" in response.json()["detail"]
