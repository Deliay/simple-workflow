"""Layered cache tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from simple_workflow import (
    Executor,
    InputValue,
    MemoryStore,
    NullStore,
    S3Store,
    ValueType,
    build_cache,
    compile_script,
)
from simple_workflow.cache import (
    decode_cached,
    digest_value,
    encode_cached,
    node_cache_key,
)

from .conftest import build_stub_registry

CHAIN = """\
[bv:BV1wM1vYsEn9]
-> band[msst:melband_roformer_instvox_duality_v2.ckpt]
-> [unzip:Vocals.wav]
-> [audio:mono]
-> [final]
"""


def _run(script, registry, cache, inputs=None):  # noqa: ANN001
    plan = compile_script(script, registry)
    executor = Executor(plan, registry, cache=cache)
    return asyncio.run(executor.run(inputs or {}))


def _tool_calls(registry) -> int:  # noqa: ANN001
    return sum(len(tool.calls) for tool in registry if hasattr(tool, "calls"))


def test_chain_is_fully_cached_on_second_run(registry) -> None:  # noqa: ANN001
    cache = MemoryStore(ttl=3600)
    _run(CHAIN, registry, cache)
    first = _tool_calls(registry)
    outputs, timings = _run(CHAIN, registry, cache)

    assert _tool_calls(registry) == first  # nothing re-executed
    assert outputs["final"] == b"mono:K"
    # bv, msst, unzip and audio — the four tool nodes — all come from the cache.
    assert sum(result.cached for result in timings) == 4


def test_changed_root_input_invalidates_downstream(registry) -> None:  # noqa: ANN001
    cache = MemoryStore(ttl=3600)
    _run(CHAIN, registry, cache)
    first = _tool_calls(registry)
    changed = CHAIN.replace("BV1wM1vYsEn9", "BV1OTHER")
    _run(changed, registry, cache)

    # Every node sits on the changed root's subtree, so the whole chain re-runs.
    # (The real `unzip` tool records no calls, so only bv/msst/audio are counted.)
    assert _tool_calls(registry) == first + 3


def test_shared_subtree_reused_across_workflows(registry) -> None:  # noqa: ANN001
    cache = MemoryStore(ttl=3600)
    # Two workflows share a prefix; the shared nodes must only run once.
    script_a = "[bv:ID] -> a[msst:m.ckpt]\n[var:a] -> [final]"
    script_b = "[bv:ID] -> a[msst:m.ckpt]\n[var:a] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    _run(script_a, registry, cache)
    before = registry.get("bv").calls.__len__()
    _run(script_b, registry, cache)

    assert registry.get("bv").calls.__len__() == before  # reused
    assert registry.get("msst").calls.__len__() == 1


def test_node_cache_key_layering(registry) -> None:  # noqa: ANN001
    plan = compile_script(CHAIN, registry)
    keys: dict[str, str] = {}
    for node_id in plan.order:
        node = plan.nodes[node_id]
        keys[node_id] = node_cache_key(
            node, [keys[parent] for parent in node.incoming]
        )

    tools = {node.tool_name: keys[node_id] for node_id, node in plan.nodes.items()}
    assert keys[plan.order[0]] == keys[plan.nodes[plan.order[0]].id]  # root is stable
    # msst's key nests the bv key; unzip nests msst; audio nests unzip.
    assert len(set(keys.values())) == len(keys)
    assert tools["bv"] != tools["msst"] != tools["unzip"] != tools["audio"]


def test_node_cache_key_is_stable_and_typed() -> None:
    assert digest_value(b"x") == digest_value(b"x")
    assert digest_value(b"x") != digest_value("x")
    assert digest_value(1.0) == digest_value(1)


def test_value_codec_roundtrip() -> None:
    assert decode_cached(encode_cached(b"abc", ValueType.BINARY), ValueType.BINARY) == b"abc"
    assert decode_cached(encode_cached("hi", ValueType.TEXT), ValueType.TEXT) == "hi"
    assert decode_cached(encode_cached(1.5, ValueType.NUMBER), ValueType.NUMBER) == 1.5


def test_memory_store_expiry_and_clear() -> None:
    async def scenario() -> None:
        store = MemoryStore(ttl=0.0)
        await store.put("k", b"v", {})
        assert await store.get("k") is None  # already expired
        assert await store.purge_expired() == 0

        store = MemoryStore(ttl=3600)
        await store.put("a", b"1", {})
        await store.put("b", b"2", {})
        assert await store.clear() == 2
        assert await store.get("a") is None

    asyncio.run(scenario())


def test_memory_store_purge() -> None:
    async def scenario() -> None:
        store = MemoryStore(ttl=0.0)
        await store.put("a", b"1", {})
        await store.put("b", b"2", {})
        assert await store.purge_expired() == 2

    asyncio.run(scenario())


def test_null_store_is_disabled() -> None:
    async def scenario() -> None:
        store = NullStore()
        assert store.enabled is False
        await store.put("k", b"v", {})
        assert await store.get("k") is None
        assert await store.clear() == 0

    asyncio.run(scenario())


def test_build_cache_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SW_CACHE_BUCKET", raising=False)
    monkeypatch.delenv("SW_CACHE_BACKEND", raising=False)
    assert isinstance(build_cache(), NullStore)

    monkeypatch.setenv("SW_CACHE_BACKEND", "memory")
    assert isinstance(build_cache(), MemoryStore)

    monkeypatch.setenv("SW_CACHE_BUCKET", "my-bucket")
    monkeypatch.setenv("SW_CACHE_PREFIX", "cache")
    cache = build_cache()
    assert isinstance(cache, S3Store)
    assert cache.bucket == "my-bucket"
    assert cache.prefix == "cache/"


def test_input_value_participates_in_key(registry) -> None:  # noqa: ANN001
    cache = MemoryStore(ttl=3600)
    script = "[input:data] -> out[unzip:vocals.wav]\n[var:out] -> [final]"
    from .conftest import make_zip

    first_zip = make_zip(**{"vocals.wav": b"one"})
    second_zip = make_zip(**{"vocals.wav": b"two"})
    out_a, _ = _run(script, registry, cache, {"data": InputValue(ValueType.BINARY, first_zip)})
    out_b, _ = _run(script, registry, cache, {"data": InputValue(ValueType.BINARY, second_zip)})
    assert out_a["final"] == b"one"
    assert out_b["final"] == b"two"


def test_registry_stub_tool_names(registry) -> None:  # noqa: ANN001
    # Guard the fixture used by the cache tests above.
    assert set(build_stub_registry().names()) >= {"bv", "msst", "unzip", "audio"}


# ---------------------------------------------------------------------------
# S3 backend (exercised through an injected fake client)
# ---------------------------------------------------------------------------


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakePaginator:
    def __init__(self, objects: dict[str, tuple[bytes, float]]) -> None:
        self._objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):  # noqa: ANN201
        contents = [
            {
                "Key": key,
                "LastModified": _dt(value[1]),
            }
            for key, value in sorted(self._objects.items())
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}


def _dt(timestamp: float):  # noqa: ANN202
    from datetime import UTC, datetime

    return datetime.fromtimestamp(timestamp, tz=UTC)


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, float]] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]) -> None:
        self.objects[Key] = (bytes(Body), time.time())

    def get_object(self, *, Bucket: str, Key: str):  # noqa: ANN201
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data, modified = self.objects[Key]
        return {"Body": _FakeBody(data), "LastModified": _dt(modified)}

    def delete_objects(self, *, Bucket: str, Delete: dict) -> None:
        for obj in Delete["Objects"]:
            self.objects.pop(obj["Key"], None)

    def get_paginator(self, name: str):  # noqa: ANN201
        return _FakePaginator(self.objects)


@pytest.fixture
def s3_store():
    pytest.importorskip("botocore")
    store = S3Store("bucket", prefix="sw/cache/", ttl=3600)
    fake = _FakeS3()
    store._get_client = lambda: fake  # type: ignore[method-assign]
    return store, fake


def test_s3_roundtrip_and_sharding(s3_store) -> None:  # noqa: ANN001
    store, fake = s3_store

    async def scenario() -> None:
        await store.put("abcdef", b"payload", {"sw-type": "binary"})
        assert await store.get("abcdef") == b"payload"
        # sharded by the first two characters under the prefix
        assert any(key.startswith("sw/cache/ab/abcdef") for key in fake.objects)

    asyncio.run(scenario())


def test_s3_miss_returns_none(s3_store) -> None:  # noqa: ANN001
    store, _ = s3_store
    assert asyncio.run(store.get("missing")) is None


def test_s3_ttl_expiry(s3_store) -> None:  # noqa: ANN001
    store, fake = s3_store

    async def scenario() -> None:
        await store.put("k1", b"v", {})
        object_key = store._object_key("k1")
        data, _ = fake.objects[object_key]
        fake.objects[object_key] = (data, time.time() - store.ttl - 1)
        assert await store.get("k1") is None  # stale on read
        assert object_key not in fake.objects  # and evicted

    asyncio.run(scenario())


def test_s3_purge_and_clear(s3_store) -> None:  # noqa: ANN001
    store, fake = s3_store

    async def scenario() -> None:
        await store.put("aa1", b"1", {})
        await store.put("bb2", b"2", {})
        expired_key = store._object_key("aa1")
        fake.objects[expired_key] = (b"1", time.time() - store.ttl - 1)
        assert await store.purge_expired() == 1
        assert await store.clear() == 1
        assert fake.objects == {}

    asyncio.run(scenario())


def test_s3_build_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("botocore")
    monkeypatch.setenv("SW_CACHE_BUCKET", "b")
    monkeypatch.setenv("SW_CACHE_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("SW_CACHE_TTL", "1800")
    cache = build_cache()
    assert isinstance(cache, S3Store)
    assert cache.endpoint_url == "http://minio:9000"
    assert cache.ttl == 1800
