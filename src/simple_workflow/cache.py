"""Layered, S3-backed result cache for workflow nodes.

Each tool node is given a **layered** cache key: a SHA-256 digest over the
node's own identity (its tool name and literal arguments) combined with the
cache keys of every node feeding it.  A node's output therefore only counts as
a hit when the entire sub-graph that produced it is identical, so a re-run can
reuse every node up to and including the first one whose inputs actually
changed.

For the example pipeline::

    [bv:BV1wM1vYsEn9]
    -> band-roformer[msst:melband_roformer_instvox_duality_v2.ckpt]
    -> [unzip:Vocals.wav]
    -> [audio:mono]
    -> [final]

the keys nest like this::

    key(bv)     = H("tool", "bv", "BV1wM1vYsEn9")
    key(msst)   = H("tool", "msst", "melband_roformer_instvox_duality_v2.ckpt", key(bv))
    key(unzip)  = H("tool", "unzip", "Vocals.wav", key(msst))
    key(audio)  = H("tool", "audio", "mono", key(unzip))

Entries are stored in S3 and expire after a configurable TTL (default 60
minutes).  A background task periodically evicts the expired objects, and
:meth:`CacheStore.clear` removes everything on demand.

The cache is disabled unless ``SW_CACHE_BUCKET`` is set (or the in-memory
backend is explicitly selected), so the engine keeps working out of the box.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .compiler import Node
from .types import ValueType

logger = logging.getLogger(__name__)

#: Bump this to invalidate every entry whenever the key scheme changes.
_KEY_SCHEME = "simple-workflow-cache-v1"
_KEY_SEPARATOR = b"\x1f"

DEFAULT_TTL = 3600.0
DEFAULT_PREFIX = "simple-workflow/cache/"
DEFAULT_CLEANUP_INTERVAL = 300.0


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def _hash_parts(*parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(_KEY_SCHEME.encode("utf-8"))
    for part in parts:
        digest.update(_KEY_SEPARATOR)
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def digest_value(value: Any) -> str:  # noqa: ANN401 - any of the three value types
    """Return a stable digest of a concrete runtime value."""
    digest = hashlib.sha256()
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"b")
        digest.update(bytes(value))
    elif isinstance(value, str):
        digest.update(b"t")
        digest.update(value.encode("utf-8"))
    elif isinstance(value, (int, float)):
        digest.update(b"n")
        digest.update(repr(float(value)).encode("ascii"))
    else:  # pragma: no cover - values are validated by the engine
        digest.update(b"?")
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def node_cache_key(
    node: Node,
    parent_keys: list[str],
    *,
    input_value: Any = None,  # noqa: ANN401 - any of the three value types
) -> str:
    """Derive the layered cache key for ``node``.

    ``parent_keys`` are the keys of the nodes feeding this one, in edge order.
    ``input_value`` is only used for ``input`` nodes, where the run-time input
    participates in the key.
    """
    if node.kind == "input":
        return _hash_parts("input", node.input_name or "input", digest_value(input_value))
    if node.kind == "ref":
        return _hash_parts("ref", *parent_keys)
    if node.kind == "tool":
        return _hash_parts("tool", node.tool_name or "", *node.args, *parent_keys)
    return _hash_parts("sink", *parent_keys)


# ---------------------------------------------------------------------------
# Value codec (the object body is raw bytes; the node knows its output type)
# ---------------------------------------------------------------------------


def encode_cached(value: Any, value_type: ValueType) -> bytes:  # noqa: ANN401
    if value_type is ValueType.BINARY:
        return bytes(value)
    if value_type is ValueType.NUMBER:
        return repr(float(value)).encode("ascii")
    return str(value).encode("utf-8")


def decode_cached(body: bytes, value_type: ValueType) -> Any:  # noqa: ANN401
    if value_type is ValueType.BINARY:
        return body
    if value_type is ValueType.NUMBER:
        return float(body.decode("ascii"))
    return body.decode("utf-8")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheStore(Protocol):
    """A byte-oriented cache backend."""

    enabled: bool

    async def get(self, key: str) -> bytes | None:
        """Return the cached body for ``key`` or ``None`` on a miss/expiry."""

    async def put(self, key: str, data: bytes, metadata: Mapping[str, str]) -> None:
        """Store ``data`` under ``key``."""

    async def clear(self) -> int:
        """Delete every entry; return the number removed."""

    async def purge_expired(self) -> int:
        """Delete entries past their TTL; return the number removed."""


class NullStore:
    """The no-op cache used when the cache is disabled."""

    enabled = False

    async def get(self, key: str) -> bytes | None:
        return None

    async def put(self, key: str, data: bytes, metadata: Mapping[str, str]) -> None:
        return None

    async def clear(self) -> int:
        return 0

    async def purge_expired(self) -> int:
        return 0


class MemoryStore:
    """A small in-memory cache.  Handy for tests and local development."""

    enabled = True

    def __init__(self, ttl: float = DEFAULT_TTL) -> None:
        self.ttl = ttl
        self._entries: dict[str, tuple[float, bytes]] = {}

    async def get(self, key: str) -> bytes | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, data = entry
        if expires_at <= time.time():
            self._entries.pop(key, None)
            return None
        return data

    async def put(self, key: str, data: bytes, metadata: Mapping[str, str]) -> None:
        self._entries[key] = (time.time() + self.ttl, bytes(data))

    async def clear(self) -> int:
        removed = len(self._entries)
        self._entries.clear()
        return removed

    async def purge_expired(self) -> int:
        now = time.time()
        expired = [key for key, (expires_at, _) in self._entries.items() if expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
        return len(expired)

    def __len__(self) -> int:  # pragma: no cover - debugging aid
        return len(self._entries)


def _chunks(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


class S3Store:
    """An S3-backed cache.

    ``boto3`` is imported lazily so the dependency stays optional.  Objects are
    written under ``prefix`` sharded by the first two hex characters of the key,
    which keeps listing cheap and spreads the load across partitions.
    """

    enabled = True

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = DEFAULT_PREFIX,
        ttl: float = DEFAULT_TTL,
        endpoint_url: str | None = None,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        self.ttl = ttl
        self.endpoint_url = endpoint_url
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.session_token = session_token
        self._client: Any = None

    # -- client ------------------------------------------------------------
    def _get_client(self) -> Any:  # noqa: ANN401 - boto3 is untyped/optional
        if self._client is None:
            import importlib

            boto3 = importlib.import_module("boto3")
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token,
            )
        return self._client

    def _object_key(self, key: str) -> str:
        return f"{self.prefix}{key[:2]}/{key}"

    @staticmethod
    def _is_missing(exc: BaseException) -> bool:
        import importlib

        client_error = importlib.import_module("botocore.exceptions").ClientError
        if not isinstance(exc, client_error):
            return False
        code = exc.response.get("Error", {}).get("Code")
        return code in ("NoSuchKey", "NoSuchBucket", "404", "NotFound")

    # -- sync primitives ---------------------------------------------------
    def _get_sync(self, key: str) -> bytes | None:
        client = self._get_client()
        object_key = self._object_key(key)
        try:
            response = client.get_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a miss
            if self._is_missing(exc):
                return None
            raise
        last_modified = response["LastModified"].timestamp()
        if time.time() - last_modified > self.ttl:
            self._delete_keys_sync([object_key])
            return None
        body: bytes = response["Body"].read()
        return body

    def _put_sync(self, key: str, data: bytes, metadata: Mapping[str, str]) -> None:
        client = self._get_client()
        client.put_object(
            Bucket=self.bucket,
            Key=self._object_key(key),
            Body=data,
            Metadata={name: str(value) for name, value in metadata.items()},
        )

    def _delete_keys_sync(self, keys: list[str]) -> None:
        if not keys:
            return
        client = self._get_client()
        for chunk in _chunks([{"Key": key} for key in keys], 1000):
            client.delete_objects(
                Bucket=self.bucket, Delete={"Objects": chunk, "Quiet": True}
            )

    def _list_sync(self) -> Any:  # noqa: ANN401 - boto3 paginator
        client = self._get_client()
        paginator = client.get_paginator("list_objects_v2")
        return paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)

    def _clear_sync(self) -> int:
        removed = 0
        batch: list[str] = []
        for page in self._list_sync():
            for obj in page.get("Contents", []):
                batch.append(obj["Key"])
                if len(batch) >= 1000:
                    self._delete_keys_sync(batch)
                    removed += len(batch)
                    batch = []
        self._delete_keys_sync(batch)
        removed += len(batch)
        return removed

    def _purge_sync(self) -> int:
        cutoff = time.time() - self.ttl
        expired: list[str] = []
        for page in self._list_sync():
            for obj in page.get("Contents", []):
                if obj["LastModified"].timestamp() < cutoff:
                    expired.append(obj["Key"])
        if expired:
            self._delete_keys_sync(expired)
        return len(expired)

    # -- async API ---------------------------------------------------------
    async def get(self, key: str) -> bytes | None:
        try:
            return await asyncio.to_thread(self._get_sync, key)
        except Exception:  # noqa: BLE001 - a cache miss must never fail the run
            logger.warning("cache read failed for key %s", key, exc_info=True)
            return None

    async def put(self, key: str, data: bytes, metadata: Mapping[str, str]) -> None:
        try:
            await asyncio.to_thread(self._put_sync, key, data, metadata)
        except Exception:  # noqa: BLE001
            logger.warning("cache write failed for key %s", key, exc_info=True)

    async def clear(self) -> int:
        try:
            return await asyncio.to_thread(self._clear_sync)
        except Exception:  # noqa: BLE001
            logger.warning("cache clear failed", exc_info=True)
            return 0

    async def purge_expired(self) -> int:
        try:
            return await asyncio.to_thread(self._purge_sync)
        except Exception:  # noqa: BLE001
            logger.warning("cache purge failed", exc_info=True)
            return 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r; using %s", name, raw, default)
        return default


def build_cache() -> CacheStore:
    """Build the cache from the ``SW_CACHE_*`` environment variables.

    * ``SW_CACHE_BUCKET``  - enables the S3 backend when set.
    * ``SW_CACHE_BACKEND`` - ``memory`` selects the in-memory backend.
    * ``SW_CACHE_TTL``     - entry TTL in seconds (default 3600).
    * ``SW_CACHE_PREFIX``  - object key prefix inside the bucket.
    * ``SW_CACHE_ENDPOINT`` / ``SW_CACHE_REGION`` / ``SW_CACHE_ACCESS_KEY`` /
      ``SW_CACHE_SECRET_KEY`` / ``SW_CACHE_SESSION_TOKEN`` - S3 client options.

    When neither a bucket nor a backend is configured the cache is disabled.
    """
    ttl = _env_float("SW_CACHE_TTL", DEFAULT_TTL)
    bucket = os.environ.get("SW_CACHE_BUCKET", "").strip()
    backend = os.environ.get("SW_CACHE_BACKEND", "").strip().lower()

    if bucket:
        return S3Store(
            bucket,
            prefix=os.environ.get("SW_CACHE_PREFIX", DEFAULT_PREFIX),
            ttl=ttl,
            endpoint_url=os.environ.get("SW_CACHE_ENDPOINT") or None,
            region=os.environ.get("SW_CACHE_REGION") or None,
            access_key=os.environ.get("SW_CACHE_ACCESS_KEY") or None,
            secret_key=os.environ.get("SW_CACHE_SECRET_KEY") or None,
            session_token=os.environ.get("SW_CACHE_SESSION_TOKEN") or None,
        )
    if backend == "memory":
        return MemoryStore(ttl=ttl)
    if backend and backend != "none":
        logger.warning("unknown SW_CACHE_BACKEND=%r; cache disabled", backend)
    return NullStore()


def cleanup_interval() -> float:
    """Seconds between background cleanups (``SW_CACHE_CLEANUP_INTERVAL``)."""
    return _env_float("SW_CACHE_CLEANUP_INTERVAL", DEFAULT_CLEANUP_INTERVAL)


__all__ = [
    "CacheStore",
    "MemoryStore",
    "NullStore",
    "S3Store",
    "build_cache",
    "cleanup_interval",
    "decode_cached",
    "digest_value",
    "encode_cached",
    "node_cache_key",
]
