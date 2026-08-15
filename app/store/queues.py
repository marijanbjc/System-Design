"""Redis Streams as the decoupling layer between the hot path and the async workers.

In the target architecture this is Kafka on the ingest bus plus a dedicated task
queue; Streams give the same shape (durable, consumable, inspectable) at PoC cost.
"""

import json
from typing import Any

import redis


def publish(client: redis.Redis, stream: str, payload: dict[str, Any]) -> str:
    """Append a task to a stream and return its entry id."""
    entry = client.xadd(stream, {"payload": json.dumps(payload, ensure_ascii=False)})
    return entry.decode() if isinstance(entry, bytes) else str(entry)


def consume(client: redis.Redis, stream: str, count: int = 10) -> list[tuple[str, dict[str, Any]]]:
    """Read and delete up to `count` pending entries.

    Deliberately simple: no consumer groups, no acking. Single-worker PoC — the
    at-least-once semantics of a real deployment are an explicit simplification.
    """
    entries = client.xrange(stream, count=count)
    result: list[tuple[str, dict[str, Any]]] = []
    for entry_id, fields in entries:
        raw = fields.get(b"payload") or fields.get("payload")
        if raw is None:
            continue
        payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        result.append((entry_id.decode() if isinstance(entry_id, bytes) else entry_id, payload))
        client.xdel(stream, entry_id)
    return result


def depth(client: redis.Redis, stream: str) -> int:
    """Queue depth — the metric that triggers the "we need capacity" alert."""
    return int(client.xlen(stream))
