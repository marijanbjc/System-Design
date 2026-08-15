"""Redis Streams — развязка между горячим путём и асинхронными воркерами.

Три очереди с разными потребителями и разными SLA: генерация, ревью оператором,
доставка. В целевой архитектуре это Kafka на шину ингеста плюс отдельная очередь
задач; Streams дают ту же форму (durable, читаемая, инспектируемая) ценой PoC.
"""

import json
from typing import Any

import redis


def publish(client: redis.Redis, stream: str, payload: dict[str, Any]) -> str:
    """Положить задачу в очередь и вернуть идентификатор записи."""
    entry = client.xadd(stream, {"payload": json.dumps(payload, ensure_ascii=False)})
    return entry.decode() if isinstance(entry, bytes) else str(entry)


def consume(client: redis.Redis, stream: str, count: int = 10) -> list[tuple[str, dict[str, Any]]]:
    """Прочитать и удалить до `count` записей.

    Намеренно просто: без consumer groups и без подтверждений. Воркер в PoC один,
    а семантика at-least-once реального развёртывания — явное упрощение.
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


def peek(client: redis.Redis, stream: str, count: int = 50) -> list[dict[str, Any]]:
    """Прочитать записи, не удаляя их: так очередь оператора показывается в API."""
    entries = client.xrange(stream, count=count)
    payloads = []
    for _, fields in entries:
        raw = fields.get(b"payload") or fields.get("payload")
        if raw is not None:
            payloads.append(json.loads(raw.decode() if isinstance(raw, bytes) else raw))
    return payloads
