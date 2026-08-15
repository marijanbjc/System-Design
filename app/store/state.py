"""Redis state: surge counters, automation policy, surge texts, PII vault."""

import time

import redis

from app.config import get_settings
from app.models import AutomationLevel
from app.pii import PiiMap

POLICY_KEY = "policy:automation"
SURGE_TEXT_PREFIX = "TEXT:SURGE:"
SURGE_COUNTER_PREFIX = "surge:"
PII_VAULT_PREFIX = "pii:"


# --- surge detection (architecture.md 6.1) -------------------------------------------------

def bump_surge_counter(client: redis.Redis, topic: str, now: float | None = None) -> None:
    """Increment the current minute bucket for the topic.

    Minute buckets instead of one key per ticket: reading the window is a single MGET
    over a fixed number of keys, so the cost does not depend on traffic volume. A
    `SCAN MATCH` walk would be O(all keys in the database) on every hot-path request —
    exactly when we can least afford it.
    """
    settings = get_settings()
    minute = int((now or time.time()) // 60)
    key = f"{SURGE_COUNTER_PREFIX}{topic}:{minute}"
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, settings.surge_key_ttl_seconds)
    pipe.execute()


def surge_count(client: redis.Redis, topic: str, now: float | None = None) -> int:
    """Sum the sliding window of minute buckets for the topic."""
    settings = get_settings()
    minute = int((now or time.time()) // 60)
    keys = [
        f"{SURGE_COUNTER_PREFIX}{topic}:{minute - offset}"
        for offset in range(settings.surge_window_minutes)
    ]
    return sum(int(value) for value in client.mget(keys) if value)


def surge_text(client: redis.Redis, topic: str) -> str | None:
    """Pre-written incident stub for the topic. Its presence is also the permission.

    Combining permission and text in one key is a simplification, named in
    architecture.md 12: a topic with no text is treated as not allowed, which is safe
    but inflexible.
    """
    value = client.get(f"{SURGE_TEXT_PREFIX}{topic}")
    return value.decode() if isinstance(value, bytes) else value


def set_surge_text(client: redis.Redis, topic: str, text: str) -> None:
    client.set(f"{SURGE_TEXT_PREFIX}{topic}", text)


def delete_surge_text(client: redis.Redis, topic: str) -> None:
    client.delete(f"{SURGE_TEXT_PREFIX}{topic}")


# --- automation policy ---------------------------------------------------------------------

def automation_level(client: redis.Redis, topic: str) -> AutomationLevel:
    """Deterministic gate per topic. Unknown topic defaults to REVIEW_REQUIRED."""
    value = client.hget(POLICY_KEY, topic)
    if value is None:
        return AutomationLevel.REVIEW_REQUIRED
    decoded = value.decode() if isinstance(value, bytes) else value
    try:
        return AutomationLevel(decoded)
    except ValueError:
        return AutomationLevel.REVIEW_REQUIRED


def set_automation_level(client: redis.Redis, topic: str, level: AutomationLevel) -> None:
    client.hset(POLICY_KEY, topic, level.value)


# --- PII vault -----------------------------------------------------------------------------

def store_pii_map(client: redis.Redis, ticket_id: str, mapping: PiiMap) -> None:
    """Keep placeholder -> value pairs long enough to outlive the review queue."""
    if not mapping:
        return
    settings = get_settings()
    key = f"{PII_VAULT_PREFIX}{ticket_id}"
    client.hset(key, mapping=dict(mapping))
    client.expire(key, settings.pii_vault_ttl_seconds)


def load_pii_map(client: redis.Redis, ticket_id: str) -> PiiMap:
    raw = client.hgetall(f"{PII_VAULT_PREFIX}{ticket_id}")
    return PiiMap(
        {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }
    )
