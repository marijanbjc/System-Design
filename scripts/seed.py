"""Seed Redis: vector indexes, automation policy, surge stubs. Idempotent."""

import json
from pathlib import Path

import redis

from app.config import get_settings
from app.embed import embed
from app.models import AutomationLevel
from app.store import state, vectors

DATA = Path(__file__).resolve().parent.parent / "data"

# Compliance-owned table. Risky categories are never closed automatically, and this is
# a deterministic table rather than a model output on purpose.
POLICY: dict[str, AutomationLevel] = {
    "returns": AutomationLevel.AUTO_OK,
    "delivery": AutomationLevel.AUTO_OK,
    "loyalty": AutomationLevel.AUTO_OK,
    "account": AutomationLevel.REVIEW_REQUIRED,
    "payment": AutomationLevel.REVIEW_REQUIRED,
    "billing_dispute": AutomationLevel.OPERATOR_ONLY,
    "security": AutomationLevel.OPERATOR_ONLY,
}

# Only topics with a manager-written stub may be auto-answered during a surge.
SURGE_STUBS: dict[str, str] = {
    "payment": (
        "Мы знаем о проблеме с оплатой и уже её чиним. "
        "Как только всё заработает, мы вам напишем — обращение зарегистрировано."
    ),
    "delivery": (
        "Мы знаем о задержках доставки в вашем регионе и работаем над этим. "
        "Как только ситуация нормализуется, мы вам напишем."
    ),
}


def seed(client: redis.Redis) -> dict[str, int]:
    vectors.ensure_indexes(client)

    triples = json.loads((DATA / "triples.json").read_text(encoding="utf-8"))
    for i, item in enumerate(triples):
        vectors.add_triple(
            client, f"t{i}", item["topic"], item["question"], item["answer"], embed(item["question"])
        )

    articles = json.loads((DATA / "kb.json").read_text(encoding="utf-8"))
    for i, item in enumerate(articles):
        indexed = f"{item.get('likely_question', '')} {item['title']} {item['body']}".strip()
        vectors.add_article(
            client, f"a{i}", item["topic"], item["title"], item["body"], embed(indexed)
        )

    for topic, level in POLICY.items():
        state.set_automation_level(client, topic, level)
    for topic, text in SURGE_STUBS.items():
        state.set_surge_text(client, topic, text)

    return {"triples": len(triples), "articles": len(articles), "policies": len(POLICY)}


def main() -> None:
    client = redis.from_url(get_settings().redis_url)
    stats = seed(client)
    print(f"seeded: {stats}")


if __name__ == "__main__":
    main()
