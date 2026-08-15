"""Наполнение Redis: векторные индексы, политика автоматизации, заглушки на всплеск.

Идемпотентно — можно запускать сколько угодно раз.
"""

import json
from pathlib import Path

import redis

from app.config import get_settings
from app.ml.encoder import embed
from app.models import AutomationLevel
from app.storage import state
from app.storage.vector_index import VectorStore

DATA = Path(__file__).resolve().parent.parent / "data"

# Таблица комплаенса. Рисковые категории не закрываются автоматически никогда, и это
# детерминированная таблица, а не выход модели — намеренно.
POLICY: dict[str, AutomationLevel] = {
    "returns": AutomationLevel.AUTO_OK,
    "delivery": AutomationLevel.AUTO_OK,
    "loyalty": AutomationLevel.AUTO_OK,
    "account": AutomationLevel.REVIEW_REQUIRED,
    "payment": AutomationLevel.REVIEW_REQUIRED,
    "billing_dispute": AutomationLevel.OPERATOR_ONLY,
    "security": AutomationLevel.OPERATOR_ONLY,
}

# Авто-ответ при всплеске разрешён только темам, для которых менеджер написал текст.
# Формулировка не требует от пользователя действий: обещание вернуться самим не
# порождает вторую волну обращений после резолва инцидента.
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
    """Создать индексы и загрузить тройки, статьи, политику и заглушки."""
    vectors = VectorStore(client)
    vectors.ensure()

    triples = json.loads((DATA / "triples.json").read_text(encoding="utf-8"))
    for i, item in enumerate(triples):
        vectors.add_triple(
            f"t{i}", item["topic"], item["question"], item["answer"], embed(item["question"])
        )

    articles = json.loads((DATA / "kb.json").read_text(encoding="utf-8"))
    for i, item in enumerate(articles):
        # Вероятный вопрос гостя индексируется вместе с текстом: пользователь
        # спрашивает, а статья написана декларативно, и этот мостик поднимает recall.
        indexed = f"{item.get('likely_question', '')} {item['title']} {item['body']}".strip()
        vectors.add_article(f"a{i}", item["topic"], item["title"], item["body"], embed(indexed))

    for topic, level in POLICY.items():
        state.set_automation_level(client, topic, level)
    for topic, text in SURGE_STUBS.items():
        state.set_surge_text(client, topic, text)

    return {
        "triples": len(triples),
        "articles": len(articles),
        "policies": len(POLICY),
        "surge_stubs": len(SURGE_STUBS),
    }


def main() -> None:
    """Точка входа: python scripts/seed.py."""
    client = redis.from_url(get_settings().redis_url)
    print(f"загружено: {seed(client)}")


if __name__ == "__main__":
    main()
