"""Векторные индексы в Redis Stack поверх redisvl: тройки и статьи базы знаний.

Два индекса с одинаковой механикой, но разной ролью:
- `idx:triples` — предодобренные тройки (обращение, тема, ответ), ядро Tier 1;
- `idx:kb` — статьи от менеджеров, контекст для генерации в Tier 2.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema

from app.config import get_settings

TRIPLES_INDEX = "idx:triples"
TRIPLES_PREFIX = "triple"
KB_INDEX = "idx:kb"
KB_PREFIX = "kb"

VECTOR_FIELD = "embedding"


@dataclass(frozen=True)
class Hit:
    """Один результат поиска: идентификатор, косинусная близость и поля документа."""

    doc_id: str
    sim: float
    fields: dict[str, str]


def _schema(name: str, prefix: str, text_fields: list[str]) -> IndexSchema:
    settings = get_settings()
    return IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": prefix, "storage_type": "hash"},
            "fields": [
                {"name": "topic", "type": "tag"},
                *({"name": field, "type": "text"} for field in text_fields),
                {
                    "name": VECTOR_FIELD,
                    "type": "vector",
                    "attrs": {
                        "dims": settings.embed_dim,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                        "datatype": "float32",
                    },
                },
            ],
        }
    )


class VectorStore:
    """Обёртка над двумя индексами. Создание идемпотентно — можно звать на каждом старте."""

    def __init__(self, client: redis.Redis) -> None:
        self.triples = SearchIndex(
            _schema(TRIPLES_INDEX, TRIPLES_PREFIX, ["question", "answer"]), redis_client=client
        )
        self.kb = SearchIndex(_schema(KB_INDEX, KB_PREFIX, ["title", "body"]), redis_client=client)

    def ensure(self) -> None:
        """Создать индексы, если их ещё нет."""
        self.triples.create(overwrite=False)
        self.kb.create(overwrite=False)

    def add_triple(
        self, doc_id: str, topic: str, question: str, answer: str, vector: np.ndarray
    ) -> None:
        """Положить в индекс предодобренную тройку."""
        self.triples.load(
            [
                {
                    "topic": topic,
                    "question": question,
                    "answer": answer,
                    VECTOR_FIELD: vector.astype(np.float32).tobytes(),
                }
            ],
            keys=[f"{TRIPLES_PREFIX}:{doc_id}"],
        )

    def add_article(
        self, doc_id: str, topic: str, title: str, body: str, vector: np.ndarray
    ) -> None:
        """Положить в индекс статью базы знаний, загруженную менеджером."""
        self.kb.load(
            [
                {
                    "topic": topic,
                    "title": title,
                    "body": body,
                    VECTOR_FIELD: vector.astype(np.float32).tobytes(),
                }
            ],
            keys=[f"{KB_PREFIX}:{doc_id}"],
        )

    def search_triples(self, vector: np.ndarray, topic: str, limit: int = 3) -> list[Hit]:
        """KNN по тройкам с фильтром по теме, которую уже определил классификатор.

        Фильтр — то, что не даёт слипнуться лексически близким, но разным по смыслу
        обращениям («отменить заказ» и «отменить подписку»). Если классификатор ошибся
        темой, поиск просто вернёт пусто и мы провалимся в генерацию — направление
        отказа безопасное.
        """
        query = VectorQuery(
            vector=vector.astype(np.float32).tolist(),
            vector_field_name=VECTOR_FIELD,
            return_fields=["topic", "question", "answer"],
            num_results=limit,
            filter_expression=Tag("topic") == topic,
        )
        return _to_hits(self.triples.query(query), ["topic", "question", "answer"])

    def search_kb(self, vector: np.ndarray, limit: int = 3) -> list[Hit]:
        """KNN по базе знаний. Без фильтра по теме: контекст может лежать где угодно."""
        query = VectorQuery(
            vector=vector.astype(np.float32).tolist(),
            vector_field_name=VECTOR_FIELD,
            return_fields=["topic", "title", "body"],
            num_results=limit,
        )
        return _to_hits(self.kb.query(query), ["topic", "title", "body"])


def _to_hits(rows: list[dict[str, Any]], fields: list[str]) -> list[Hit]:
    """Перевести ответ redisvl в наши Hit: косинусная близость = 1 − расстояние."""
    return [
        Hit(
            doc_id=row["id"],
            sim=1.0 - float(row["vector_distance"]),
            fields={name: row.get(name, "") for name in fields},
        )
        for row in rows
    ]
