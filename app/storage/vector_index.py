"""Векторные индексы в Redis Stack поверх redisvl: тройки и статьи базы знаний.

Два индекса с одинаковой механикой, но разной ролью:
- `idx:triples` — предодобренные тройки (обращение, тема, ответ), ядро Tier 1;
- `idx:kb` — статьи от менеджеров, контекст для генерации в Tier 2.

Важная асимметрия в том, **что именно эмбеддится**. В индексе троек вектор строится по
обращению пользователя, в индексе базы знаний — по *вероятному вопросу гостя*, а не по
тексту статьи. Причина: пользователь пишет вопрос, а статья написана декларативно, и
эмбеддинг вопроса к вопросу совпадает заметно лучше, чем вопрос к справочному тексту.
Сам текст статьи хранится рядом и отдаётся как контекст, но в поиске не участвует.
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

TRIPLE_FIELDS = ["topic", "question", "answer"]
KB_FIELDS = ["topic", "likely_question", "title", "body"]


@dataclass(frozen=True)
class Hit:
    """Один результат поиска: идентификатор, косинусная близость и поля документа."""

    doc_id: str
    sim: float
    fields: dict[str, str]


def _schema(name: str, prefix: str, text_fields: list[str]) -> IndexSchema:
    """Схема индекса: тег темы для фильтрации, текстовые поля и вектор."""
    settings = get_settings()
    return IndexSchema.from_dict(
        {
            "index": {"name": name, "prefix": prefix, "storage_type": "hash"},
            "fields": [
                {"name": "topic", "type": "tag"},
                *({"name": field, "type": "text"} for field in text_fields),
                {
                    "name": settings.vector_field,
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
        settings = get_settings()
        self.triples = SearchIndex(
            _schema(settings.triples_index, settings.triples_prefix, ["question", "answer"]),
            redis_client=client,
        )
        self.kb = SearchIndex(
            _schema(settings.kb_index, settings.kb_prefix, ["likely_question", "title", "body"]),
            redis_client=client,
        )

    def ensure(self) -> None:
        """Создать индексы, если их ещё нет."""
        self.triples.create(overwrite=False)
        self.kb.create(overwrite=False)

    def add_triple(
        self, doc_id: str, topic: str, question: str, answer: str, vector: np.ndarray
    ) -> None:
        """Положить в индекс предодобренную тройку. Вектор строится по обращению."""
        settings = get_settings()
        self.triples.load(
            [
                {
                    "topic": topic,
                    "question": question,
                    "answer": answer,
                    settings.vector_field: vector.astype(np.float32).tobytes(),
                }
            ],
            keys=[f"{settings.triples_prefix}:{doc_id}"],
        )

    def add_article(
        self,
        doc_id: str,
        topic: str,
        title: str,
        body: str,
        likely_question: str,
        vector: np.ndarray,
    ) -> None:
        """Положить в индекс статью базы знаний.

        Вектор обязан быть построен по `likely_question`: ищем мы по вопросу, а текст
        статьи отдаём как контекст (см. пояснение в шапке модуля).
        """
        settings = get_settings()
        self.kb.load(
            [
                {
                    "topic": topic,
                    "likely_question": likely_question,
                    "title": title,
                    "body": body,
                    settings.vector_field: vector.astype(np.float32).tobytes(),
                }
            ],
            keys=[f"{settings.kb_prefix}:{doc_id}"],
        )

    def search_triples(self, vector: np.ndarray, topic: str) -> list[Hit]:
        """KNN по тройкам с фильтром по теме, которую уже определил классификатор.

        Фильтр — то, что не даёт слипнуться лексически близким, но разным по смыслу
        обращениям («отменить заказ» и «отменить подписку»). Если классификатор ошибся
        темой, поиск просто вернёт пусто и мы провалимся в генерацию — направление
        отказа безопасное.
        """
        return self._knn(self.triples, vector, TRIPLE_FIELDS, topic)

    def search_kb(self, vector: np.ndarray) -> list[Hit]:
        """KNN по базе знаний. Без фильтра по теме: контекст может лежать где угодно."""
        return self._knn(self.kb, vector, KB_FIELDS, None)

    @staticmethod
    def _knn(
        index: SearchIndex, vector: np.ndarray, fields: list[str], topic: str | None
    ) -> list[Hit]:
        settings = get_settings()
        query = VectorQuery(
            vector=vector.astype(np.float32).tolist(),
            vector_field_name=settings.vector_field,
            return_fields=fields,
            num_results=settings.retrieval_top_k,
            filter_expression=(Tag("topic") == topic) if topic else None,
        )
        return _to_hits(index.query(query), fields)


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
