"""Сборка зависимостей приложения в одном месте.

Здесь видно, что от чего зависит: горячий путь — от классификатора, векторного
хранилища и аудита; воркер — дополнительно от LLM.
"""

from dataclasses import dataclass

import redis

from app.config import get_settings
from app.llm.mock import MockLLM
from app.ml.classifier import Classifier
from app.routing.router import Router
from app.storage.audit import AuditStore
from app.storage.vector_index import VectorStore
from app.workers.generation import GenerationWorker


@dataclass
class Container:
    """Синглтоны процесса, собранные один раз на старте."""

    redis: redis.Redis
    vectors: VectorStore
    audit: AuditStore
    router: Router
    worker: GenerationWorker


container: Container | None = None


def build_container() -> Container:
    """Поднять все компоненты и создать индексы, если их ещё нет."""
    settings = get_settings()
    client = redis.from_url(settings.redis_url)

    vectors = VectorStore(client)
    vectors.ensure()

    audit = AuditStore()
    classifier = Classifier()

    return Container(
        redis=client,
        vectors=vectors,
        audit=audit,
        router=Router(client, classifier, vectors, audit),
        worker=GenerationWorker(client, vectors, audit, MockLLM()),
    )


def get_container() -> Container:
    """Доступ к собранным зависимостям из обработчиков."""
    if container is None:
        raise RuntimeError("контейнер зависимостей не собран")
    return container
