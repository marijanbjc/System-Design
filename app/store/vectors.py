"""Redis Stack vector indexes: triples (typical Q/A) and knowledge-base articles."""

from dataclasses import dataclass

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import get_settings

TRIPLES_INDEX = "idx:triples"
TRIPLES_PREFIX = "triple:"
KB_INDEX = "idx:kb"
KB_PREFIX = "kb:"


@dataclass(frozen=True)
class Hit:
    doc_id: str
    sim: float
    fields: dict[str, str]


def _vector_field(name: str = "embedding") -> VectorField:
    settings = get_settings()
    return VectorField(
        name,
        "FLAT",
        {"TYPE": "FLOAT32", "DIM": settings.embed_dim, "DISTANCE_METRIC": "COSINE"},
    )


def ensure_indexes(client: redis.Redis) -> None:
    """Create both indexes if they do not exist yet. Idempotent."""
    _ensure(
        client,
        TRIPLES_INDEX,
        TRIPLES_PREFIX,
        [TagField("topic"), TextField("question"), TextField("answer"), _vector_field()],
    )
    _ensure(
        client,
        KB_INDEX,
        KB_PREFIX,
        [TagField("topic"), TextField("title"), TextField("body"), _vector_field()],
    )


def _ensure(client: redis.Redis, index: str, prefix: str, fields: list) -> None:
    try:
        client.ft(index).info()
    except redis.ResponseError:
        client.ft(index).create_index(
            fields=fields,
            definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
        )


def add_triple(
    client: redis.Redis, doc_id: str, topic: str, question: str, answer: str, vector: np.ndarray
) -> None:
    """Store a pre-approved (question, topic, answer) triple."""
    client.hset(
        f"{TRIPLES_PREFIX}{doc_id}",
        mapping={
            "topic": topic,
            "question": question,
            "answer": answer,
            "embedding": vector.astype(np.float32).tobytes(),
        },
    )


def add_article(
    client: redis.Redis, doc_id: str, topic: str, title: str, body: str, vector: np.ndarray
) -> None:
    """Store a knowledge-base article supplied by a manager."""
    client.hset(
        f"{KB_PREFIX}{doc_id}",
        mapping={
            "topic": topic,
            "title": title,
            "body": body,
            "embedding": vector.astype(np.float32).tobytes(),
        },
    )


def search_triples(
    client: redis.Redis, vector: np.ndarray, topic: str, limit: int = 3
) -> list[Hit]:
    """KNN over triples, filtered by the topic the classifier already decided.

    The filter is what keeps two lexically close but semantically different requests
    ("cancel an order" vs "cancel a subscription") from matching each other. If the
    classifier got the topic wrong the search simply returns nothing and we fall
    through to generation — a fail-safe direction.
    """
    return _knn(client, TRIPLES_INDEX, vector, ["topic", "question", "answer"], limit, topic)


def search_kb(client: redis.Redis, vector: np.ndarray, limit: int = 3) -> list[Hit]:
    """KNN over knowledge-base articles. No topic filter: context may live anywhere."""
    return _knn(client, KB_INDEX, vector, ["topic", "title", "body"], limit, None)


def _knn(
    client: redis.Redis,
    index: str,
    vector: np.ndarray,
    return_fields: list[str],
    limit: int,
    topic: str | None,
) -> list[Hit]:
    prefilter = f"(@topic:{{{_escape(topic)}}})" if topic else "*"
    query = (
        Query(f"{prefilter}=>[KNN {limit} @embedding $vec AS dist]")
        .sort_by("dist")
        .return_fields("dist", *return_fields)
        .dialect(2)
    )
    result = client.ft(index).search(query, query_params={"vec": vector.astype(np.float32).tobytes()})
    hits = []
    for doc in result.docs:
        fields = {name: getattr(doc, name, "") for name in return_fields}
        hits.append(Hit(doc_id=doc.id, sim=1.0 - float(doc.dist), fields=fields))
    return hits


def _escape(value: str) -> str:
    return value.replace("-", r"\-").replace(" ", r"\ ")
