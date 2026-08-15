"""Офлайн-индексатор базы знаний: статья менеджера → вероятный вопрос → индекс.

Единственное место, где решается, что именно попадёт в вектор статьи. Ищем мы по
*вопросу в пользовательской формулировке*, а не по справочному тексту: пользователь
пишет вопрос, статья написана декларативно, и вопрос к вопросу совпадает заметно лучше.

Если менеджер приложил формулировку сам — берём её. Если нет — генерируем LLM офлайн.
На этот путь rate limiter намеренно не распространяется: запросы редкие, некритичные и
с онлайн-генерацией не конкурируют; при отказе статья индексируется по заголовку.
"""

import logging
import uuid

from app.llm.mock import MockLLM
from app.ml.encoder import embed
from app.storage.vector_index import VectorStore

logger = logging.getLogger(__name__)


class KbIndexer:
    """Приводит статью к виду, пригодному для поиска, и кладёт её в индекс."""

    def __init__(self, vectors: VectorStore, llm: MockLLM) -> None:
        self._vectors = vectors
        self._llm = llm

    def index_article(
        self,
        topic: str,
        title: str,
        body: str,
        likely_question: str | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Проиндексировать статью и вернуть её идентификатор."""
        article_id = doc_id or str(uuid.uuid4())
        question = likely_question or self._make_question(title, body)
        self._vectors.add_article(
            doc_id=article_id,
            topic=topic,
            title=title,
            body=body,
            likely_question=question,
            vector=embed(question),
        )
        return article_id

    def _make_question(self, title: str, body: str) -> str:
        """Сгенерировать вероятный вопрос гостя; при отказе модели откатиться к заголовку."""
        try:
            return self._llm.generate_likely_question(title, body)
        except Exception as exc:  # noqa: BLE001 — обогащение некритично, статья нужна в индексе
            logger.warning("не удалось сгенерировать вопрос к статье '%s': %s", title, exc)
            return title
