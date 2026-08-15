"""Схемы запросов административных ручек и ревью оператора."""

from pydantic import BaseModel

from app.models import AutomationLevel


class TripleIn(BaseModel):
    """Тройка (обращение, тема, ответ) для индекса типовых."""

    question: str
    topic: str
    answer: str


class ArticleIn(BaseModel):
    """Статья базы знаний от менеджера. Парсер вики и чанкер — в целевой архитектуре."""

    title: str
    body: str
    topic: str
    likely_question: str | None = None


class PolicyIn(BaseModel):
    """Уровень автоматизации темы."""

    topic: str
    level: AutomationLevel


class SurgeDefaultIn(BaseModel):
    """Заготовленный ответ на случай всплеска по теме."""

    topic: str
    text: str


class ReviewIn(BaseModel):
    """Решение оператора по черновику: approved | edited | rejected."""

    operator_id: str
    action: str
    answer_text: str | None = None
