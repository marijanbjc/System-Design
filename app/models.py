"""Модели предметной области: контракт API, запись тикета и структурированный выход LLM."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Channel(StrEnum):
    """Канал, из которого пришло обращение."""

    EMAIL = "email"
    CHAT = "chat"
    WEB = "web"
    APP = "app"


class Risk(StrEnum):
    """Уровень риска обращения: определяется классификатором и ужесточается правилами."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Route(StrEnum):
    """Выбранная роутером ветвь обработки."""

    TIER1_AUTO = "tier1_auto"
    SURGE = "surge"
    TIER2_AUTO = "tier2_auto"
    TIER2_REVIEW = "tier2_review"
    TIER3_OPERATOR = "tier3_operator"


class Status(StrEnum):
    """Состояние тикета.

    `answered_surge` вынесен отдельно осознанно: это не «мы решили вопрос», а «мы
    отписались заготовленным текстом по известной аварии». Смешивать нельзя — иначе
    метрика автоматизации во время сбоя вырастет по причине, не связанной с качеством.
    """

    NEW = "new"
    ANSWERED = "answered"
    ANSWERED_SURGE = "answered_surge"
    PENDING_GENERATION = "pending_generation"
    PENDING_OPERATOR = "pending_operator"
    AWAITING_APPROVAL = "awaiting_approval"


class AutomationLevel(StrEnum):
    """Детерминированная политика по теме — жёсткая гарантия поверх вероятностной модели."""

    AUTO_OK = "AUTO_OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    OPERATOR_ONLY = "OPERATOR_ONLY"


class IncomingMessage(BaseModel):
    """Единый вид обращения, к которому входной адаптер приводит сообщение любого канала."""

    channel: Channel
    text_raw: str = Field(min_length=1)
    user_ref: str | None = None
    channel_ref: str | None = None


class TicketResponse(BaseModel):
    """Единое тело ответа и для 200, и для 202 (architecture.md §4.1)."""

    ticket_id: str
    topic: str
    risk: Risk
    route: Route
    status: Status
    answer: str | None = None


class LLMDraft(BaseModel):
    """Структурированный выход LLM.

    В схеме намеренно нет полей `topic` и `risk`: их определяет наш классификатор до
    вызова модели, поэтому у prompt injection нет рычага расширить собственные права.
    Все флаги ниже работают только в сторону ужесточения маршрута — увести тикет к
    человеку они могут, разрешить авто-отправку нет.
    """

    answer_draft: str
    has_enough_context: float = Field(ge=0.0, le=1.0)
    is_on_topic: bool
    is_toxic: bool
    confidence: float = Field(ge=0.0, le=1.0)


class Ticket(BaseModel):
    """Полная запись тикета: по набору полей и флагов маршрут восстанавливается целиком."""

    ticket_id: str
    user_ref: str | None = None
    channel: Channel
    channel_ref: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Три представления текста, которые нельзя путать: сырой хранится у нас (это наш
    # контур), нормализованный идёт в модели, обезличенный существует только на выходе.
    text_raw: str
    text_normalized: str = ""

    topic: str = "unknown"
    risk: Risk = Risk.MEDIUM
    conf_cls: float = 0.0
    conf_risk: float = 0.0

    route: Route | None = None
    status: Status = Status.NEW
    is_typical: bool = False
    auto_sent: bool = False
    operator_touched: bool = False
    deny_listed: bool = False
    surge: bool = False
    injection_suspected: bool = False
    unsafe_prefilter: bool = False
    # Группирует тикеты одного сбоя: отдельная корзина в метриках и, в целевой картине,
    # массовая рассылка при резолве инцидента одной операцией.
    incident_id: str | None = None

    best_sim: float = 0.0
    best_chunk_sim: float = 0.0
    retrieved_triple_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    encoder_version: str = ""
    classifier_version: str = ""
    llm_model: str | None = None
    prompt_version: str | None = None
    conf_gen: float | None = None
    llm_flags: dict[str, Any] = Field(default_factory=dict)
    tokens: int = 0
    cost: float = 0.0

    # draft_text отдельно от answer_text: без этой пары не отличить «оператор одобрил
    # как есть» от «оператор переписал» — это и метрика качества черновиков, и
    # обучающий сигнал для будущей джобы пополнения индекса троек.
    draft_text: str | None = None
    answer_text: str | None = None
    answer_source: str | None = None

    operator_id: str | None = None
    review_action: str | None = None

    routed_at: datetime | None = None
    generated_at: datetime | None = None
    reviewed_at: datetime | None = None
    delivered_at: datetime | None = None
