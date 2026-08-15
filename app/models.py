"""Pydantic schemas: API contract, ticket state, and the LLM structured output."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Channel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    WEB = "web"
    APP = "app"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Route(StrEnum):
    TIER1_AUTO = "tier1_auto"
    SURGE = "surge"
    TIER2_AUTO = "tier2_auto"
    TIER2_REVIEW = "tier2_review"
    TIER3_OPERATOR = "tier3_operator"


class Status(StrEnum):
    NEW = "new"
    ANSWERED = "answered"
    ANSWERED_SURGE = "answered_surge"
    PENDING_GENERATION = "pending_generation"
    PENDING_OPERATOR = "pending_operator"
    AWAITING_APPROVAL = "awaiting_approval"


class AutomationLevel(StrEnum):
    AUTO_OK = "AUTO_OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    OPERATOR_ONLY = "OPERATOR_ONLY"


class IncomingMessage(BaseModel):
    """Channel-agnostic inbound message produced by an input adapter."""

    channel: Channel
    text_raw: str = Field(min_length=1)
    user_ref: str | None = None
    channel_ref: str | None = None


class TicketResponse(BaseModel):
    """Uniform body for both 200 and 202 (architecture.md 4.1)."""

    ticket_id: str
    topic: str
    risk: Risk
    route: Route
    status: Status
    answer: str | None = None


class LLMDraft(BaseModel):
    """Structured LLM output. Deliberately carries no `topic` and no `risk`.

    Those come from our own classifier before the call, so a prompt injection has
    no lever to widen its own permissions. Every flag here can only tighten the
    route, never unlock auto-send.
    """

    answer_draft: str
    has_enough_context: float = Field(ge=0.0, le=1.0)
    is_on_topic: bool
    is_toxic: bool
    confidence: float = Field(ge=0.0, le=1.0)


class Ticket(BaseModel):
    """Full ticket record; the flag set makes the route reconstructable."""

    ticket_id: str
    user_ref: str | None = None
    channel: Channel
    channel_ref: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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

    draft_text: str | None = None
    answer_text: str | None = None
    answer_source: str | None = None

    operator_id: str | None = None
    review_action: str | None = None

    routed_at: datetime | None = None
    generated_at: datetime | None = None
    reviewed_at: datetime | None = None
    delivered_at: datetime | None = None
