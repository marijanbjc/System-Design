"""FastAPI surface: intake, status, operator review and admin handles (architecture.md 10)."""

import uuid

import redis
from fastapi import APIRouter, FastAPI, HTTPException, Response
from pydantic import BaseModel

from app.audit import AuditStore
from app.classifier import load_classifier
from app.config import get_settings
from app.embed import embed
from app.llm.openai_client import build_llm
from app.models import AutomationLevel, IncomingMessage, Status, Ticket, TicketResponse
from app.router import Router
from app.store import state, vectors
from app.worker import GenerationWorker

router = APIRouter()


class Deps:
    """Process-wide singletons, wired once at startup."""

    redis: redis.Redis
    audit: AuditStore
    router: Router
    worker: GenerationWorker


deps = Deps()


class TripleIn(BaseModel):
    question: str
    topic: str
    answer: str


class ArticleIn(BaseModel):
    title: str
    body: str
    topic: str
    likely_question: str | None = None


class PolicyIn(BaseModel):
    topic: str
    level: AutomationLevel


class SurgeDefaultIn(BaseModel):
    topic: str
    text: str


class ReviewIn(BaseModel):
    operator_id: str
    action: str  # approved | edited | rejected
    answer_text: str | None = None


def _to_response(ticket: Ticket) -> TicketResponse:
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        topic=ticket.topic,
        risk=ticket.risk,
        route=ticket.route,
        status=ticket.status,
        answer=ticket.answer_text,
    )


@router.post("/tickets")
def create_ticket(message: IncomingMessage, response: Response) -> TicketResponse:
    """200 when the outcome is final and the answer is in the body, 202 otherwise."""
    ticket = deps.router.handle(message)
    response.status_code = 200 if ticket.answer_text else 202
    return _to_response(ticket)


@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str) -> TicketResponse:
    ticket = deps.audit.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return _to_response(ticket)


@router.get("/tickets/{ticket_id}/audit")
def get_audit(ticket_id: str) -> list[dict]:
    """Full route of the ticket — the artefact a case review actually reads."""
    return deps.audit.trail(ticket_id)


@router.post("/tickets/{ticket_id}/review")
def review_ticket(ticket_id: str, payload: ReviewIn) -> TicketResponse:
    """Operator decision. `draft_text` is kept so we can tell 'approved' from 'rewritten'."""
    ticket = deps.audit.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket not found")

    ticket.operator_id = payload.operator_id
    ticket.review_action = payload.action
    ticket.operator_touched = True
    if payload.action != "rejected":
        ticket.answer_text = payload.answer_text or ticket.draft_text
        ticket.answer_source = "operator"
        ticket.status = Status.ANSWERED
    deps.audit.save_ticket(ticket)
    deps.audit.log(ticket_id, "Reviewed", action=payload.action, operator_id=payload.operator_id)
    return _to_response(ticket)


@router.post("/kb/triples", status_code=201)
def add_triple(payload: TripleIn) -> dict[str, str]:
    doc_id = str(uuid.uuid4())
    vectors.add_triple(
        deps.redis, doc_id, payload.topic, payload.question, payload.answer, embed(payload.question)
    )
    return {"id": doc_id}


@router.post("/kb/articles", status_code=201)
def add_article(payload: ArticleIn) -> dict[str, str]:
    """Manager-supplied article. The likely guest question is indexed alongside the text:
    users ask questions while articles are declarative, and that bridge lifts recall.
    """
    doc_id = str(uuid.uuid4())
    indexed_text = f"{payload.likely_question or ''} {payload.title} {payload.body}".strip()
    vectors.add_article(
        deps.redis, doc_id, payload.topic, payload.title, payload.body, embed(indexed_text)
    )
    return {"id": doc_id}


@router.put("/admin/policy")
def set_policy(payload: PolicyIn) -> dict[str, str]:
    state.set_automation_level(deps.redis, payload.topic, payload.level)
    return {"topic": payload.topic, "level": payload.level}


@router.post("/admin/surge-default")
def set_surge_default(payload: SurgeDefaultIn) -> dict[str, str]:
    state.set_surge_text(deps.redis, payload.topic, payload.text)
    return {"topic": payload.topic}


@router.delete("/admin/surge-default/{topic}")
def delete_surge_default(topic: str) -> dict[str, str]:
    state.delete_surge_text(deps.redis, topic)
    return {"topic": topic}


@router.post("/admin/run-worker")
def run_worker() -> dict[str, int]:
    """Drains the generation queue on demand — stands in for a long-running worker."""
    return {"processed": deps.worker.run_once()}


def create_app() -> FastAPI:
    settings = get_settings()
    deps.redis = redis.from_url(settings.redis_url)
    deps.audit = AuditStore()
    vectors.ensure_indexes(deps.redis)
    classifier = load_classifier()
    deps.router = Router(deps.redis, classifier, deps.audit)
    deps.worker = GenerationWorker(deps.redis, deps.audit, build_llm())

    app = FastAPI(title="Support Ticket Bot PoC", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
