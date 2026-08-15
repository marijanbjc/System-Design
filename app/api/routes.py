"""HTTP-поверхность: приём обращений, статус, ревью оператора, админ-ручки."""

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import get_container
from app.api.schemas import ArticleIn, PolicyIn, ReviewIn, SurgeDefaultIn, TripleIn
from app.config import get_settings
from app.ml.encoder import embed
from app.models import IncomingMessage, Status, Ticket, TicketResponse
from app.queues import streams
from app.storage import state

router = APIRouter()


def _to_response(ticket: Ticket) -> TicketResponse:
    """Свести тикет к единому телу ответа (architecture.md §4.1)."""
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        topic=ticket.topic,
        risk=ticket.risk,
        route=ticket.route,
        status=ticket.status,
        answer=ticket.answer_text,
    )


@router.post("/tickets", tags=["Приём"])
def create_ticket(message: IncomingMessage, response: Response) -> TicketResponse:
    """Принять обращение.

    200 — исход финальный и ответ лежит в теле; 202 — исхода пока нет, надо поллить
    `GET /tickets/{id}`. Кодами не различаем «генерируем» и «ушло оператору»: действие
    клиента одинаковое, а внутреннее состояние передаётся полем `status`.
    """
    ticket = get_container().router.handle(message)
    response.status_code = 200 if ticket.answer_text else 202
    return _to_response(ticket)


@router.get("/tickets/{ticket_id}", tags=["Приём"])
def get_ticket(ticket_id: str) -> TicketResponse:
    """Статус тикета и ответ, если он готов."""
    ticket = get_container().audit.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="тикет не найден")
    return _to_response(ticket)


@router.get("/tickets/{ticket_id}/audit", tags=["Приём"])
def get_audit(ticket_id: str) -> list[dict]:
    """Полный маршрут тикета — то, ради чего пишется аудит."""
    return get_container().audit.trail(ticket_id)


@router.post("/tickets/{ticket_id}/review", tags=["Оператор"])
def review_ticket(ticket_id: str, payload: ReviewIn) -> TicketResponse:
    """Решение оператора по черновику.

    `draft_text` сохраняется отдельно от `answer_text`: по этой паре видно, одобрил
    оператор черновик как есть или переписал его.
    """
    deps = get_container()
    ticket = deps.audit.load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="тикет не найден")

    ticket.operator_id = payload.operator_id
    ticket.review_action = payload.action
    ticket.operator_touched = True
    ticket.reviewed_at = datetime.now(UTC)
    if payload.action != "rejected":
        ticket.answer_text = payload.answer_text or ticket.draft_text
        ticket.answer_source = "operator"
        ticket.status = Status.ANSWERED
        streams.publish(
            deps.redis,
            get_settings().stream_delivery,
            {"ticket_id": ticket.ticket_id, "channel": ticket.channel},
        )
    deps.audit.save_ticket(ticket)
    deps.audit.log(ticket_id, "Reviewed", action=payload.action, operator_id=payload.operator_id)
    return _to_response(ticket)


@router.post("/kb/triples", status_code=201, tags=["Админ"])
def add_triple(payload: TripleIn) -> dict[str, str]:
    """Добавить предодобренную тройку в индекс типовых (ручка менеджера)."""
    deps = get_container()
    doc_id = str(uuid.uuid4())
    deps.vectors.add_triple(
        doc_id, payload.topic, payload.question, payload.answer, embed(payload.question)
    )
    return {"id": doc_id}


@router.post("/kb/articles", status_code=201, tags=["Админ"])
def add_article(payload: ArticleIn) -> dict[str, str]:
    """Добавить статью базы знаний.

    Поиск идёт по вероятному вопросу гостя, а не по тексту статьи. Если менеджер не
    приложил формулировку, индексатор сгенерирует её офлайн (см. `KbIndexer`).
    """
    doc_id = get_container().kb_indexer.index_article(
        topic=payload.topic,
        title=payload.title,
        body=payload.body,
        likely_question=payload.likely_question,
    )
    return {"id": doc_id}


@router.put("/admin/policy", tags=["Админ"])
def set_policy(payload: PolicyIn) -> dict[str, str]:
    """Задать уровень автоматизации темы."""
    state.set_automation_level(get_container().redis, payload.topic, payload.level)
    return {"topic": payload.topic, "level": payload.level}


@router.post("/admin/surge-default", tags=["Админ"])
def set_surge_default(payload: SurgeDefaultIn) -> dict[str, str]:
    """Задать заготовленный ответ на случай всплеска по теме."""
    state.set_surge_text(get_container().redis, payload.topic, payload.text)
    return {"topic": payload.topic}


@router.delete("/admin/surge-default/{topic}", tags=["Админ"])
def delete_surge_default(topic: str) -> dict[str, str]:
    """Убрать заглушку и запретить авто-ответ по теме при всплеске."""
    state.delete_surge_text(get_container().redis, topic)
    return {"topic": topic}


@router.get("/operator/queue", tags=["Оператор"])
def operator_queue() -> list[dict]:
    """Очередь задач оператора: что именно ждёт человека и по какой причине."""
    return streams.peek(get_container().redis, get_settings().stream_review)


@router.post("/admin/drain-queues", tags=["Админ"])
def drain_queues() -> dict[str, int]:
    """Разобрать очереди генерации и доставки.

    В PoC воркеры не крутятся отдельными процессами, а очереди настоящие: без этой
    ручки Tier 2 остался бы в `pending_generation` навсегда. В целевой архитектуре
    её место занимают долгоживущие воркеры.
    """
    deps = get_container()
    return {"generated": deps.worker.run_once(), "delivered": deps.delivery.run_once()}
