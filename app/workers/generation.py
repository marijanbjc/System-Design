"""Воркер генерации: поиск по БЗ → пре-гейт → обезличивание → LLM → регидрация → гейт.

Здесь живёт всё медленное и всё внешнее. Отказ на любом шаге деградирует тикет в
оператора и никогда не прорастает обратно в горячий путь.
"""

from datetime import datetime, timezone

import redis

from app.config import get_settings
from app.llm.base import LLMClient, LLMUnavailable
from app.models import Route, Status, Ticket
from app.ml.encoder import embed
from app.preprocessing.pii import rehydrate, scrub
from app.queues import streams
from app.queues.limiter import BudgetBreaker, RateLimiter
from app.routing.gates import post_gate, pre_gate
from app.storage import state
from app.storage.audit import AuditStore
from app.storage.vector_index import VectorStore


class GenerationWorker:
    """Разбирает очередь `stream:gen` порциями."""

    def __init__(
        self,
        client: redis.Redis,
        vectors: VectorStore,
        audit: AuditStore,
        llm: LLMClient,
    ) -> None:
        self._redis = client
        self._vectors = vectors
        self._audit = audit
        self._llm = llm
        self._limiter = RateLimiter(client)
        self._budget = BudgetBreaker(client)

    def run_once(self, batch: int = 10) -> int:
        """Обработать накопившиеся задачи; возвращает число обработанных тикетов."""
        tasks = streams.consume(self._redis, get_settings().stream_gen, count=batch)
        for _, payload in tasks:
            ticket = self._audit.load_ticket(payload["ticket_id"])
            if ticket is not None:
                self.process(ticket)
        return len(tasks)

    def process(self, ticket: Ticket) -> Ticket:
        """Полный асинхронный путь одного тикета."""
        settings = get_settings()
        vector = embed(ticket.text_normalized)
        chunks = self._vectors.search_kb(vector)
        if chunks:
            ticket.best_chunk_sim = chunks[0].sim
            ticket.retrieved_chunk_ids = [hit.doc_id for hit in chunks]

        decision = pre_gate(chunks)
        if not decision.auto_send:
            return self._to_review(ticket, reason=decision.reason)

        if self._budget.is_open():
            return self._to_review(ticket, reason="budget_breaker_open")

        # Ограничитель защищает провайдера: задача ждёт разрешения, но не теряется.
        if not self._limiter.acquire(timeout=settings.llm_timeout_seconds):
            return self._to_review(ticket, reason="rate_limit_timeout")

        scrubbed, pii_map = scrub(ticket.text_normalized)
        state.store_pii_map(self._redis, ticket.ticket_id, pii_map)
        self._audit.log(
            ticket.ticket_id, "Generated.scrubbed", placeholders=sorted(pii_map.keys())
        )

        context = [hit.fields.get("body", "") for hit in chunks]
        try:
            draft, tokens = self._llm.generate(scrubbed, context)
        except LLMUnavailable as exc:
            self._audit.log(ticket.ticket_id, "Generated.failed", error=str(exc))
            return self._to_review(ticket, reason="llm_unavailable")

        self._budget.charge(tokens)
        ticket.tokens = tokens
        ticket.cost = round(tokens * 1e-5, 6)
        ticket.llm_model = getattr(self._llm, "model_name", "unknown")
        ticket.prompt_version = settings.llm_prompt_version
        ticket.conf_gen = draft.confidence
        ticket.llm_flags = draft.model_dump(exclude={"answer_draft"})
        ticket.draft_text = rehydrate(draft.answer_draft, pii_map)
        ticket.generated_at = datetime.now(timezone.utc)

        self._audit.log(
            ticket.ticket_id,
            "Generated.done",
            tokens=tokens,
            best_chunk_sim=ticket.best_chunk_sim,
            **ticket.llm_flags,
        )

        level = state.automation_level(self._redis, ticket.topic)
        verdict = post_gate(
            level, ticket.risk, draft, ticket.injection_suspected, ticket.unsafe_prefilter
        )
        if not verdict.auto_send:
            return self._to_review(ticket, reason=verdict.reason)
        return self._auto_send(ticket)

    def _auto_send(self, ticket: Ticket) -> Ticket:
        """Гейт пройден: ответ уходит пользователю напрямую."""
        ticket.route = Route.TIER2_AUTO
        ticket.status = Status.ANSWERED
        ticket.auto_sent = True
        ticket.answer_text = ticket.draft_text
        ticket.answer_source = "generated"
        return self._finish(ticket, "Gated.auto_send")

    def _to_review(self, ticket: Ticket, reason: str) -> Ticket:
        """Гейт не пропустил — тикет уходит человеку, при наличии черновика вместе с ним."""
        ticket.route = Route.TIER2_REVIEW
        ticket.status = Status.AWAITING_APPROVAL if ticket.draft_text else Status.PENDING_OPERATOR
        streams.publish(
            self._redis,
            get_settings().stream_review,
            {"ticket_id": ticket.ticket_id, "reason": reason},
        )
        return self._finish(ticket, "Gated.to_operator", reason=reason)

    def _finish(self, ticket: Ticket, event: str, **payload: object) -> Ticket:
        """Сохранить исход, записать аудит и, если ответ готов, поставить его в доставку."""
        self._audit.save_ticket(ticket)
        self._audit.log(
            ticket.ticket_id, event, route=ticket.route, status=ticket.status, **payload
        )
        if ticket.status is Status.ANSWERED:
            streams.publish(
                self._redis,
                get_settings().stream_delivery,
                {"ticket_id": ticket.ticket_id, "channel": ticket.channel},
            )
        return ticket
