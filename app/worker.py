"""Generation worker: KB search -> pre-gate -> scrub -> LLM -> rehydrate -> gate.

Everything slow and everything external lives here. A failure at any point degrades
the ticket to an operator; it never propagates back to the hot path.
"""

from datetime import datetime, timezone

import redis

from app.audit import AuditStore
from app.config import get_settings
from app.embed import embed
from app.llm.base import LLMUnavailable
from app.models import AutomationLevel, Risk, Route, Status, Ticket
from app.pii import rehydrate, scrub
from app.store import queues, state, vectors
from app.store.limiter import BudgetBreaker, RateLimiter


class GenerationWorker:
    """Drains `stream:gen` one batch at a time."""

    def __init__(self, client: redis.Redis, audit: AuditStore, llm) -> None:
        self._redis = client
        self._audit = audit
        self._llm = llm
        self._limiter = RateLimiter(client)
        self._budget = BudgetBreaker(client)

    def run_once(self, batch: int = 10) -> int:
        """Process pending tasks; returns how many tickets were handled."""
        settings = get_settings()
        tasks = queues.consume(self._redis, settings.stream_gen, count=batch)
        for _, payload in tasks:
            ticket = self._audit.load_ticket(payload["ticket_id"])
            if ticket is not None:
                self.process(ticket)
        return len(tasks)

    def process(self, ticket: Ticket) -> Ticket:
        settings = get_settings()
        vector = embed(ticket.text_normalized)
        chunks = vectors.search_kb(self._redis, vector)
        if chunks:
            ticket.best_chunk_sim = chunks[0].sim
            ticket.retrieved_chunk_ids = [hit.doc_id for hit in chunks]

        # Pre-gate: with no context there is nothing to generate from, so we do not
        # spend an LLM call at all.
        if not chunks or chunks[0].sim < settings.tau_kb:
            return self._to_review(ticket, reason="pre_gate_no_context")

        if self._budget.is_open():
            return self._to_review(ticket, reason="budget_breaker_open")

        # The rate limiter protects the provider; a task waits, it is never dropped.
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
        return self._gate(ticket, draft)

    def _gate(self, ticket: Ticket, draft) -> Ticket:
        """Local conditions are checked before the model's flags, and the model's flags
        can only tighten the route. An injected 'everything is fine' cannot unlock
        auto-send on a risky topic.
        """
        settings = get_settings()
        level = state.automation_level(self._redis, ticket.topic)

        if (
            ticket.injection_suspected
            or ticket.unsafe_prefilter
            or level is not AutomationLevel.AUTO_OK
            or ticket.risk is not Risk.LOW
        ):
            return self._to_review(ticket, reason="policy_or_risk")
        if draft.is_toxic or not draft.is_on_topic:
            return self._to_review(ticket, reason="llm_safety_flag")
        if draft.has_enough_context < settings.tau_ctx:
            return self._to_review(ticket, reason="low_context_score")
        if draft.confidence < settings.tau_conf:
            return self._to_review(ticket, reason="low_confidence")

        ticket.route = Route.TIER2_AUTO
        ticket.status = Status.ANSWERED
        ticket.auto_sent = True
        ticket.answer_text = ticket.draft_text
        ticket.answer_source = "generated"
        return self._finish(ticket, "Gated.auto_send")

    def _to_review(self, ticket: Ticket, reason: str) -> Ticket:
        ticket.route = Route.TIER2_REVIEW
        ticket.status = Status.AWAITING_APPROVAL if ticket.draft_text else Status.PENDING_OPERATOR
        queues.publish(
            self._redis,
            get_settings().stream_review,
            {"ticket_id": ticket.ticket_id, "reason": reason},
        )
        return self._finish(ticket, "Gated.to_operator", reason=reason)

    def _finish(self, ticket: Ticket, event: str, **payload: object) -> Ticket:
        self._audit.save_ticket(ticket)
        self._audit.log(
            ticket.ticket_id, event, route=ticket.route, status=ticket.status, **payload
        )
        if ticket.status is Status.ANSWERED:
            queues.publish(
                self._redis,
                get_settings().stream_delivery,
                {"ticket_id": ticket.ticket_id, "channel": ticket.channel},
            )
        return ticket
