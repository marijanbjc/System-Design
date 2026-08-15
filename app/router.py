"""Hot path: normalize -> detect -> embed -> classify -> route. No external calls (architecture.md 8.1).

Priority is fixed and deterministic: safety -> surge -> triples -> generation. Because
nothing here leaves our perimeter, the 500 ms budget and graceful degradation hold by
construction rather than by hope.
"""

import uuid
from datetime import datetime, timezone

import redis

from app.audit import AuditStore
from app.classifier import Classifier
from app.config import get_settings
from app.embed import embed
from app.models import AutomationLevel, IncomingMessage, Risk, Route, Status, Ticket
from app.normalize import detect_injection, detect_unsafe, normalize
from app.store import queues, state, vectors


class Router:
    """Owns the synchronous decision. Async work is handed off through streams."""

    def __init__(self, client: redis.Redis, classifier: Classifier, audit: AuditStore) -> None:
        self._redis = client
        self._classifier = classifier
        self._audit = audit

    def handle(self, message: IncomingMessage) -> Ticket:
        settings = get_settings()
        ticket = self._register(message)

        ticket.text_normalized = normalize(message.text_raw)
        ticket.injection_suspected = detect_injection(ticket.text_normalized)
        ticket.unsafe_prefilter = detect_unsafe(ticket.text_normalized)

        vector = embed(ticket.text_normalized)
        prediction = self._classifier.predict(vector)
        ticket.topic = prediction.topic
        ticket.risk = prediction.risk
        ticket.conf_cls = prediction.conf_cls
        ticket.conf_risk = prediction.conf_risk

        level = state.automation_level(self._redis, ticket.topic)
        # Low classifier confidence means we do not trust the topic, so we must not
        # trust the policy row looked up by it either — fall back to the safe default.
        if ticket.conf_cls < settings.conf_cls_min:
            level = AutomationLevel.REVIEW_REQUIRED
        ticket.deny_listed = level is AutomationLevel.OPERATOR_ONLY

        self._audit.log(
            ticket.ticket_id,
            "Routed.classified",
            topic=ticket.topic,
            risk=ticket.risk,
            conf_cls=ticket.conf_cls,
            conf_risk=ticket.conf_risk,
            level=level,
            injection=ticket.injection_suspected,
            unsafe=ticket.unsafe_prefilter,
        )

        # 1) Sensitive topic, unsafe content, injection or high risk -> human only.
        if (
            level is AutomationLevel.OPERATOR_ONLY
            or ticket.unsafe_prefilter
            or ticket.injection_suspected
            or ticket.risk is Risk.HIGH
        ):
            return self._to_operator(ticket, reason="safety_or_policy")

        # 2) Mass surge on a topic that has a pre-written stub.
        state.bump_surge_counter(self._redis, ticket.topic)
        count = state.surge_count(self._redis, ticket.topic)
        stub = state.surge_text(self._redis, ticket.topic)
        if count >= settings.surge_threshold and stub:
            return self._surge_answer(ticket, stub, count)

        # 3) Typical request: close neighbour in the triples index, filtered by topic.
        hits = vectors.search_triples(self._redis, vector, ticket.topic)
        if hits:
            ticket.best_sim = hits[0].sim
            ticket.retrieved_triple_ids = [hit.doc_id for hit in hits]
        if hits and hits[0].sim >= settings.tau_high:
            if level is AutomationLevel.AUTO_OK and ticket.risk is Risk.LOW:
                return self._tier1_answer(ticket, hits[0].fields.get("answer", ""))
            # The triple exists but policy or risk forbids auto-send. Hand the
            # pre-approved answer to the operator as a draft instead of generating a
            # new one: it is better quality and costs no LLM call.
            return self._to_operator(ticket, reason="policy_review", draft=hits[0].fields.get("answer"))

        # 4) No close triple -> assemble an answer from the knowledge base, async.
        return self._enqueue_generation(ticket)

    # --- outcomes ---------------------------------------------------------------------

    def _register(self, message: IncomingMessage) -> Ticket:
        ticket = Ticket(
            ticket_id=str(uuid.uuid4()),
            channel=message.channel,
            channel_ref=message.channel_ref,
            user_ref=message.user_ref,
            text_raw=message.text_raw,
            encoder_version=get_settings().encoder_version,
            classifier_version=get_settings().classifier_version,
        )
        self._audit.save_ticket(ticket)
        self._audit.log(ticket.ticket_id, "TicketCreated", channel=message.channel)
        return ticket

    def _tier1_answer(self, ticket: Ticket, answer: str) -> Ticket:
        ticket.route = Route.TIER1_AUTO
        ticket.status = Status.ANSWERED
        ticket.is_typical = True
        ticket.auto_sent = True
        ticket.answer_text = answer
        ticket.answer_source = "retrieved"
        return self._finish(ticket, "Answered.tier1", best_sim=ticket.best_sim)

    def _surge_answer(self, ticket: Ticket, stub: str, count: int) -> Ticket:
        ticket.route = Route.SURGE
        ticket.status = Status.ANSWERED_SURGE
        ticket.surge = True
        ticket.incident_id = f"incident:{ticket.topic}"
        ticket.answer_text = stub
        ticket.answer_source = "surge"
        return self._finish(ticket, "Answered.surge", surge_count=count)

    def _to_operator(self, ticket: Ticket, reason: str, draft: str | None = None) -> Ticket:
        ticket.route = Route.TIER3_OPERATOR
        ticket.status = Status.PENDING_OPERATOR
        ticket.draft_text = draft
        queues.publish(
            self._redis,
            get_settings().stream_review,
            {"ticket_id": ticket.ticket_id, "reason": reason},
        )
        return self._finish(ticket, "Routed.operator", reason=reason, has_draft=draft is not None)

    def _enqueue_generation(self, ticket: Ticket) -> Ticket:
        ticket.route = Route.TIER2_REVIEW
        ticket.status = Status.PENDING_GENERATION
        queues.publish(
            self._redis, get_settings().stream_gen, {"ticket_id": ticket.ticket_id}
        )
        return self._finish(ticket, "Routed.generation", best_sim=ticket.best_sim)

    def _finish(self, ticket: Ticket, event: str, **payload: object) -> Ticket:
        ticket.routed_at = datetime.now(timezone.utc)
        self._audit.save_ticket(ticket)
        self._audit.log(
            ticket.ticket_id, event, route=ticket.route, status=ticket.status, **payload
        )
        return ticket
