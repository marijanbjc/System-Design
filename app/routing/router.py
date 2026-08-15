"""Горячий путь: нормализация → детекторы → эмбеддинг → классификация → маршрут.

Ни одного внешнего вызова. Именно поэтому бюджет 500 мс и корректная деградация
получаются по построению, а не «обычно выполняются».

Порядок веток фиксирован: безопасность → всплеск → тройки → генерация.
"""

import uuid
from datetime import UTC, datetime

import redis

from app.config import get_settings
from app.ml.classifier import Classifier
from app.ml.encoder import embed
from app.models import AutomationLevel, IncomingMessage, Risk, Route, Status, Ticket
from app.preprocessing.detectors import detect_injection, detect_unsafe
from app.preprocessing.normalizer import normalize
from app.queues import streams
from app.storage import state
from app.storage.audit import AuditStore
from app.storage.vector_index import VectorStore


class Router:
    """Владеет синхронным решением. Всё медленное уходит через очереди."""

    def __init__(
        self,
        client: redis.Redis,
        classifier: Classifier,
        vectors: VectorStore,
        audit: AuditStore,
    ) -> None:
        self._redis = client
        self._classifier = classifier
        self._vectors = vectors
        self._audit = audit

    def handle(self, message: IncomingMessage) -> Ticket:
        """Принять обращение, создать тикет и выбрать маршрут."""
        settings = get_settings()
        ticket = self._register(message)

        ticket.text_normalized = normalize(message.text_raw)
        ticket.injection_suspected = detect_injection(ticket.text_normalized)
        ticket.unsafe_prefilter = detect_unsafe(ticket.text_normalized)

        prediction = self._classifier.predict(ticket.text_normalized)
        ticket.topic = prediction.topic
        ticket.risk = prediction.risk
        ticket.conf_cls = prediction.conf_cls

        level = state.automation_level(self._redis, ticket.topic)
        # Низкая уверенность классификатора означает, что теме мы не доверяем, — значит
        # нельзя доверять и строке политики, найденной по этой теме. Безопасный дефолт.
        if ticket.conf_cls < settings.conf_cls_min:
            level = AutomationLevel.REVIEW_REQUIRED
        ticket.deny_listed = level is AutomationLevel.OPERATOR_ONLY

        self._audit.log(
            ticket.ticket_id,
            "Routed.classified",
            topic=ticket.topic,
            risk=ticket.risk,
            conf_cls=ticket.conf_cls,
            level=level,
            injection=ticket.injection_suspected,
            unsafe=ticket.unsafe_prefilter,
        )

        # 1) Чувствительная тема, небезопасно, инъекция или высокий риск — только человек.
        if (
            level is AutomationLevel.OPERATOR_ONLY
            or ticket.unsafe_prefilter
            or ticket.injection_suspected
            or ticket.risk is Risk.HIGH
        ):
            return self._to_operator(ticket, reason="safety_or_policy")

        # 2) Массовый всплеск по теме, для которой заготовлена заглушка. Проверяется
        # до индекса троек: во время аварии штатный FAQ-ответ не просто бесполезен,
        # он вреден («проверьте баланс карты», когда лежит платёжный шлюз).
        state.bump_surge_counter(self._redis, ticket.topic)
        count = state.surge_count(self._redis, ticket.topic)
        stub = state.surge_text(self._redis, ticket.topic)
        if count >= settings.surge_threshold and stub:
            return self._surge_answer(ticket, stub, count)

        # 3) Типовой вопрос: близкий кандидат в индексе троек, поиск с фильтром по теме.
        hits = self._vectors.search_triples(embed(ticket.text_normalized), ticket.topic)
        if hits:
            ticket.best_sim = hits[0].sim
            ticket.retrieved_triple_ids = [hit.doc_id for hit in hits]
        if hits and hits[0].sim >= settings.tau_high:
            if level is AutomationLevel.AUTO_OK and ticket.risk is Risk.LOW:
                return self._tier1_answer(ticket, hits[0].fields.get("answer", ""))
            # Тройка есть, но тема или риск не разрешают авто-отправку. Отдаём найденный
            # предодобренный ответ оператору как черновик: генерировать заново незачем,
            # он качественнее и не стоит вызова LLM.
            return self._to_operator(
                ticket, reason="policy_review", draft=hits[0].fields.get("answer")
            )

        # 4) Похожей тройки нет — собираем ответ из базы знаний асинхронно.
        return self._enqueue_generation(ticket)

    # --- исходы ---------------------------------------------------------------------------

    def _register(self, message: IncomingMessage) -> Ticket:
        """Создать тикет до любой обработки: SLA закрывается именно здесь."""
        settings = get_settings()
        ticket = Ticket(
            ticket_id=str(uuid.uuid4()),
            channel=message.channel,
            channel_ref=message.channel_ref,
            user_ref=message.user_ref,
            text_raw=message.text_raw,
            encoder_version=settings.encoder_version,
            classifier_version=settings.classifier_version,
        )
        self._audit.save_ticket(ticket)
        self._audit.log(ticket.ticket_id, "TicketCreated", channel=message.channel)
        return ticket

    def _tier1_answer(self, ticket: Ticket, answer: str) -> Ticket:
        """Tier 1: отдаём предодобренный ответ как есть, без LLM и без оператора."""
        ticket.route = Route.TIER1_AUTO
        ticket.status = Status.ANSWERED
        ticket.auto_sent = True
        ticket.answer_text = answer
        ticket.answer_source = "retrieved"
        return self._finish(ticket, "Answered.tier1", best_sim=ticket.best_sim)

    def _surge_answer(self, ticket: Ticket, stub: str, count: int) -> Ticket:
        """Инцидентная заглушка. Отдельный статус, чтобы не смешивать с автоматизацией."""
        ticket.route = Route.SURGE
        ticket.status = Status.ANSWERED_SURGE
        ticket.surge = True
        ticket.incident_id = f"incident:{ticket.topic}"
        ticket.answer_text = stub
        ticket.answer_source = "surge"
        return self._finish(ticket, "Answered.surge", surge_count=count)

    def _to_operator(self, ticket: Ticket, reason: str, draft: str | None = None) -> Ticket:
        """Tier 3: человек. LLM на этой ветке не вызывается вовсе."""
        ticket.route = Route.TIER3_OPERATOR
        ticket.status = Status.PENDING_OPERATOR
        ticket.draft_text = draft
        streams.publish(
            self._redis,
            get_settings().stream_review,
            {"ticket_id": ticket.ticket_id, "reason": reason},
        )
        return self._finish(ticket, "Routed.operator", reason=reason, has_draft=draft is not None)

    def _enqueue_generation(self, ticket: Ticket) -> Ticket:
        """Tier 2: задача в очередь генерации, пользователю — 202."""
        ticket.route = Route.TIER2_REVIEW
        ticket.status = Status.PENDING_GENERATION
        streams.publish(self._redis, get_settings().stream_gen, {"ticket_id": ticket.ticket_id})
        return self._finish(ticket, "Routed.generation", best_sim=ticket.best_sim)

    def _finish(self, ticket: Ticket, event: str, **payload: object) -> Ticket:
        """Зафиксировать исход, а готовый ответ поставить в очередь доставки."""
        ticket.routed_at = datetime.now(UTC)
        self._audit.save_ticket(ticket)
        self._audit.log(
            ticket.ticket_id, event, route=ticket.route, status=ticket.status, **payload
        )
        if ticket.status in (Status.ANSWERED, Status.ANSWERED_SURGE):
            streams.publish(
                self._redis,
                get_settings().stream_delivery,
                {"ticket_id": ticket.ticket_id, "channel": ticket.channel},
            )
        return ticket
