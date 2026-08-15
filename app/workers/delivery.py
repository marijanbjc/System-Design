"""Воркер доставки: разносит готовые ответы обратно в каналы.

В PoC это мок — доставка пишется в лог. В целевой архитектуре здесь исходящие
адаптеры каналов: ответ письмом, сообщением в чат, пушем в приложение или вебхуком.
Отдельный воркер, а не вызов из генерации, потому что доставка — своя зона отказа:
почтовый шлюз может лежать, когда ответ уже готов.
"""

import logging
from datetime import UTC, datetime

import redis

from app.config import get_settings
from app.models import Ticket
from app.queues import streams
from app.storage.audit import AuditStore

logger = logging.getLogger(__name__)


class DeliveryWorker:
    """Разбирает очередь `stream:delivery`."""

    def __init__(self, client: redis.Redis, audit: AuditStore) -> None:
        self._redis = client
        self._audit = audit

    def run_once(self, batch: int = 10) -> int:
        """Доставить накопившиеся ответы; возвращает число доставленных тикетов."""
        tasks = streams.consume(self._redis, get_settings().stream_delivery, count=batch)
        for _, payload in tasks:
            ticket = self._audit.load_ticket(payload["ticket_id"])
            if ticket is not None:
                self.deliver(ticket)
        return len(tasks)

    def deliver(self, ticket: Ticket) -> Ticket:
        """Отправить ответ в канал обращения и зафиксировать время доставки."""
        logger.info(
            "[DELIVER via %s] ticket=%s answer=%s",
            ticket.channel,
            ticket.ticket_id,
            (ticket.answer_text or "")[:120],
        )
        ticket.delivered_at = datetime.now(UTC)
        self._audit.save_ticket(ticket)
        self._audit.log(
            ticket.ticket_id,
            "Delivered",
            channel=ticket.channel,
            answer_source=ticket.answer_source,
        )
        return ticket
