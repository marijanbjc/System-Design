"""Append-only аудит и хранение тикетов в SQLite.

Каждое автоматическое решение попадает сюда со скорами, флагами и версиями моделей,
чтобы маршрут любого обращения восстанавливался целиком. SQLite на горячем пути —
явное упрощение (один писатель, риск блокировок под пиком), см. architecture.md §12.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models import Ticket

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    event TEXT NOT NULL,
    at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_ticket ON audit_events(ticket_id);
"""


class AuditStore:
    """Тонкая обёртка над SQLite. Без ORM и без миграций — осознанно для PoC."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or get_settings().audit_db_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connect().executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def save_ticket(self, ticket: Ticket) -> None:
        """Сохранить или обновить запись тикета целиком."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tickets (ticket_id, created_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET payload = excluded.payload",
                (ticket.ticket_id, ticket.created_at.isoformat(), ticket.model_dump_json()),
            )

    def load_ticket(self, ticket_id: str) -> Ticket | None:
        """Прочитать тикет по идентификатору."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return Ticket.model_validate_json(row["payload"]) if row else None

    def log(self, ticket_id: str, event: str, **payload: Any) -> None:
        """Записать один неизменяемый шаг маршрута."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_events (ticket_id, event, at, payload) VALUES (?, ?, ?, ?)",
                (
                    ticket_id,
                    event,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def trail(self, ticket_id: str) -> list[dict[str, Any]]:
        """Полный упорядоченный маршрут тикета — то, что реально читают на разборе кейса."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event, at, payload FROM audit_events WHERE ticket_id = ? ORDER BY id",
                (ticket_id,),
            ).fetchall()
        return [
            {"event": row["event"], "at": row["at"], **json.loads(row["payload"])} for row in rows
        ]
