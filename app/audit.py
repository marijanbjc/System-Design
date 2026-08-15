"""Append-only audit log in SQLite, plus ticket persistence.

Every automated decision lands here with its scores, flags and model versions, so any
ticket's route can be reconstructed end to end. SQLite on the hot path is an explicit
simplification (single writer, lock risk under an incident spike) — see architecture.md 12.
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
    """Thin SQLite wrapper. No ORM, no migrations — deliberate for a PoC."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or get_settings().audit_db_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connect().executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def save_ticket(self, ticket: Ticket) -> None:
        """Upsert the whole ticket record."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO tickets (ticket_id, created_at, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(ticket_id) DO UPDATE SET payload = excluded.payload",
                (ticket.ticket_id, ticket.created_at.isoformat(), ticket.model_dump_json()),
            )

    def load_ticket(self, ticket_id: str) -> Ticket | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM tickets WHERE ticket_id = ?", (ticket_id,)
            ).fetchone()
        return Ticket.model_validate_json(row["payload"]) if row else None

    def log(self, ticket_id: str, event: str, **payload: Any) -> None:
        """Append one immutable step of the route."""
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
        """Full ordered route of a ticket — what a case review actually reads."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event, at, payload FROM audit_events WHERE ticket_id = ? ORDER BY id",
                (ticket_id,),
            ).fetchall()
        return [
            {"event": row["event"], "at": row["at"], **json.loads(row["payload"])} for row in rows
        ]
