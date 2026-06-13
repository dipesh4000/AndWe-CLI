"""
Async SQLite conversation store for KuroCode.

Uses ``aiosqlite`` with ``PRAGMA user_version`` migrations.
DB lives at ``~/.local/share/kurocode/history.db`` by default
(override via ``KUROCODE_DB_PATH`` env var or ``db_path`` TOML key).

Usage::

    from kurocode.infra.store import ConversationStore

    async with ConversationStore() as store:
        cid = await store.create_conversation("Hello session", "gpt-4o-mini")
        await store.add_message(cid, "user", "Hi!")
        msgs = await store.get_messages(cid)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import NamedTuple

import aiosqlite

from kurocode.exceptions import StoreError

_DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "kurocode" / "history.db"

# ---------------------------------------------------------------------------
# Public row types
# ---------------------------------------------------------------------------


class ConversationRow(NamedTuple):
    """A single row from the ``conversations`` table."""

    id: str
    title: str
    model: str
    created_at: int  # unix timestamp


class MessageRow(NamedTuple):
    """A single row from the ``messages`` table."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: int  # unix timestamp


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

# Each entry is the SQL to migrate from (index) → (index + 1).
# Never edit existing entries — append new ones instead.
_MIGRATIONS: list[str] = [
    # v0 → v1: initial schema
    """
    CREATE TABLE conversations (
        id         TEXT    PRIMARY KEY,
        title      TEXT    NOT NULL,
        model      TEXT    NOT NULL,
        created_at INTEGER NOT NULL
    );

    CREATE TABLE messages (
        id              TEXT    PRIMARY KEY,
        conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT    NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
        content         TEXT    NOT NULL,
        created_at      INTEGER NOT NULL
    );

    CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);
    """,
]

_TARGET_VERSION: int = len(_MIGRATIONS)


async def _migrate(conn: aiosqlite.Connection) -> None:
    """Apply any pending migrations to *conn*, bumping ``PRAGMA user_version``."""
    async with conn.execute("PRAGMA user_version") as cur:
        row = await cur.fetchone()
    current: int = row[0] if row else 0

    for idx, sql in enumerate(_MIGRATIONS):
        version = idx + 1
        if current >= version:
            continue
        # executescript issues an implicit COMMIT before running.
        await conn.executescript(sql)
        await conn.execute(f"PRAGMA user_version = {version}")
        await conn.commit()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ConversationStore:
    """
    Async context manager that wraps an ``aiosqlite`` connection.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.  Parent directories are
        created automatically on first open.
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "ConversationStore":
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
            await _migrate(conn)
            self._conn = conn
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(
                f"Failed to open database at {self._db_path}.",
                hint=str(exc),
            ) from exc
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise StoreError(
                "ConversationStore is not open.",
                hint="Use 'async with ConversationStore() as store: ...'",
            )
        return self._conn

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def create_conversation(self, title: str, model: str) -> str:
        """
        Insert a new conversation row.

        Returns
        -------
        str
            The UUID assigned to the new conversation.
        """
        conv_id = str(uuid.uuid4())
        try:
            await self._db.execute(
                "INSERT INTO conversations (id, title, model, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, model, int(time.time())),
            )
            await self._db.commit()
        except Exception as exc:
            raise StoreError(
                "Failed to create conversation.", hint=str(exc)
            ) from exc
        return conv_id

    async def list_conversations(self, limit: int = 50) -> list[ConversationRow]:
        """Return the *limit* most-recent conversations (newest first)."""
        try:
            async with self._db.execute(
                "SELECT id, title, model, created_at"
                "  FROM conversations"
                " ORDER BY created_at DESC"
                " LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as exc:
            raise StoreError(
                "Failed to list conversations.", hint=str(exc)
            ) from exc
        return [ConversationRow(*tuple(row)) for row in rows]

    async def delete_conversation(self, conv_id: str) -> None:
        """Delete *conv_id* and all its messages (ON DELETE CASCADE)."""
        try:
            await self._db.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            await self._db.commit()
        except Exception as exc:
            raise StoreError(
                f"Failed to delete conversation '{conv_id}'.", hint=str(exc)
            ) from exc

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def get_messages(self, conv_id: str) -> list[MessageRow]:
        """Return all messages for *conv_id* in chronological order."""
        try:
            async with self._db.execute(
                "SELECT id, conversation_id, role, content, created_at"
                "  FROM messages"
                " WHERE conversation_id = ?"
                " ORDER BY created_at",
                (conv_id,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as exc:
            raise StoreError(
                f"Failed to fetch messages for conversation '{conv_id}'.",
                hint=str(exc),
            ) from exc
        return [MessageRow(*tuple(row)) for row in rows]

    async def add_message(
        self, conv_id: str, role: str, content: str
    ) -> str:
        """
        Append a message to *conv_id*.

        Parameters
        ----------
        conv_id:
            Target conversation UUID.
        role:
            One of ``"system"``, ``"user"``, or ``"assistant"``.
        content:
            Message text.

        Returns
        -------
        str
            The UUID assigned to the new message.
        """
        msg_id = str(uuid.uuid4())
        try:
            await self._db.execute(
                "INSERT INTO messages"
                "  (id, conversation_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (msg_id, conv_id, role, content, int(time.time())),
            )
            await self._db.commit()
        except Exception as exc:
            raise StoreError("Failed to add message.", hint=str(exc)) from exc
        return msg_id
