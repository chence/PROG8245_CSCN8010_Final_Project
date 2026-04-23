from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import get_config
from src.utils import ensure_parent, new_session_id, utc_timestamp


class ChatDatabase:
    def __init__(self, db_path: str | Path | None = None) -> None:
        config = get_config()
        self.db_path = ensure_parent(db_path or config.database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    language TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    original_text TEXT NOT NULL,
                    english_text TEXT,
                    language TEXT,
                    intent TEXT,
                    confidence REAL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                """
            )

    def create_session(self, language: str | None = None) -> str:
        session_id = new_session_id()
        now = utc_timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, language) VALUES (?, ?, ?, ?)",
                (session_id, now, now, language),
            )
        return session_id

    def ensure_session(self, session_id: str | None, language: str | None = None) -> str:
        if session_id:
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing:
                    return session_id
        return self.create_session(language=language)

    def log_message(
        self,
        session_id: str,
        role: str,
        original_text: str,
        english_text: str | None = None,
        language: str | None = None,
        intent: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_timestamp()
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, original_text, english_text, language, intent, confidence,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    original_text,
                    english_text,
                    language,
                    intent,
                    confidence,
                    payload,
                    now,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ?, language = COALESCE(?, language) WHERE session_id = ?",
                (now, language, session_id),
            )

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, original_text, english_text, language, intent, confidence, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in reversed(rows):
            results.append(
                {
                    "role": row["role"],
                    "original_text": row["original_text"],
                    "english_text": row["english_text"],
                    "language": row["language"],
                    "intent": row["intent"],
                    "confidence": row["confidence"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return results

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, original_text, english_text, language, intent, confidence, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "role": row["role"],
                    "original_text": row["original_text"],
                    "english_text": row["english_text"],
                    "language": row["language"],
                    "intent": row["intent"],
                    "confidence": row["confidence"],
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                    "created_at": row["created_at"],
                }
            )
        return results

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    s.session_id,
                    s.created_at,
                    s.updated_at,
                    s.language,
                    (
                        SELECT m.original_text
                        FROM messages m
                        WHERE m.session_id = s.session_id
                        AND m.role = 'user'
                        ORDER BY m.id ASC
                        LIMIT 1
                    ) AS first_user_message,
                    (
                        SELECT COUNT(*)
                        FROM messages m
                        WHERE m.session_id = s.session_id
                    ) AS message_count,
                    (
                        SELECT ss.summary_text
                        FROM session_summaries ss
                        WHERE ss.session_id = s.session_id
                    ) AS summary_text
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "language": row["language"],
                "first_user_message": row["first_user_message"] or "",
                "message_count": row["message_count"] or 0,
                "summary_text": row["summary_text"] or "",
            }
            for row in rows
        ]

    def get_session_summary(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT summary_text
                FROM session_summaries
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return ""
        return row["summary_text"] or ""

    def upsert_session_summary(self, session_id: str, summary_text: str) -> None:
        now = utc_timestamp()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_summaries(session_id, summary_text, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary_text, now),
            )

    def clear_all_sessions(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages")
            connection.execute("DELETE FROM session_summaries")
            connection.execute("DELETE FROM sessions")
