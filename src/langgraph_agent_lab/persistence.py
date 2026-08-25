"""Checkpointer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _sqlite_path(database_url: str | None) -> str:
    """Normalize a database_url (plain path or sqlite:/// URL) to a file path."""
    if not database_url:
        return "checkpoints.db"
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        return database_url[len(prefix) :]
    if database_url.startswith("file:"):
        return database_url[len("file:") :]
    return database_url


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    Supported kinds: none, memory, sqlite (requires langgraph-checkpoint-sqlite),
    postgres (requires langgraph-checkpoint-postgres).
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                "Install with: pip install 'langgraph-checkpoint-sqlite'"
            ) from exc

        import sqlite3

        path = Path(_sqlite_path(database_url))
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(connection)
    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires langgraph-checkpoint-postgres. "
                "Install with: pip install 'langgraph-checkpoint-postgres'"
            ) from exc

        if not database_url:
            raise ValueError("postgres checkpointer requires DATABASE_URL")
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires psycopg: pip install psycopg"
            ) from exc

        connection = psycopg.connect(database_url, autocommit=True)
        checkpointer = PostgresSaver(connection)
        checkpointer.setup()
        return checkpointer
    raise ValueError(f"Unknown checkpointer kind: {kind}")
