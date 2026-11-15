"""Runtime helpers for explicitly opening the existing pooled MySQL store."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@contextmanager
def pooled_embedding_store():
    """Yield a MySQL embedding store using the backend's existing connection pool."""
    from database import TimedConnection, get_connection
    try:
        from .db_embedding_store import MySQLEmbeddingStore
    except ImportError:
        from db_embedding_store import MySQLEmbeddingStore

    connection = TimedConnection(get_connection())
    try:
        store = MySQLEmbeddingStore(connection)
        yield store
    finally:
        connection.close()
