"""Pooled MySQL access and safe slow-query diagnostics for the FastAPI backend."""

import logging
import os
import re
import threading
import time
from pathlib import Path

import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
from dotenv import load_dotenv

# Load backend-local settings before building the connection configuration.
load_dotenv(Path(__file__).with_name(".env"))

logger = logging.getLogger(__name__)


def _positive_int(name: str, default: int, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer") from error
    if value < 1 or (maximum is not None and value > maximum):
        limit = f" between 1 and {maximum}" if maximum is not None else " greater than zero"
        raise RuntimeError(f"{name} must be{limit}")
    return value


POOL_SIZE = _positive_int("DB_POOL_SIZE", 5, maximum=32)
SLOW_QUERY_MS = _positive_int("DB_SLOW_QUERY_MS", 250)


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean .env value without weakening secure TLS defaults."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "smart_attendance"),
    "connection_timeout": _positive_int("DB_CONNECT_TIMEOUT_SECONDS", 10),
}

# Connector/Python added socket response timeouts in 9.2.0. Older installed
# connector versions still receive the supported TCP connection timeout above.
connector_version = tuple(getattr(mysql.connector, "__version_info__", (0, 0, 0)))
if connector_version[:2] >= (9, 2):
    DB_CONFIG["read_timeout"] = _positive_int("DB_READ_TIMEOUT_SECONDS", 30)
    DB_CONFIG["write_timeout"] = _positive_int("DB_WRITE_TIMEOUT_SECONDS", 30)

# A local MySQL configuration needs no TLS entries. Aiven deployments can set
# DB_SSL_CA to the supplied CA file, enabling certificate validation here.
ssl_ca = os.getenv("DB_SSL_CA", "").strip()
if ssl_ca:
    DB_CONFIG.update(
        {
            "ssl_ca": ssl_ca,
            "ssl_verify_cert": env_bool("DB_SSL_VERIFY_CERT", True),
            "ssl_verify_identity": env_bool("DB_SSL_VERIFY_IDENTITY", True),
        }
    )

_pool: MySQLConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> MySQLConnectionPool:
    """Create the per-process pool on first use, not during module import."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = MySQLConnectionPool(
                    pool_name="smart_attendance_pool",
                    pool_size=POOL_SIZE,
                    pool_reset_session=True,
                    **DB_CONFIG,
                )
    return _pool


def get_connection():
    """Acquire a pooled connection; call close() to return it to the pool."""
    return _get_pool().get_connection()


def _query_label(statement: str) -> str:
    """Return only SQL operation/table names, never SQL values or parameters."""
    compact = " ".join(statement.split())
    verb_match = re.match(r"([A-Za-z]+)", compact)
    verb = verb_match.group(1).upper() if verb_match else "UNKNOWN"
    tables = list(dict.fromkeys(
        match.group(1).strip("`")
        for match in re.finditer(r"\b(?:FROM|JOIN|UPDATE|INTO)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", compact, re.I)
    ))
    return f"{verb}:{','.join(tables) if tables else 'unknown'}"


class TimedCursor:
    """Cursor proxy that emits only slow/failing query labels and durations."""
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, operation, *args, **kwargs):
        label = _query_label(operation)
        started = time.perf_counter()
        completed = False
        try:
            result = self._cursor.execute(operation, *args, **kwargs)
            completed = True
            return result
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.warning("Database query failed operation=%s duration_ms=%.1f error_type=%s",
                           label, elapsed_ms, type(error).__name__)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if completed and elapsed_ms >= SLOW_QUERY_MS:
                logger.warning("Slow database query operation=%s duration_ms=%.1f", label, elapsed_ms)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class TimedConnection:
    """Connection proxy; closing its pooled connection returns it to the pool."""
    def __init__(self, connection):
        self._connection = connection

    def cursor(self, *args, **kwargs):
        return TimedCursor(self._connection.cursor(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._connection, name)


def get_db():
    """FastAPI dependency that always returns its pooled connection to the pool."""
    connection = TimedConnection(get_connection())
    try:
        yield connection
    finally:
        connection.close()


def check_connection() -> bool:
    """Readiness check that always closes its cursor and returns the connection."""
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = TimedCursor(connection.cursor())
        cursor.execute("SELECT 1")
        cursor.fetchone()
        return True
    except Error:
        return False
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Error:
                pass
        if connection is not None:
            try:
                connection.close()
            except Error:
                pass
