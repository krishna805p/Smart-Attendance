import sys
import unittest
from datetime import date, time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database
import main


class FakeConnection:
    def __init__(self):
        self.closed = False
        self._cursor = FakeCursor()

    def cursor(self, *args, **kwargs):
        return self._cursor

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, rows=None):
        self.closed = False
        self.rows = list(rows or [])
        self.executed = []

    def execute(self, statement, params=None):
        self.executed.append((statement, params))

    def fetchone(self):
        return {"id": 7}

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class DatabasePoolTests(unittest.TestCase):
    def test_pool_is_created_lazily_with_configured_sizing_and_timeouts(self):
        with patch.object(database, "_pool", None), \
             patch.object(database, "POOL_SIZE", 3), \
             patch.object(database, "MySQLConnectionPool") as pool_class:
            sentinel = object()
            pool_class.return_value = sentinel
            self.assertIs(database._get_pool(), sentinel)
            kwargs = pool_class.call_args.kwargs
            self.assertEqual(kwargs["pool_size"], 3)
            self.assertEqual(kwargs["connection_timeout"], 10)
            self.assertEqual(kwargs["read_timeout"], 30)
            self.assertEqual(kwargs["write_timeout"], 30)
            self.assertTrue(kwargs["pool_reset_session"])

    def test_connection_dependency_returns_pooled_connection_on_normal_exit(self):
        connection = FakeConnection()
        class Pool:
            def get_connection(self):
                return connection
        with patch.object(database, "_get_pool", return_value=Pool()):
            dependency = database.get_db()
            wrapped = next(dependency)
            self.assertIsInstance(wrapped, database.TimedConnection)
            dependency.close()
        self.assertTrue(connection.closed)

    def test_connection_dependency_returns_pooled_connection_after_exception(self):
        connection = FakeConnection()
        class Pool:
            def get_connection(self):
                return connection
        with patch.object(database, "_get_pool", return_value=Pool()):
            dependency = database.get_db()
            next(dependency)
            with self.assertRaisesRegex(RuntimeError, "route failed"):
                dependency.throw(RuntimeError("route failed"))
        self.assertTrue(connection.closed)

    def test_student_attendance_response_remains_array_with_optional_pagination(self):
        record = {
            "id": 4,
            "student_id": 7,
            "date": date(2026, 1, 2),
            "time": time(9, 5),
            "status": "Present",
        }
        cursor = FakeCursor([record])
        class DB:
            def cursor(self, **_kwargs):
                return cursor
        legacy = main.get_student_attendance(
            "ROLL-7", limit=None, offset=0,
            user={"role": "student", "sub": "ROLL-7"}, db=DB(),
        )
        self.assertIsInstance(legacy, list)
        self.assertEqual(len(legacy), 1)
        self.assertNotIn("LIMIT", cursor.executed[-1][0])

        cursor.executed.clear()
        paged = main.get_student_attendance(
            "ROLL-7", limit=10, offset=20,
            user={"role": "student", "sub": "ROLL-7"}, db=DB(),
        )
        self.assertIsInstance(paged, list)
        self.assertEqual(paged[0]["id"], 4)
        self.assertIn("LIMIT %s OFFSET %s", cursor.executed[-1][0])
        self.assertEqual(cursor.executed[-1][1], (7, 10, 20))


if __name__ == "__main__":
    unittest.main()
