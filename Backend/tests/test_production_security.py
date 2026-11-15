import hashlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database
import main
from fastapi import HTTPException


class PasswordTests(unittest.TestCase):
    def test_new_password_hash_is_argon2id_and_verifies(self):
        encoded = main.hash_password("correct horse battery staple")
        self.assertTrue(encoded.startswith("$argon2id$"))
        self.assertTrue(main.verify_password("correct horse battery staple", encoded))
        self.assertFalse(main.verify_password("wrong", encoded))

    def test_legacy_pbkdf2_and_plaintext_remain_verifiable(self):
        password = "legacy-test-only"
        salt = b"0123456789abcdef"
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120_000).hex()
        legacy_pbkdf2 = f"pbkdf2_sha256$120000${salt.hex()}${digest}"
        self.assertTrue(main.verify_password(password, legacy_pbkdf2))
        self.assertTrue(main.verify_password(password, password))
        self.assertFalse(main.verify_password("different", legacy_pbkdf2))

    def test_successful_legacy_login_upgrade_path_preserves_account(self):
        class FakeCursor:
            def __init__(self):
                self.statement = None
            def execute(self, statement, values):
                self.statement, self.values = statement, values
            def close(self):
                self.closed = True
        class FakeDB:
            def __init__(self):
                self.item = FakeCursor()
                self.commits = 0
            def cursor(self):
                return self.item
            def commit(self):
                self.commits += 1
            def rollback(self):
                raise AssertionError("unexpected rollback")
        db = FakeDB()
        main.upgrade_password_hash(db, "students", 17, "legacy-test-only", "legacy-test-only")
        self.assertIn("UPDATE students", db.item.statement)
        self.assertTrue(db.item.values[0].startswith("$argon2id$"))
        self.assertEqual(db.item.values[1:], (17, "legacy-test-only"))
        self.assertEqual(db.commits, 1)

    def test_existing_signed_token_and_role_checks_remain_enforced(self):
        with patch.dict(os.environ, {"AUTH_TOKEN_SECRET": "test-only-secret-value-longer-than-32"}):
            token = main.create_access_token("student", "TEST-ROLL")
            user = main.get_current_user("Bearer " + token)
        self.assertEqual(user["role"], "student")
        self.assertEqual(user["sub"], "TEST-ROLL")
        with self.assertRaises(HTTPException) as caught:
            main.require_role(user, "teacher")
        self.assertEqual(caught.exception.status_code, 403)


class RateLimitTests(unittest.TestCase):
    def test_limit_and_window_reset(self):
        now = [100.0]
        limiter = main.LoginRateLimiter(limit=2, window_seconds=60, clock=lambda: now[0])
        self.assertEqual(limiter.allow("client:user"), (True, 0))
        self.assertEqual(limiter.allow("client:user"), (True, 0))
        allowed, retry = limiter.allow("client:user")
        self.assertFalse(allowed)
        self.assertGreater(retry, 0)
        now[0] += 61
        self.assertEqual(limiter.allow("client:user"), (True, 0))

    def test_request_limit_returns_retry_after(self):
        class Client:
            host = "test-client"
        class Request:
            client = Client()
        original = main.login_rate_limiter
        main.login_rate_limiter = main.LoginRateLimiter(limit=1)
        try:
            main.enforce_login_limit(Request(), "test-user")
            with self.assertRaises(HTTPException) as caught:
                main.enforce_login_limit(Request(), "test-user")
            self.assertEqual(caught.exception.status_code, 429)
            self.assertIn("Retry-After", caught.exception.headers)
        finally:
            main.login_rate_limiter = original


class CorsAndHealthTests(unittest.TestCase):
    def test_production_cors_has_no_wildcard_or_local_origins(self):
        with patch.dict(os.environ, {
            "CORS_ALLOWED_ORIGINS": "https://attendance.example",
            "CORS_PREVIEW_ORIGINS": "https://preview.example",
            "CORS_ALLOW_LOCAL_ORIGINS": "true",
        }, clear=False):
            origins = main.cors_origins("production")
        self.assertEqual(origins, ["https://attendance.example", "https://preview.example"])
        self.assertNotIn("*", origins)
        self.assertFalse(any("localhost" in origin or "127.0.0.1" in origin for origin in origins))

    def test_development_cors_includes_local_origins(self):
        with patch.dict(os.environ, {
            "CORS_ALLOWED_ORIGINS": "",
            "CORS_PREVIEW_ORIGINS": "",
            "CORS_ALLOW_LOCAL_ORIGINS": "true",
        }, clear=False):
            self.assertIn("http://127.0.0.1:5500", main.cors_origins("development"))

    def test_liveness_is_database_independent_and_readiness_is_not(self):
        with patch.object(main, "check_connection", side_effect=AssertionError("liveness queried DB")):
            self.assertEqual(main.liveness_check(), {"status": "ok"})
        with patch.object(main, "check_connection", return_value=False):
            response = main.readiness_check()
        self.assertEqual(response.status_code, 503)
        with patch.object(main, "check_connection", return_value=True):
            self.assertEqual(main.readiness_check(), {"status": "ok", "database": "connected"})

    def test_database_health_closes_connection_and_cursor_after_query_failure(self):
        class FakeCursor:
            closed = False
            def execute(self, _query):
                raise database.Error("synthetic test failure")
            def close(self):
                self.closed = True
        class FakeConnection:
            closed = False
            def __init__(self):
                self.item = FakeCursor()
            def cursor(self):
                return self.item
            def close(self):
                self.closed = True
        connection = FakeConnection()
        with patch.object(database, "get_connection", return_value=connection):
            self.assertFalse(database.check_connection())
        self.assertTrue(connection.item.closed)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
