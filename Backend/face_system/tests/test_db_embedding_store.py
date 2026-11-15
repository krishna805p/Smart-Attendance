from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_embedding_store import (
    MODEL_NAME, MODEL_VERSION, PREPROCESSING_VERSION, EnrollmentAlreadyExists,
    EmbeddingStorageError, MySQLEmbeddingStore,
)
from embedding_store import EmbeddingStore
from migrate_embeddings import migration_plan


def vector(index: int = 0) -> np.ndarray:
    result = np.zeros(128, dtype=np.float32)
    result[index] = 1.0
    return result


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.rows = []
        self.last_query = ""
        self.rowcount = 0

    def execute(self, query, params=()):
        self.last_query = query
        if query.startswith("SELECT id FROM students"):
            if isinstance(params[0], str):
                student_id = next((sid for sid, roll in self.db.students.items()
                                   if roll == params[0]), None)
            else:
                student_id = int(params[0])
            self.rows = [{"id": student_id}] if student_id in self.db.students else []
        elif query.startswith("SELECT id FROM student_face_embeddings"):
            student_id = int(params[0])
            existing = [row for row in self.db.rows if row["student_id"] == student_id]
            self.rows = [{"id": item["id"]} for item in existing[:1]]
        elif query.startswith("SELECT embedding, model_name"):
            student_id = int(params[0])
            self.rows = [self._select_row(row) for row in self.db.rows
                         if row["student_id"] == student_id]
        elif query.startswith("SELECT student_id, embedding"):
            self.rows = [self._select_row(row, include_student=True) for row in self.db.rows]
        elif query.startswith("SELECT s.roll_number"):
            self.rows = [self._select_row(row, include_roll=True) for row in self.db.rows]
        else:
            raise AssertionError(f"Unexpected query: {query}")

    @staticmethod
    def _select_row(row, include_student=False, include_roll=False):
        base = {
            "embedding": row["embedding"], "model_name": row["model_name"],
            "model_version": row["model_version"],
            "preprocessing_version": row["preprocessing_version"],
        }
        if include_student:
            return {"student_id": row["student_id"], **base}
        if include_roll:
            return {"roll_number": row["roll_number"], "student_id": row["student_id"], **base}
        return base

    def executemany(self, query, rows):
        self.last_query = query
        if "preprocessing_version" not in query:
            raise AssertionError("INSERT must explicitly provide preprocessing_version")
        for student_id, encrypted, model_name, model_version, preprocessing_version in rows:
            record = {
                "id": self.db.next_id, "student_id": int(student_id),
                "embedding": bytes(encrypted), "model_name": model_name,
                "model_version": model_version,
                "preprocessing_version": preprocessing_version,
                "roll_number": self.db.students[int(student_id)],
            }
            self.db.next_id += 1
            self.db.rows.append(record)
            self.rowcount += 1

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class FakeDB:
    def __init__(self):
        self.students = {7: "ROLL7", 8: "ROLL8"}
        self.rows = []
        self.next_id = 1
        self.commits = 0
        self.rollbacks = 0
        self._snapshot = None

    def cursor(self, **_kwargs):
        self._snapshot = copy.deepcopy(self.rows)
        return FakeCursor(self)

    def commit(self):
        self.commits += 1
        self._snapshot = None

    def rollback(self):
        self.rollbacks += 1
        if self._snapshot is not None:
            self.rows = self._snapshot
            self.next_id = max((row["id"] for row in self.rows), default=0) + 1
            self._snapshot = None


class FailingInsertCursor(FakeCursor):
    def executemany(self, query, rows):
        super().executemany(query, rows)
        raise RuntimeError("simulated insert failure")


class FailingInsertDB(FakeDB):
    def cursor(self, **_kwargs):
        self._snapshot = copy.deepcopy(self.rows)
        return FailingInsertCursor(self)


class DBEmbeddingStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        self.store = MySQLEmbeddingStore(self.db, key=bytes(range(32)))

    def test_aes_gcm_round_trip_and_expected_payload_size(self):
        encrypted = self.store._encrypt(7, vector())
        self.assertEqual(len(encrypted), 544)
        self.assertEqual(self.store._decrypt(
            7, encrypted, MODEL_NAME, MODEL_VERSION, PREPROCESSING_VERSION
        ).tolist(), vector().tolist())

    def test_tampered_ciphertext_is_rejected(self):
        encrypted = bytearray(self.store._encrypt(7, vector()))
        encrypted[-1] ^= 0x01
        with self.assertRaisesRegex(EmbeddingStorageError, "authentication failed"):
            self.store._decrypt(7, encrypted, MODEL_NAME, MODEL_VERSION, PREPROCESSING_VERSION)

    def test_preprocessing_version_is_validated_and_authenticated(self):
        encrypted = self.store._encrypt(7, vector())
        with self.assertRaisesRegex(EmbeddingStorageError, "preprocessing version"):
            self.store._decrypt(7, encrypted, MODEL_NAME, MODEL_VERSION, "other-preprocessing-v2")
        with self.assertRaises(ValueError):
            MySQLEmbeddingStore(self.db, key=bytes(range(32)), preprocessing_version="")

    def test_insert_and_retrieval_use_actual_table_preprocessing_column(self):
        self.store.save_student_embeddings(7, [vector(), vector(1)])
        self.assertEqual(len(self.db.rows), 2)
        self.assertTrue(all(row["preprocessing_version"] == PREPROCESSING_VERSION
                            for row in self.db.rows))
        loaded = self.store.get_student_embeddings(7)
        self.assertEqual(len(loaded), 2)
        self.assertTrue(np.array_equal(loaded[0], vector()))
        self.assertTrue(np.array_equal(loaded[1], vector(1)))

    def test_per_student_retrieval_isolation_and_roll_mapping(self):
        self.store.save_student_embeddings(7, [vector(0)])
        self.store.save_student_embeddings(8, [vector(1)])
        self.assertEqual(len(self.store.get_student_embeddings(7)), 1)
        self.assertEqual(self.store.get_all_student_embeddings_by_roll().keys(), {"ROLL7", "ROLL8"})

    def test_duplicate_enrollment_is_rejected_and_existing_templates_preserved(self):
        self.store.save_student_embeddings(7, [vector(0)])
        first_blob = self.db.rows[0]["embedding"]
        with self.assertRaises(EnrollmentAlreadyExists):
            self.store.save_student_embeddings(7, [vector(1)])
        self.assertEqual(len(self.db.rows), 1)
        self.assertEqual(self.db.rows[0]["embedding"], first_blob)
        self.assertEqual(self.db.rollbacks, 1)

    def test_insert_failure_rolls_back_partial_database_rows(self):
        failing = MySQLEmbeddingStore(FailingInsertDB(), key=bytes(range(32)))
        with self.assertRaisesRegex(RuntimeError, "simulated insert failure"):
            failing.save_student_embeddings(7, [vector(), vector(1)])
        self.assertEqual(failing.db.rows, [])
        self.assertEqual(failing.db.rollbacks, 1)

    def test_migration_dry_run_and_apply_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            source = EmbeddingStore(Path(temp) / "legacy.json")
            source.save_student("ROLL7", [vector()])
            dry = migration_plan(source, self.store)
            self.assertEqual(dry["mode"], "dry_run")
            self.assertEqual(dry["eligible_students"], 1)
            self.assertEqual(self.db.rows, [])
            applied = migration_plan(source, self.store, apply=True)
            self.assertEqual(applied["inserted_students"], 1)
            rerun = migration_plan(source, self.store, apply=True)
            self.assertEqual(rerun["already_enrolled_students"], 1)
            self.assertEqual(len(self.db.rows), 1)
            self.assertNotIn("ROLL7", str(dry))


if __name__ == "__main__":
    unittest.main()
