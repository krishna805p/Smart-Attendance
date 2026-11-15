"""Aiven/MySQL-backed storage for encrypted MobileFaceNet templates."""

from __future__ import annotations

import base64
import os
import secrets
from collections import defaultdict
from typing import Iterable, Mapping

import mysql.connector
import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    from .embedding_store import EmbeddingStore, EmbeddingStoreError, normalize_embedding
except ImportError:  # standalone scripts run from Backend/face_system
    from embedding_store import EmbeddingStore, EmbeddingStoreError, normalize_embedding


MODEL_NAME = "qualcomm-mobilefacenet"
MODEL_VERSION = "qaihub-v0.66.0-onnx-float-align112-v1"
PREPROCESSING_VERSION = "retinaface5pt112-rgb01-imagenetnorm-tta-v1"
_FORMAT_MAGIC = b"FSE1"
_NONCE_BYTES = 12
_EMBEDDING_BYTES = 128 * np.dtype(np.float32).itemsize


class EmbeddingStorageError(RuntimeError):
    """Stored templates are malformed, incompatible, or cannot be used."""


class EmbeddingStorageUnavailable(EmbeddingStorageError):
    """The database or encryption configuration is unavailable."""


class EnrollmentAlreadyExists(EmbeddingStorageError):
    """A student already has templates and replacement was not authorized."""


class StudentRecordNotFound(EmbeddingStorageError):
    """The requested existing student row was not found."""


def encryption_key_from_env(value: str | None = None) -> bytes:
    """Read a base64-encoded 256-bit key supplied outside source control."""
    encoded = value if value is not None else os.getenv("FACE_EMBEDDING_ENCRYPTION_KEY", "")
    if not encoded:
        raise EmbeddingStorageUnavailable(
            "Face embedding encryption is not configured; set FACE_EMBEDDING_ENCRYPTION_KEY."
        )
    try:
        key = base64.urlsafe_b64decode(encoded.encode("ascii") + b"=" * (-len(encoded) % 4))
    except (ValueError, UnicodeEncodeError) as exc:
        raise EmbeddingStorageUnavailable("Face embedding encryption key is invalid.") from exc
    if len(key) != 32:
        raise EmbeddingStorageUnavailable("Face embedding encryption key must decode to 32 bytes.")
    return key


class MySQLEmbeddingStore:
    """Parameterized, transaction-safe embedding store bound to one DB connection.

    The connection is normally a wrapper returned by Backend/database.py's pool.
    Template data is AES-256-GCM encrypted before it is sent to MySQL. Student ID
    and model identity are authenticated as associated data to prevent row swaps.
    """

    def __init__(self, db, key: bytes | None = None, *, model_name: str = MODEL_NAME,
                 model_version: str = MODEL_VERSION,
                 preprocessing_version: str = PREPROCESSING_VERSION) -> None:
        self.db = db
        self._key = encryption_key_from_env() if key is None else bytes(key)
        if len(self._key) != 32:
            raise EmbeddingStorageUnavailable("Face embedding encryption key must be 32 bytes.")
        self._aes = AESGCM(self._key)
        self.model_name = model_name
        self.model_version = model_version
        if not isinstance(preprocessing_version, str) or not preprocessing_version.strip() \
                or len(preprocessing_version) > 128:
            raise ValueError("preprocessing_version must be 1-128 nonblank characters")
        self.preprocessing_version = preprocessing_version

    @staticmethod
    def validate_roll_number(roll_number: str) -> str:
        return EmbeddingStore.validate_roll_number(roll_number)

    @staticmethod
    def _student_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError("student_id must be a positive integer")
        return int(value)

    def _aad(self, student_id: int, model_name: str, model_version: str,
             preprocessing_version: str) -> bytes:
        return (f"face-embedding:v1:{student_id}:{model_name}:{model_version}:"
                f"{preprocessing_version}").encode("utf-8")

    def _encrypt(self, student_id: int, embedding: object) -> bytes:
        vector = normalize_embedding(embedding)
        nonce = secrets.token_bytes(_NONCE_BYTES)
        cipher = self._aes.encrypt(
            nonce, vector.astype("<f4", copy=False).tobytes(),
            self._aad(student_id, self.model_name, self.model_version,
                      self.preprocessing_version),
        )
        return _FORMAT_MAGIC + nonce + cipher

    def _decrypt(self, student_id: int, blob: bytes, model_name: str,
                 model_version: str, preprocessing_version: str) -> np.ndarray:
        if model_name != self.model_name or model_version != self.model_version:
            raise EmbeddingStorageError("Stored face template uses an incompatible model version.")
        if preprocessing_version != self.preprocessing_version:
            raise EmbeddingStorageError(
                "Stored face template uses an incompatible preprocessing version."
            )
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise EmbeddingStorageError("Stored face template is not binary data.")
        payload = bytes(blob)
        minimum = len(_FORMAT_MAGIC) + _NONCE_BYTES + _EMBEDDING_BYTES + 16
        if len(payload) != minimum or not payload.startswith(_FORMAT_MAGIC):
            raise EmbeddingStorageError("Stored face template has an invalid binary format.")
        nonce_start = len(_FORMAT_MAGIC)
        nonce = payload[nonce_start:nonce_start + _NONCE_BYTES]
        cipher = payload[nonce_start + _NONCE_BYTES:]
        try:
            raw = self._aes.decrypt(
                nonce, cipher,
                self._aad(student_id, model_name, model_version, preprocessing_version),
            )
        except InvalidTag as exc:
            raise EmbeddingStorageError("Stored face template authentication failed.") from exc
        if len(raw) != _EMBEDDING_BYTES:
            raise EmbeddingStorageError("Stored face template has an invalid vector length.")
        vector = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise EmbeddingStorageError("Stored face template is not a normalized finite vector.")
        return vector

    @staticmethod
    def _row_value(row, key: str, index: int):
        return row[key] if isinstance(row, Mapping) else row[index]

    def get_student_id_by_roll(self, roll_number: str) -> int:
        roll = self.validate_roll_number(roll_number)
        cursor = None
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT id FROM students WHERE roll_number = %s", (roll,))
            row = cursor.fetchone()
            if row is None:
                raise StudentRecordNotFound("Student record was not found.")
            return self._student_id(self._row_value(row, "id", 0))
        except StudentRecordNotFound:
            raise
        except mysql.connector.Error as exc:
            raise EmbeddingStorageUnavailable("Student database is temporarily unavailable.") from exc
        finally:
            if cursor is not None:
                cursor.close()

    def _read_rows(self, query: str, params: tuple) -> list:
        cursor = None
        try:
            cursor = self.db.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except mysql.connector.Error as exc:
            raise EmbeddingStorageUnavailable("Face enrollment storage is temporarily unavailable.") from exc
        finally:
            if cursor is not None:
                cursor.close()

    def get_student_embeddings(self, student_id: int) -> list[np.ndarray]:
        sid = self._student_id(student_id)
        rows = self._read_rows(
            "SELECT embedding, model_name, model_version, preprocessing_version "
            "FROM student_face_embeddings "
            "WHERE student_id = %s ORDER BY id", (sid,)
        )
        return [self._decrypt(sid, self._row_value(row, "embedding", 0),
                              self._row_value(row, "model_name", 1),
                              self._row_value(row, "model_version", 2),
                              self._row_value(row, "preprocessing_version", 3)) for row in rows]

    def get_student_embeddings_by_roll(self, roll_number: str) -> list[np.ndarray]:
        return self.get_student_embeddings(self.get_student_id_by_roll(roll_number))

    def get_all_student_embeddings(self) -> dict[int, list[np.ndarray]]:
        rows = self._read_rows(
            "SELECT student_id, embedding, model_name, model_version, preprocessing_version "
            "FROM student_face_embeddings ORDER BY student_id, id", ()
        )
        output: dict[int, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            sid = self._student_id(self._row_value(row, "student_id", 0))
            output[sid].append(self._decrypt(
                sid, self._row_value(row, "embedding", 1),
                self._row_value(row, "model_name", 2),
                self._row_value(row, "model_version", 3),
                self._row_value(row, "preprocessing_version", 4),
            ))
        return dict(output)

    def get_all_student_embeddings_by_roll(self) -> dict[str, list[np.ndarray]]:
        rows = self._read_rows(
            "SELECT s.roll_number, e.student_id, e.embedding, e.model_name, e.model_version, "
            "e.preprocessing_version "
            "FROM student_face_embeddings e JOIN students s ON s.id = e.student_id "
            "ORDER BY s.roll_number, e.id", ()
        )
        output: dict[str, list[np.ndarray]] = defaultdict(list)
        for row in rows:
            roll = self._row_value(row, "roll_number", 0)
            sid = self._student_id(self._row_value(row, "student_id", 1))
            output[str(roll)].append(self._decrypt(
                sid, self._row_value(row, "embedding", 2),
                self._row_value(row, "model_name", 3),
                self._row_value(row, "model_version", 4),
                self._row_value(row, "preprocessing_version", 5),
            ))
        return dict(output)

    def student_has_enrollment(self, student_id: int) -> bool:
        sid = self._student_id(student_id)
        rows = self._read_rows(
            "SELECT id FROM student_face_embeddings WHERE student_id = %s LIMIT 1", (sid,)
        )
        return bool(rows)

    @staticmethod
    def _prepare(embeddings: Iterable[object]) -> list[np.ndarray]:
        values = list(embeddings)
        if not values:
            raise EmbeddingStoreError("at least one embedding is required")
        return [normalize_embedding(value) for value in values]

    def _write_records(self, prepared: Mapping[int, list[np.ndarray]], *, replace: bool,
                       skip_existing: bool = False) -> tuple[int, int]:
        encrypted = {
            sid: [self._encrypt(sid, vector) for vector in vectors]
            for sid, vectors in prepared.items()
        }
        cursor = None
        saved_students = skipped_students = 0
        try:
            cursor = self.db.cursor()
            for sid in sorted(encrypted):
                cursor.execute("SELECT id FROM students WHERE id = %s FOR UPDATE", (sid,))
                if cursor.fetchone() is None:
                    if skip_existing:
                        skipped_students += 1
                        continue
                    raise StudentRecordNotFound("Student record was not found.")
                cursor.execute("SELECT id FROM student_face_embeddings WHERE student_id = %s LIMIT 1", (sid,))
                exists = cursor.fetchone() is not None
                if exists and skip_existing:
                    skipped_students += 1
                    continue
                if exists and not replace:
                    raise EnrollmentAlreadyExists("Student already has a face enrollment.")
                if exists:
                    cursor.execute("DELETE FROM student_face_embeddings WHERE student_id = %s", (sid,))
                cursor.executemany(
                    "INSERT INTO student_face_embeddings "
                    "(student_id, embedding, model_name, model_version, preprocessing_version) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    [(sid, blob, self.model_name, self.model_version, self.preprocessing_version)
                     for blob in encrypted[sid]],
                )
                saved_students += 1
            self.db.commit()
            return saved_students, skipped_students
        except (StudentRecordNotFound, EnrollmentAlreadyExists):
            self.db.rollback()
            raise
        except mysql.connector.Error as exc:
            self.db.rollback()
            raise EmbeddingStorageUnavailable("Face enrollment storage is temporarily unavailable.") from exc
        except Exception:
            self.db.rollback()
            raise
        finally:
            if cursor is not None:
                cursor.close()

    def save_student_embeddings(self, student_id: int, embeddings: Iterable[object]) -> int:
        sid = self._student_id(student_id)
        prepared = {sid: self._prepare(embeddings)}
        self._write_records(prepared, replace=False)
        return len(prepared[sid])

    def replace_student_embeddings(self, student_id: int, embeddings: Iterable[object], *,
                                   authorized: bool = False) -> int:
        if authorized is not True:
            raise PermissionError("Replacing face enrollment requires explicit authorization.")
        sid = self._student_id(student_id)
        prepared = {sid: self._prepare(embeddings)}
        self._write_records(prepared, replace=True)
        return len(prepared[sid])

    def save_many_student_embeddings(self, records: Mapping[int, Iterable[object]]) -> tuple[int, int]:
        """Insert only missing enrollments in one transaction; safe to rerun."""
        prepared = {self._student_id(sid): self._prepare(vectors)
                    for sid, vectors in records.items()}
        if not prepared:
            return 0, 0
        return self._write_records(prepared, replace=False, skip_existing=True)

    def delete_student_embeddings(self, student_id: int) -> int:
        sid = self._student_id(student_id)
        cursor = None
        try:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM student_face_embeddings WHERE student_id = %s", (sid,))
            deleted = int(cursor.rowcount)
            self.db.commit()
            return deleted
        except mysql.connector.Error as exc:
            self.db.rollback()
            raise EmbeddingStorageUnavailable("Face enrollment storage is temporarily unavailable.") from exc
        finally:
            if cursor is not None:
                cursor.close()
