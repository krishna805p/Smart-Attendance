"""Local JSON store for normalized face embeddings (no raw images)."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np


DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "data" / "student_embeddings.json"
_ROLL_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_SCHEMA_VERSION = 1


class EmbeddingStoreError(ValueError):
    """Raised when a local embedding store is malformed or unsafe to update."""


def normalize_embedding(value: object) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingStoreError("embedding must be a numeric 128D vector") from exc
    if vector.shape != (128,):
        raise EmbeddingStoreError(f"embedding must have shape (128,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise EmbeddingStoreError("embedding contains NaN or infinity")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise EmbeddingStoreError("embedding must have a non-zero finite norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


class EmbeddingStore:
    """Read, validate, and atomically update local student embedding templates."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)

    @staticmethod
    def validate_roll_number(roll_number: str) -> str:
        if not isinstance(roll_number, str):
            raise EmbeddingStoreError("roll number must be text")
        roll = roll_number.strip()
        if not _ROLL_RE.fullmatch(roll):
            raise EmbeddingStoreError(
                "roll number must be 1-40 letters, digits, underscores, or hyphens"
            )
        return roll

    def load_all(self) -> dict[str, list[np.ndarray]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise EmbeddingStoreError(f"cannot read embedding store {self.path}: {exc}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != _SCHEMA_VERSION:
            raise EmbeddingStoreError("unsupported or missing embedding-store schema_version")
        entries = document.get("students")
        if not isinstance(entries, dict):
            raise EmbeddingStoreError("embedding-store students must be an object")
        output: dict[str, list[np.ndarray]] = {}
        for raw_roll, raw_vectors in entries.items():
            roll = self.validate_roll_number(raw_roll)
            if not isinstance(raw_vectors, list) or not raw_vectors:
                raise EmbeddingStoreError(f"student {roll} must have a non-empty embedding list")
            output[roll] = [normalize_embedding(vector) for vector in raw_vectors]
        return output

    def save_student(self, roll_number: str, embeddings: list[object], *, replace: bool = False) -> None:
        roll = self.validate_roll_number(roll_number)
        if not isinstance(embeddings, list) or not embeddings:
            raise EmbeddingStoreError("at least one embedding is required")
        normalized = [normalize_embedding(vector) for vector in embeddings]
        students = self.load_all()
        if roll in students and not replace:
            raise EmbeddingStoreError(
                f"roll number {roll} is already enrolled; pass replace=True to replace it"
            )
        students[roll] = normalized
        self._write(students)

    def delete_student(self, roll_number: str) -> bool:
        roll = self.validate_roll_number(roll_number)
        students = self.load_all()
        if roll not in students:
            return False
        del students[roll]
        self._write(students)
        return True

    def _write(self, students: Mapping[str, list[np.ndarray]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": _SCHEMA_VERSION,
            "students": {
                roll: [vector.astype(float).tolist() for vector in vectors]
                for roll, vectors in sorted(students.items())
            },
        }
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False
            ) as handle:
                temporary_path = handle.name
                json.dump(document, handle, separators=(",", ":"), allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise
