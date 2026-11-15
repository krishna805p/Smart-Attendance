"""One-time, idempotent local JSON to MySQL embedding migration utility.

Default mode is dry-run. The source JSON is read only and never deleted.
Apply requires the explicit --apply switch and a reachable configured database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .db_store_runtime import pooled_embedding_store
    from .embedding_store import DEFAULT_STORE_PATH, EmbeddingStore
    from .db_embedding_store import StudentRecordNotFound
except ImportError:
    from db_store_runtime import pooled_embedding_store
    from embedding_store import DEFAULT_STORE_PATH, EmbeddingStore
    from db_embedding_store import StudentRecordNotFound


def migration_plan(source: EmbeddingStore, target, *, apply: bool = False) -> dict[str, int | str]:
    """Validate and map the legacy file; report counts without biometric values."""
    legacy = source.load_all()
    mapped: dict[int, list[Any]] = {}
    unmatched = 0
    already_present = 0
    for roll, embeddings in legacy.items():
        try:
            student_id = target.get_student_id_by_roll(roll)
        except StudentRecordNotFound:
            unmatched += 1
            continue
        if target.student_has_enrollment(student_id):
            already_present += 1
            continue
        mapped[student_id] = embeddings

    inserted = 0
    skipped = 0
    if apply and mapped:
        inserted, skipped = target.save_many_student_embeddings(mapped)
    return {
        "mode": "apply" if apply else "dry_run",
        "source_students": len(legacy),
        "eligible_students": len(mapped),
        "unmatched_students": unmatched,
        "already_enrolled_students": already_present,
        "inserted_students": inserted,
        "concurrently_skipped_students": skipped,
        "source_preserved": "true",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_STORE_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="inspect and validate only (default)")
    mode.add_argument("--apply", action="store_true", help="insert only missing student enrollments")
    args = parser.parse_args()
    source = EmbeddingStore(args.source)
    try:
        with pooled_embedding_store() as target:
            summary = migration_plan(source, target, apply=args.apply)
    except Exception as exc:
        # Deliberately omit exception text: connector exceptions can include
        # deployment-specific connection details. Never log embeddings.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
