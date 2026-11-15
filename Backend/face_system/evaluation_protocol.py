"""Analysis helpers for held-out, score-only face-recognition evaluation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


STRATEGIES = ("top3_mean", "mean", "maximum", "quality_weighted")
DEFAULT_THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.50, 0.951, 0.05))
MARGIN_GRID = (None, 0.02, 0.05, 0.10, 0.15, 0.20)
REQUIRED_COLUMNS = {
    "capture_id", "participant_id", "role", "expected_candidate_id", "session_id",
    "condition", "partition", "candidate_id", "top3_mean", "mean", "maximum",
}


def aggregate_template_scores(scores: Iterable[float], strategy: str,
                              quality_weights: Iterable[float] | None = None) -> float:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    if np.any((values < -1.000001) | (values > 1.000001)):
        raise ValueError("cosine scores must be between -1 and 1")
    if strategy == "top3_mean":
        return float(np.mean(np.sort(values)[-3:]))
    if strategy == "mean":
        return float(np.mean(values))
    if strategy == "maximum":
        return float(np.max(values))
    if strategy == "quality_weighted":
        if quality_weights is None:
            raise ValueError("quality_weighted requires per-enrollment quality metadata")
        weights = np.asarray(list(quality_weights), dtype=np.float64)
        if weights.shape != values.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("quality weights must be finite, non-negative, and match score count")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("quality weights must have positive sum")
        return float(np.dot(values, weights) / total)
    raise ValueError(f"unsupported aggregation strategy: {strategy}")


def load_score_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"score CSV requires columns: {', '.join(sorted(REQUIRED_COLUMNS))}")
        rows = list(reader)
    if not rows:
        raise ValueError("score CSV contains no captures")
    captures: dict[str, tuple[str, str, str, str, str]] = {}
    candidate_sets: dict[str, set[str]] = defaultdict(set)
    gallery_candidate_set: set[str] | None = None
    for line, row in enumerate(rows, start=2):
        capture_id = row["capture_id"].strip()
        participant = row["participant_id"].strip()
        role = row["role"].strip()
        expected = row["expected_candidate_id"].strip()
        session = row["session_id"].strip()
        partition = row["partition"].strip()
        if not all((capture_id, participant, session)):
            raise ValueError(f"blank capture/person/session ID on CSV line {line}")
        if not participant.startswith("person_"):
            raise ValueError(f"participant_id must use an anonymous person_ label on CSV line {line}")
        if role not in ("genuine", "impostor") or partition not in ("calibration", "test"):
            raise ValueError(f"invalid role or partition on CSV line {line}")
        if (role == "genuine") != bool(expected):
            raise ValueError(f"genuine captures need expected identity; impostor captures must not on line {line}")
        condition = row["condition"].strip()
        metadata = (participant, role, expected, session, partition, condition)
        old = captures.setdefault(capture_id, metadata)
        if old != metadata:
            raise ValueError(f"capture metadata changes within capture_id {capture_id}")
        candidate = row["candidate_id"].strip()
        if not candidate.startswith("candidate_") or candidate in candidate_sets[capture_id]:
            raise ValueError(f"blank or duplicate candidate_id for capture {capture_id}")
        candidate_sets[capture_id].add(candidate)
        for name in ("top3_mean", "mean", "maximum"):
            try:
                score = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {name} score on CSV line {line}") from exc
            if not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError(f"{name} score must be finite and in [-1,1] on line {line}")
    for capture_id, candidates in candidate_sets.items():
        if gallery_candidate_set is None:
            gallery_candidate_set = candidates
        elif candidates != gallery_candidate_set:
            raise ValueError(f"gallery candidate set changed for capture {capture_id}; freeze enrollment during evaluation")
    session_partition: dict[str, str] = {}
    for participant, role, _expected, session, partition, _condition in captures.values():
        existing = session_partition.setdefault(session, partition)
        if existing != partition:
            raise ValueError(f"session {session!r} appears in both calibration and test partitions")
    participants_role: dict[str, str] = {}
    for participant, role, *_rest in captures.values():
        existing = participants_role.setdefault(participant, role)
        if existing != role:
            raise ValueError(f"participant {participant!r} has inconsistent genuine/impostor roles")
    return rows


def _capture_groups(rows: list[dict[str, str]], partition: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["partition"] == partition:
            grouped[row["capture_id"]].append(row)
    output = []
    for capture_id, members in grouped.items():
        first = members[0]
        ids = [row["candidate_id"] for row in members]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate candidate score for capture {capture_id}")
        expected = first["expected_candidate_id"].strip()
        if expected and expected not in ids:
            raise ValueError(f"expected candidate missing for capture {capture_id}")
        output.append({
            "capture_id": capture_id,
            "participant_id": first["participant_id"],
            "role": first["role"],
            "expected": expected,
            "session_id": first["session_id"],
            "condition": first["condition"],
            "scores": {row["candidate_id"]: row for row in members},
        })
    return output


def _score_distribution(captures: list[dict[str, object]], strategy: str) -> dict[str, dict[str, float | int]]:
    pair_scores: dict[str, list[float]] = {"genuine": [], "impostor": []}
    for capture in captures:
        role = capture["role"]
        expected = capture["expected"]
        for candidate, row in capture["scores"].items():
            pair_role = "genuine" if role == "genuine" and candidate == expected else "impostor"
            pair_scores[pair_role].append(float(row[strategy]))
    result = {}
    for role, scores in pair_scores.items():
        if scores:
            arr = np.asarray(scores, dtype=np.float64)
            result[role] = {"pairs": int(len(arr)), "min": float(arr.min()),
                            "median": float(np.median(arr)), "mean": float(arr.mean()),
                            "max": float(arr.max())}
    return result


def _pair_rates(captures: list[dict[str, object]], strategy: str, threshold: float) -> tuple[float, float]:
    fp = tn = fn = tp = 0
    for capture in captures:
        for candidate, row in capture["scores"].items():
            is_genuine = capture["role"] == "genuine" and candidate == capture["expected"]
            accepted = float(row[strategy]) >= threshold
            if is_genuine:
                tp += int(accepted)
                fn += int(not accepted)
            else:
                fp += int(accepted)
                tn += int(not accepted)
    return fp / max(1, fp + tn), fn / max(1, fn + tp)


def _eer_and_roc(captures: list[dict[str, object]], strategy: str) -> dict[str, object] | None:
    positive, negative = [], []
    for capture in captures:
        for candidate, row in capture["scores"].items():
            score = float(row[strategy])
            if capture["role"] == "genuine" and candidate == capture["expected"]:
                positive.append(score)
            else:
                negative.append(score)
    if not positive or not negative:
        return None
    lo = min(positive + negative) - 1e-8
    hi = max(positive + negative) + 1e-8
    thresholds = np.linspace(lo, hi, 201)
    points = []
    for threshold in thresholds:
        far = sum(score >= threshold for score in negative) / len(negative)
        frr = sum(score < threshold for score in positive) / len(positive)
        points.append({"threshold": float(threshold), "far": float(far),
                       "frr": float(frr), "tpr": float(1.0 - frr)})
    nearest = min(points, key=lambda p: abs(p["far"] - p["frr"]))
    eer = (nearest["far"] + nearest["frr"]) / 2.0
    return {"eer": float(eer), "eer_threshold_approx": nearest["threshold"],
            "roc_points": points, "genuine_pairs": len(positive), "impostor_pairs": len(negative),
            "resolution": "201-point threshold grid; pair scores are correlated within people/captures"}


def _operating_metrics(captures: list[dict[str, object]], strategy: str,
                       threshold: float, margin: float | None) -> dict[str, float | int]:
    genuine_captures = [c for c in captures if c["role"] == "genuine"]
    impostor_captures = [c for c in captures if c["role"] == "impostor"]
    capture_rows = []
    for capture in captures:
        ranked = sorted(((float(row[strategy]), candidate)
                         for candidate, row in capture["scores"].items()), reverse=True)
        if not ranked:
            continue
        best_score, best_candidate = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else None
        gap = None if second_score is None else best_score - second_score
        margin_ok = margin is None or (gap is not None and gap >= margin)
        accepted = best_score >= threshold and margin_ok
        correct = bool(capture["role"] == "genuine" and best_candidate == capture["expected"] and accepted)
        capture_rows.append((capture, accepted, correct))
    false_accepts = sum(accepted for c, accepted, _correct in capture_rows if c["role"] == "impostor")
    false_rejects = sum(not correct for c, _accepted, correct in capture_rows if c["role"] == "genuine")
    participant_outcomes: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for capture, accepted, correct in capture_rows:
        participant_outcomes[str(capture["participant_id"])].append(
            (str(capture["role"]), accepted if capture["role"] == "impostor" else not correct)
        )
    # A cluster-level event counts each person once, regardless of capture count.
    impostor_people = {p for p, values in participant_outcomes.items() if any(role == "impostor" for role, _ in values)}
    genuine_people = {p for p, values in participant_outcomes.items() if any(role == "genuine" for role, _ in values)}
    person_far = sum(any(role == "impostor" and event for role, event in participant_outcomes[p])
                     for p in impostor_people) / max(1, len(impostor_people))
    person_frr = sum(any(role == "genuine" and event for role, event in participant_outcomes[p])
                     for p in genuine_people) / max(1, len(genuine_people))
    return {
        "threshold": float(threshold), "margin": margin,
        "genuine_captures": len(genuine_captures), "impostor_captures": len(impostor_captures),
        "capture_far": false_accepts / max(1, len(impostor_captures)),
        "capture_frr": false_rejects / max(1, len(genuine_captures)),
        "person_any_capture_far": float(person_far), "person_any_capture_frr": float(person_frr),
    }


def evaluate(rows: list[dict[str, str]], thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
             frozen_threshold: float = 0.65, frozen_margin: float | None = None) -> dict[str, object]:
    if not rows:
        raise ValueError("no evaluation rows supplied")
    thresholds = tuple(float(value) for value in thresholds)
    for threshold in thresholds:
        if not math.isfinite(float(threshold)) or not -1 <= float(threshold) <= 1:
            raise ValueError("thresholds must be finite and in [-1,1]")
    if not -1 <= frozen_threshold <= 1:
        raise ValueError("frozen_threshold must be in [-1,1]")
    partitions = {}
    for partition in ("calibration", "test"):
        captures = _capture_groups(rows, partition)
        if not captures:
            continue
        people_by_role = {role: len({c["participant_id"] for c in captures if c["role"] == role})
                          for role in ("genuine", "impostor")}
        sessions_by_role = {role: len({c["session_id"] for c in captures if c["role"] == role})
                            for role in ("genuine", "impostor")}
        enough = all(people_by_role[r] >= 20 and sessions_by_role[r] >= 2
                     for r in ("genuine", "impostor"))
        by_strategy = {}
        for strategy in STRATEGIES[:3]:
            margins = MARGIN_GRID if partition == "calibration" else (frozen_margin,)
            threshold_grid = list(thresholds) if partition == "calibration" else [frozen_threshold]
            operating = [_operating_metrics(captures, strategy, threshold, margin)
                         for threshold in threshold_grid for margin in margins]
            pairwise = {"thresholds": [
                {"threshold": float(threshold),
                 "far": _pair_rates(captures, strategy, float(threshold))[0],
                 "frr": _pair_rates(captures, strategy, float(threshold))[1]}
                for threshold in threshold_grid
            ]}
            by_strategy[strategy] = {
                "score_distribution": _score_distribution(captures, strategy),
                "operating_point_sweep": operating,
                "pairwise_threshold_sweep": pairwise["thresholds"],
                "roc_eer": _eer_and_roc(captures, strategy),
            }
        by_strategy["quality_weighted"] = {
            "status": "not evaluable",
            "reason": "current enrollment store contains embeddings only, with no per-template capture quality metadata",
        }
        partitions[partition] = {
            "captures": len(captures), "participants_by_role": people_by_role,
            "sessions_by_role": sessions_by_role,
            "threshold_calibration_adequacy": "minimum heuristic met" if enough else
                "INCOMPLETE: require >=20 distinct people per role and >=2 capture sessions per role",
            "strategies": by_strategy,
        }
    if "calibration" in partitions and "test" in partitions:
        calib_sessions = {c["session_id"] for c in _capture_groups(rows, "calibration")}
        test_sessions = {c["session_id"] for c in _capture_groups(rows, "test")}
        if calib_sessions & test_sessions:
            raise ValueError("calibration and test sessions overlap")
    return {
        "method": "score-only, group-aware session-separated evaluation",
        "current_production_strategy": "mean of up to top 3 cosine scores per identity",
        "current_threshold": 0.65,
        "current_threshold_status": "provisional; not calibrated by this tool",
        "threshold_recommendation": None,
        "threshold_recommendation_status": "not provided; inspect calibration results, then freeze threshold and evaluate on untouched test sessions",
        "partitions": partitions,
        "limitations": [
            "threshold metrics are not independent evidence per frame; participant-clustered rates are also shown",
            "minimum cohort heuristic is not a formal accuracy guarantee",
            "quality-weighted aggregation cannot be computed without quality metadata for each enrolled template",
            "ROC/EER are pairwise verification summaries and do not replace open-set identification FAR/FRR",
        ],
    }
