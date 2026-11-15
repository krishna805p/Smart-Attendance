"""Analyze score-only calibration and held-out test CSVs; never reads embeddings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation_protocol import DEFAULT_THRESHOLDS, evaluate, load_score_rows


FACE_SYSTEM_DIR = Path(__file__).resolve().parent


def parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(x) or not -1 <= x <= 1 for x in values):
        raise argparse.ArgumentTypeError("thresholds must be finite values in [-1,1]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=FACE_SYSTEM_DIR / "data" / "evaluation_protocol.csv")
    parser.add_argument("--output", type=Path, default=FACE_SYSTEM_DIR / "data" / "evaluation_summary.json")
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--frozen-threshold", type=float, default=0.65)
    parser.add_argument("--frozen-margin", type=float, default=None)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(FACE_SYSTEM_DIR.resolve()):
        parser.error("--output must remain inside Backend/face_system/")
    report = evaluate(load_score_rows(args.scores), args.thresholds,
                      args.frozen_threshold, args.frozen_margin)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        "summary_path": str(output),
        "threshold_recommendation": report["threshold_recommendation"],
        "calibration": {
            name: {"score_distribution": data["score_distribution"],
                  "roc_eer": data["roc_eer"],
                  "adequacy": data["threshold_calibration_adequacy"]}
            for name, data in report.get("partitions", {}).get("calibration", {}).get("strategies", {}).items()
            if name != "quality_weighted"
        },
        "test": {
            name: {"frozen_operating_results": data["operating_point_sweep"],
                  "roc_eer": data["roc_eer"]}
            for name, data in report.get("partitions", {}).get("test", {}).get("strategies", {}).items()
            if name != "quality_weighted"
        },
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
