"""Interactive local student face enrollment."""

from __future__ import annotations

import argparse

from enrollment import enroll_from_camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll", help="student roll number; prompted if omitted")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples", type=int, default=15, choices=range(15, 21))
    parser.add_argument("--replace", action="store_true",
                        help="explicitly replace an existing roll-number record")
    args = parser.parse_args()
    roll = args.roll or input("Student roll number: ").strip()
    try:
        success = enroll_from_camera(roll, camera_index=args.camera,
                                     sample_count=args.samples, replace=args.replace)
        return 0 if success else 1
    except Exception as exc:
        print(f"Enrollment failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
