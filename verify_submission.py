#!/usr/bin/env python3
"""
Verify submission file has both 1D and 2D rows before uploading to Kaggle.

Kaggle expects BOTH node types:
  - node_type=1 → 1D drainage nodes
  - node_type=2 → 2D surface nodes

If you see "Submission node_types [2] do not match solution node_types [1, 2]",
you uploaded a 2D-ONLY file. You must upload the MERGED file (see below).

Run: python verify_submission.py [path_to_submission.csv]
"""
import csv
import sys
from pathlib import Path

EXPECTED_TOTAL = 50_910_192
CORRECT_FILE = "submissions/submission_full_with_1d.csv"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CORRECT_FILE
    path = Path(path)

    if not path.exists():
        print(f"File not found: {path}")
        print("Generate full submission with: python run_1d_2d_submission.py")
        print("Then verify: python verify_submission.py")
        sys.exit(1)

    total = 0
    type1 = 0
    type2 = 0
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]:
            print(f"Unexpected header: {header}")
        for row in reader:
            total += 1
            nt = int(row[3])
            if nt == 1:
                type1 += 1
            elif nt == 2:
                type2 += 1

    print(f"File: {path.resolve()}")
    print(f"  Total rows:    {total:,}")
    print(f"  node_type=1:   {type1:,} (1D drainage)")
    print(f"  node_type=2:   {type2:,} (2D surface)")
    print(f"  Expected:     {EXPECTED_TOTAL:,}")

    ok = total == EXPECTED_TOTAL and type1 > 0 and type2 > 0
    if ok:
        print("\n  OK — Submit THIS file to Kaggle.")
        if path.name != "submission_full_with_1d.csv":
            print(f"  (Or use the standard path: {CORRECT_FILE})")
    else:
        if total != EXPECTED_TOTAL:
            print(f"\n  ERROR: Row count should be {EXPECTED_TOTAL:,}, got {total:,}")
        if type1 == 0:
            print(
                "  ERROR: No node_type=1 (1D) rows.\n"
                "  → You must upload the MERGED file, not 2D-only!\n"
                "  → Correct file: submissions/submission_full_with_1d.csv\n"
                "  → Generate it: python run_1d_2d_submission.py"
            )
        if type2 == 0:
            print("  ERROR: No node_type=2 (2D) rows.")
        sys.exit(1)


if __name__ == "__main__":
    main()
