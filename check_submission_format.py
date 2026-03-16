#!/usr/bin/env python3
"""
Check that submission_full_with_1d.csv has the same format as submission_full_ensemble.csv.
Run: python check_submission_format.py
"""
import csv
from pathlib import Path

EXPECTED_HEADER = [
    "row_id",
    "model_id",
    "event_id",
    "node_type",
    "node_id",
    "water_level",
]
EXPECTED_ROWS = 50_910_192


def get_format(path: Path) -> dict:
    """Read header, row count, and node types from a submission CSV."""
    if not path.exists():
        return {"error": f"File not found: {path}"}
    header = None
    node_types = set()
    row_count = 0
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row_count += 1
            if len(row) >= 4:
                try:
                    node_types.add(int(row[3]))
                except ValueError:
                    pass
    return {
        "header": header,
        "row_count": row_count,
        "node_types": sorted(node_types),
        "columns": len(header) if header else 0,
    }


def main():
    base = Path("submissions")
    ref_path = base / "submission_full_ensemble.csv"
    check_path = base / "submission_full_with_1d.csv"

    print("Comparing submission format")
    print("  Reference: submission_full_ensemble.csv")
    print("  To check:  submission_full_with_1d.csv")
    print()

    ref = get_format(ref_path)
    if "error" in ref:
        print(f"  Reference: {ref['error']}")
        print("  (Optional: submission_full_ensemble.csv is from add_dummy_1d on 2D ensemble)")
        ref = None
    else:
        print("Reference (submission_full_ensemble.csv):")
        print(f"  header:    {ref['header']}")
        print(f"  columns:   {ref['columns']}")
        print(f"  row_count: {ref['row_count']:,}")
        print(f"  node_types: {ref['node_types']}")
        print()

    check = get_format(check_path)
    if "error" in check:
        print(f"Check file: {check['error']}")
        return 1
    print("To check (submission_full_with_1d.csv):")
    print(f"  header:    {check['header']}")
    print(f"  columns:   {check['columns']}")
    print(f"  row_count: {check['row_count']:,}")
    print(f"  node_types: {check['node_types']}")
    print()

    # Format requirements
    ok = True
    if check["header"] != EXPECTED_HEADER:
        print(f"  ERROR: header should be {EXPECTED_HEADER}")
        ok = False
    if check["columns"] != 6:
        print(f"  ERROR: expected 6 columns, got {check['columns']}")
        ok = False
    if check["row_count"] != EXPECTED_ROWS:
        print(f"  ERROR: expected {EXPECTED_ROWS:,} rows, got {check['row_count']:,}")
        ok = False
    if set(check["node_types"]) != {1, 2}:
        print(f"  ERROR: expected node_types [1, 2], got {check['node_types']}")
        ok = False

    if ref and not ("error" in ref):
        if ref["header"] != check["header"]:
            print("  ERROR: header differs from reference")
            ok = False
        if ref["columns"] != check["columns"]:
            print("  ERROR: column count differs from reference")
            ok = False
        if ref["row_count"] != check["row_count"]:
            print("  ERROR: row count differs from reference")
            ok = False
        if set(ref["node_types"]) != set(check["node_types"]):
            print("  ERROR: node_types differ from reference")
            ok = False

    if ok:
        print("Format matches reference and requirements. Safe to submit.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
