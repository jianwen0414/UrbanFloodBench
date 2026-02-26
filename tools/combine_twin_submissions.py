import csv
from pathlib import Path
from typing import Dict, Optional, Tuple


EXPECTED_FIELDS = ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]


def _key(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["model_id"]), int(row["event_id"])


def _write_event_rows(
    reader: csv.DictReader,
    current_row: Dict[str, str],
    target_key: Tuple[int, int],
    expected_node_type: str,
    writer: csv.DictWriter,
    start_row_id: int,
) -> Tuple[Optional[Dict[str, str]], int, int]:
    row_id = start_row_id
    n_written = 0
    n_nodes = set()

    row = current_row
    while row is not None and _key(row) == target_key:
        if row["node_type"] != expected_node_type:
            raise ValueError(
                f"Unexpected node_type in event {target_key}: "
                f"expected {expected_node_type}, got {row['node_type']}"
            )

        out_row = {
            "row_id": str(row_id),
            "model_id": str(int(row["model_id"])),
            "event_id": str(int(row["event_id"])),
            "node_type": str(int(row["node_type"])),
            "node_id": str(int(row["node_id"])),
            "water_level": row["water_level"],
        }
        writer.writerow(out_row)

        n_nodes.add(int(row["node_id"]))
        row_id += 1
        n_written += 1
        row = next(reader, None)

    return row, n_written, len(n_nodes)


def combine_submissions(
    file_1d: Path,
    file_2d: Path,
    output_file: Path,
) -> None:
    if not file_1d.exists():
        raise FileNotFoundError(f"Missing file: {file_1d}")
    if not file_2d.exists():
        raise FileNotFoundError(f"Missing file: {file_2d}")

    with (
        file_1d.open("r", newline="", encoding="utf-8") as f1,
        file_2d.open("r", newline="", encoding="utf-8") as f2,
        output_file.open("w", newline="", encoding="utf-8") as fout,
    ):
        r1 = csv.DictReader(f1)
        r2 = csv.DictReader(f2)

        if r1.fieldnames != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected columns in {file_1d.name}: {r1.fieldnames}")
        if r2.fieldnames != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected columns in {file_2d.name}: {r2.fieldnames}")

        writer = csv.DictWriter(fout, fieldnames=EXPECTED_FIELDS)
        writer.writeheader()

        pending_1d = next(r1, None)
        pending_2d = next(r2, None)

        row_id = 0
        event_count = 0
        rows_1d_total = 0
        rows_2d_total = 0

        while pending_1d is not None and pending_2d is not None:
            key_1d = _key(pending_1d)
            key_2d = _key(pending_2d)

            if key_1d != key_2d:
                raise ValueError(
                    f"Event mismatch between files: 1D has {key_1d}, 2D has {key_2d}"
                )

            pending_1d, n_1d, n_nodes_1d = _write_event_rows(
                r1, pending_1d, key_1d, "1", writer, row_id
            )
            row_id += n_1d
            rows_1d_total += n_1d

            pending_2d, n_2d, n_nodes_2d = _write_event_rows(
                r2, pending_2d, key_2d, "2", writer, row_id
            )
            row_id += n_2d
            rows_2d_total += n_2d

            event_count += 1
            if event_count <= 3 or event_count % 10 == 0:
                print(
                    f"Processed event {key_1d}: "
                    f"1D rows={n_1d:,} ({n_nodes_1d} nodes), "
                    f"2D rows={n_2d:,} ({n_nodes_2d} nodes), "
                    f"running total={row_id:,}"
                )

        if pending_1d is not None or pending_2d is not None:
            raise ValueError("One file has extra trailing events.")

    print("\n=== Combine Complete ===")
    print(f"Events processed : {event_count}")
    print(f"1D rows written  : {rows_1d_total:,}")
    print(f"2D rows written  : {rows_2d_total:,}")
    print(f"Total rows       : {rows_1d_total + rows_2d_total:,}")
    print(f"Output           : {output_file}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    file_1d = root / "submission_1d.csv"
    file_2d = root / "submission_2d.csv"
    output_file = root / "submission.csv"

    combine_submissions(file_1d, file_2d, output_file)


if __name__ == "__main__":
    main()
