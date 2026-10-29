
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(RAW_DIR.rglob("*.csv"))

if not files:
    raise SystemExit(f"No CSV files found in: {RAW_DIR}")

total_rows = 0
processed_files = 0

print(f"Found {len(files)} CSV files.")
print("Preparing processed copies...")

for input_path in files:
    relative_path = input_path.relative_to(RAW_DIR)

    # Preserve the original folder structure in processed data.
    output_path = PROCESSED_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        reader = csv.DictReader(source, delimiter=";")

        if not reader.fieldnames or "datetime" not in reader.fieldnames:
            print(f"SKIPPED: {relative_path} (missing datetime column)")
            continue

        # Keep every original column and append the source filename.
        fieldnames = list(reader.fieldnames)
        if "source_file" not in fieldnames:
            fieldnames.append("source_file")

        rows_written = 0

        with output_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=fieldnames,
                delimiter=";"
            )
            writer.writeheader()

            for row in reader:
                row["source_file"] = relative_path.as_posix()
                writer.writerow(row)
                rows_written += 1

    total_rows += rows_written
    processed_files += 1
    print(f"Processed: {relative_path} — {rows_written} rows")

print("\n" + "=" * 55)
print("DATASET PREPARATION COMPLETE")
print("=" * 55)
print(f"Files processed: {processed_files}")
print(f"Total rows copied: {total_rows}")
print(f"Output folder: {PROCESSED_DIR}")
print("Original raw files were not modified.")
