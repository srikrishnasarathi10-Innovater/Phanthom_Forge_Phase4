
import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

if not DATA_DIR.exists():
    raise SystemExit(f"Dataset directory not found: {DATA_DIR}")

csv_files = sorted(DATA_DIR.rglob("*.csv"))

if not csv_files:
    raise SystemExit(f"No CSV files found inside: {DATA_DIR}")

print("=" * 75)
print("PHANTOM DETECTION AI - SKAB DATASET INSPECTION")
print("=" * 75)
print(f"Dataset directory : {DATA_DIR}")
print(f"CSV files found   : {len(csv_files)}")

header_groups = defaultdict(list)
total_rows = 0
total_missing = Counter()
all_column_names = set()
all_columns = Counter()
timestamp_formats = Counter()
timestamp_failures = Counter()
label_candidates = set()

for file_path in csv_files:
    try:
        with file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file, delimiter=";")
            headers = reader.fieldnames

            if not headers:
                print(f"\nWARNING: No header found: {file_path.name}")
                continue

            headers = [h.strip() if h else "" for h in headers]
            header_groups[tuple(headers)].append(file_path)
            all_column_names.update(headers)
            all_columns.update(headers)

            for header in headers:
                if any(word in header.lower() for word in (
                    "anomaly", "changepoint", "label", "target"
                )):
                    label_candidates.add(header)

            row_count = 0
            missing = Counter()
            first_timestamp = None
            last_timestamp = None
            timestamp_column = next(
                (h for h in headers if h.lower() == "datetime"),
                None
            )

            for row in reader:
                row_count += 1

                for header in headers:
                    value = row.get(header)
                    if value is None or not value.strip():
                        missing[header] += 1

                if timestamp_column:
                    raw_timestamp = row.get(timestamp_column, "").strip()

                    if raw_timestamp:
                        try:
                            parsed = datetime.strptime(
                                raw_timestamp, "%Y-%m-%d %H:%M:%S"
                            )
                            if first_timestamp is None:
                                first_timestamp = parsed
                            last_timestamp = parsed
                            timestamp_formats["standard"] += 1
                        except ValueError:
                            timestamp_failures[file_path.name] += 1

            total_rows += row_count
            total_missing.update(missing)

            print(f"\nFile: {file_path.relative_to(DATA_DIR)}")
            print(f"  Rows: {row_count}")
            print(f"  Columns ({len(headers)}): {headers}")

            if timestamp_column:
                print(f"  First timestamp: {first_timestamp}")
                print(f"  Last timestamp : {last_timestamp}")
            else:
                print("  Timestamp column: not found")

            if missing:
                print(f"  Non-empty-field gaps: {dict(missing)}")

    except (OSError, UnicodeError, csv.Error) as error:
        print(f"\nERROR reading {file_path}: {error}")

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"Files inspected         : {len(csv_files)}")
print(f"Total data rows         : {total_rows}")
print(f"Distinct column names   : {len(all_column_names)}")
print(f"Column names             : {sorted(all_column_names)}")
print(f"Column occurrence counts : {dict(all_columns)}")
print(f"Candidate label columns  : {sorted(label_candidates)}")

print("\nFiles grouped by identical headers:")
for headers, files in header_groups.items():
    print(f"\n{len(files)} file(s): {headers}")
    for path in files:
        print(f"  - {path.relative_to(DATA_DIR)}")

print("\nTimestamp parsing issues:")
if timestamp_failures:
    for name, count in timestamp_failures.items():
        print(f"  {name}: {count} unparsed timestamp(s)")
else:
    print("  No parsing failures detected in non-empty datetime values.")

print("\nInspection complete. Original CSV files were not modified.")
