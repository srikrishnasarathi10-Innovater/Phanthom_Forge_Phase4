
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

LABEL_COLUMNS = ["anomaly", "changepoint"]

files = sorted(DATA_DIR.rglob("*.csv"))

if not files:
    raise SystemExit(f"No CSV files found in {DATA_DIR}")

total_rows = 0
total_missing = Counter()
total_invalid_numeric = Counter()
total_duplicate_timestamps = 0
total_invalid_timestamps = 0
label_values = {name: Counter() for name in LABEL_COLUMNS}

print(f"Inspecting {len(files)} CSV files")
print("Original files will not be modified.")

for path in files:
    row_count = 0
    missing = Counter()
    invalid_numeric = Counter()
    seen_timestamps = set()
    duplicate_timestamps = 0
    invalid_timestamps = 0
    file_labels = {name: Counter() for name in LABEL_COLUMNS}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        headers = reader.fieldnames or []

        if "datetime" not in headers:
            print(f"\nWARNING: Missing datetime column: {path.name}")
            continue

        for row in reader:
            row_count += 1

            timestamp_text = (row.get("datetime") or "").strip()

            if not timestamp_text:
                missing["datetime"] += 1
            else:
                try:
                    timestamp = datetime.strptime(
                        timestamp_text, "%Y-%m-%d %H:%M:%S"
                    )
                    if timestamp in seen_timestamps:
                        duplicate_timestamps += 1
                    seen_timestamps.add(timestamp)
                except ValueError:
                    invalid_timestamps += 1

            for column in SENSOR_COLUMNS:
                if column not in headers:
                    continue

                value = (row.get(column) or "").strip()

                if not value:
                    missing[column] += 1
                else:
                    try:
                        number = float(value)
                        if number != number or number in (
                            float("inf"), float("-inf")
                        ):
                            invalid_numeric[column] += 1
                    except ValueError:
                        invalid_numeric[column] += 1

            for column in LABEL_COLUMNS:
                if column not in headers:
                    continue

                value = (row.get(column) or "").strip()
                if not value:
                    missing[column] += 1
                else:
                    file_labels[column][value] += 1
                    label_values[column][value] += 1

    total_rows += row_count
    total_missing.update(missing)
    total_invalid_numeric.update(invalid_numeric)
    total_duplicate_timestamps += duplicate_timestamps
    total_invalid_timestamps += invalid_timestamps

    print(f"\n{path.relative_to(DATA_DIR)}")
    print(f"  Rows: {row_count}")
    print(f"  Missing fields: {dict(missing) or 'None'}")
    print(f"  Invalid sensor values: {dict(invalid_numeric) or 'None'}")
    print(f"  Duplicate timestamps within file: {duplicate_timestamps}")
    print(f"  Invalid timestamps: {invalid_timestamps}")

    for column in LABEL_COLUMNS:
        if column in headers:
            print(f"  {column} values: {dict(file_labels[column])}")

print("\n" + "=" * 65)
print("DATA QUALITY SUMMARY")
print("=" * 65)
print(f"Files checked: {len(files)}")
print(f"Total data rows: {total_rows}")
print(f"Missing fields by column: {dict(total_missing) or 'None'}")
print(f"Invalid numeric values: {dict(total_invalid_numeric) or 'None'}")
print(f"Duplicate timestamps within files: {total_duplicate_timestamps}")
print(f"Invalid timestamps: {total_invalid_timestamps}")
print(f"All anomaly label counts: {dict(label_values['anomaly'])}")
print(f"All changepoint label counts: {dict(label_values['changepoint'])}")

print("\nQuality check complete. No files were modified.")
