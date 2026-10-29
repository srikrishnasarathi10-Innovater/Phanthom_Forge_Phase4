
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "synthetic"
    / "erp_production_records.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "synthetic"
    / "erp_discrepancy_results.csv"
)


def calculate_discrepancy(record):
    recorded = int(record["recorded_quantity"])
    verified = int(record["verified_quantity"])

    difference = recorded - verified

    if recorded != 0:
        discrepancy_percentage = abs(difference) / abs(recorded) * 100
    else:
        discrepancy_percentage = 0.0 if verified == 0 else 100.0

    if difference == 0:
        status = "NORMAL"
    else:
        status = "QUANTITY_DISCREPANCY"

    result = dict(record)
    result["quantity_difference"] = difference
    result["discrepancy_percentage"] = round(
        discrepancy_percentage, 2
    )
    result["risk_status"] = status

    return result


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as source:
        reader = csv.DictReader(source)
        records = [
            calculate_discrepancy(row)
            for row in reader
        ]

    if not records:
        raise SystemExit("No ERP records found.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys())

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(records)

    flagged = sum(
        record["risk_status"] != "NORMAL"
        for record in records
    )

    print("ERP discrepancy calculation complete.")
    print(f"Records checked: {len(records)}")
    print(f"Records flagged: {flagged}")
    print(f"Normal records: {len(records) - flagged}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
