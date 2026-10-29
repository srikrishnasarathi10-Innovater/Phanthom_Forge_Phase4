
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIR / "erp_production_records.csv"

# Reproducible demo data: the same records are generated each run.
random.seed(42)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MACHINES = ["MACH-01", "MACH-02", "MACH-03", "MACH-04"]
PRODUCTS = ["PROD-A", "PROD-B", "PROD-C"]
START_TIME = datetime(2026, 10, 20, 8, 0, 0)

FIELDS = [
    "order_id",
    "machine_id",
    "product_id",
    "order_start",
    "order_end",
    "expected_quantity",
    "recorded_quantity",
    "verified_quantity",
    "unit",
    "production_status",
    "erp_record_type",
]

def make_record(index):
    expected = random.choice([500, 800, 1000, 1200, 1500])

    # Most records agree; selected demo cases contain discrepancies.
    scenario = index % 10

    if scenario == 0:
        recorded = expected
        verified = expected - random.randint(80, 180)
        status = "Completed"
        record_type = "Quantity discrepancy"
    elif scenario == 1:
        recorded = expected + random.randint(50, 120)
        verified = expected
        status = "Completed"
        record_type = "Over-recording"
    elif scenario == 2:
        recorded = expected
        verified = expected
        status = "Completed"
        record_type = "Normal"
    elif scenario == 3:
        recorded = expected
        verified = expected
        status = "Completed"
        record_type = "Normal"
    else:
        recorded = expected
        verified = expected
        status = "Completed"
        record_type = "Normal"

    start = START_TIME + timedelta(hours=index * 2)
    end = start + timedelta(minutes=random.randint(30, 120))

    return {
        "order_id": f"ORD-{1001 + index}",
        "machine_id": MACHINES[index % len(MACHINES)],
        "product_id": random.choice(PRODUCTS),
        "order_start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "order_end": end.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_quantity": expected,
        "recorded_quantity": recorded,
        "verified_quantity": verified,
        "unit": "units",
        "production_status": status,
        "erp_record_type": record_type,
    }

def main():
    records = [make_record(i) for i in range(50)]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    discrepancies = sum(
        row["recorded_quantity"] != row["verified_quantity"]
        for row in records
    )

    print("Synthetic ERP dataset created successfully.")
    print(f"Records generated: {len(records)}")
    print(f"Records with quantity discrepancies: {discrepancies}")
    print(f"Output file: {OUTPUT_FILE}")
    print("These are synthetic demo records, not real factory data.")

if __name__ == "__main__":
    main()
