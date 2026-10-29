
from pathlib import Path
import pandas as pd


# ============================================================
# PHANTOM DETECTION AI — INVESTIGATION CASE BUILDER
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "synthetic"

ERP_PATH = DATA_DIR / "erp_discrepancy_results.csv"
SENSOR_PATH = DATA_DIR / "sensor_evidence_summary.csv"
MAPPING_PATH = DATA_DIR / "sensor_machine_demo_mapping.csv"
OUTPUT_PATH = DATA_DIR / "investigation_cases.csv"

# Demo-only triage thresholds. These are not validated
# industrial thresholds and do not establish fraud.
HIGH_DISCREPANCY_PCT = 10.0
ELEVATED_SENSOR_ANOMALY_PCT = 30.0


def require_file(path):
    """Stop with a clear message if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def main():
    print("=" * 65)
    print("PHANTOM DETECTION AI - INVESTIGATION CASE BUILDER")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check and load existing project data
    # --------------------------------------------------------
    for path in [ERP_PATH, SENSOR_PATH, MAPPING_PATH]:
        require_file(path)

    erp = pd.read_csv(ERP_PATH)
    sensor = pd.read_csv(SENSOR_PATH)
    mapping = pd.read_csv(MAPPING_PATH)

    # Clean column names.
    erp.columns = erp.columns.str.strip()
    sensor.columns = sensor.columns.str.strip()
    mapping.columns = mapping.columns.str.strip()

    # Validate required columns.
    required_erp = {
        "order_id",
        "machine_id",
        "discrepancy_percentage",
    }
    required_sensor = {
        "source_file",
        "total_readings",
        "predicted_anomalies",
        "predicted_anomaly_rate_pct",
    }
    required_mapping = {
        "machine_id",
        "source_file",
        "mapping_type",
    }

    for name, frame, required in [
        ("ERP data", erp, required_erp),
        ("Sensor summary", sensor, required_sensor),
        ("Demo mapping", mapping, required_mapping),
    ]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: "
                f"{sorted(missing)}"
            )

    # Normalize identifiers to avoid whitespace mismatches.
    for frame in [erp, mapping]:
        frame["machine_id"] = (
            frame["machine_id"].astype("string").str.strip()
        )

    sensor["source_file"] = (
        sensor["source_file"].astype("string").str.strip()
    )
    mapping["source_file"] = (
        mapping["source_file"].astype("string").str.strip()
    )

    # Check mapping integrity.
    if mapping["machine_id"].duplicated().any():
        duplicates = mapping.loc[
            mapping["machine_id"].duplicated(keep=False),
            "machine_id",
        ].tolist()
        raise ValueError(
            "Each machine must have only one demo mapping. "
            f"Duplicates: {duplicates}"
        )

    if sensor["source_file"].duplicated().any():
        raise ValueError(
            "Sensor summary contains duplicate source_file values."
        )

    unknown_files = (
        set(mapping["source_file"].dropna())
        - set(sensor["source_file"].dropna())
    )
    if unknown_files:
        raise ValueError(
            "Mapped files were not found in the sensor summary: "
            f"{sorted(unknown_files)}"
        )

    # --------------------------------------------------------
    # 2. Attach the explicitly configured demo mapping
    # --------------------------------------------------------
    mapping_for_join = mapping.rename(
        columns={
            "source_file": "sensor_source_file",
            "mapping_type": "sensor_mapping_type",
        }
    )

    cases = erp.merge(
        mapping_for_join[
            [
                "machine_id",
                "sensor_source_file",
                "sensor_mapping_type",
            ]
        ],
        on="machine_id",
        how="left",
        validate="many_to_one",
    )

    # Attach statistics for the selected recording.
    sensor_for_join = sensor[
        [
            "source_file",
            "total_readings",
            "predicted_anomalies",
            "predicted_anomaly_rate_pct",
        ]
    ].rename(
        columns={
            "source_file": "sensor_source_file",
            "total_readings": "sensor_total_readings",
            "predicted_anomalies": "sensor_predicted_anomalies",
            "predicted_anomaly_rate_pct": "sensor_anomaly_rate_pct",
        }
    )

    cases = cases.merge(
        sensor_for_join,
        on="sensor_source_file",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # 3. Label simulated versus missing sensor associations
    # --------------------------------------------------------
    has_mapping = cases["sensor_source_file"].notna()
    has_sensor_data = cases["sensor_anomaly_rate_pct"].notna()

    cases["sensor_link_type"] = "NO_DEMO_MAPPING"
    cases.loc[has_mapping, "sensor_link_type"] = (
        "SIMULATED_DEMO_MAPPING"
    )
    cases.loc[
        has_mapping & ~has_sensor_data, "sensor_link_type"
    ] = "MAPPED_SENSOR_DATA_MISSING"

    # SKAB recordings are not verified recordings from ERP machines.
    cases["real_sensor_erp_link_verified"] = False

    # --------------------------------------------------------
    # 4. Calculate investigation priority
    # --------------------------------------------------------
    cases["discrepancy_percentage"] = pd.to_numeric(
        cases["discrepancy_percentage"],
        errors="coerce",
    ).fillna(0.0)

    cases["sensor_anomaly_rate_pct"] = pd.to_numeric(
        cases["sensor_anomaly_rate_pct"],
        errors="coerce",
    )

    erp_discrepancy = cases["discrepancy_percentage"].abs() > 0
    high_discrepancy = (
        cases["discrepancy_percentage"].abs()
        >= HIGH_DISCREPANCY_PCT
    )
    elevated_sensor = (
        cases["sensor_anomaly_rate_pct"]
        >= ELEVATED_SENSOR_ANOMALY_PCT
    ).fillna(False)

    cases["erp_discrepancy_flag"] = erp_discrepancy
    cases["elevated_sensor_anomaly_flag"] = elevated_sensor

    cases["investigation_priority"] = "LOW"
    cases.loc[
        erp_discrepancy | elevated_sensor,
        "investigation_priority",
    ] = "MEDIUM"
    cases.loc[
        high_discrepancy,
        "investigation_priority",
    ] = "HIGH"

    # --------------------------------------------------------
    # 5. Add responsible, explainable review fields
    # --------------------------------------------------------
    cases["investigation_status"] = "PENDING_HUMAN_REVIEW"
    cases["fraud_confirmed"] = False

    cases["priority_reason"] = "No configured demo trigger detected"
    cases.loc[
        elevated_sensor,
        "priority_reason",
    ] = "Elevated sensor-model anomaly rate; review required"
    cases.loc[
        erp_discrepancy,
        "priority_reason",
    ] = "ERP quantity discrepancy; review required"
    cases.loc[
        erp_discrepancy & elevated_sensor,
        "priority_reason",
    ] = "ERP discrepancy and elevated sensor anomaly rate"
    cases.loc[
        high_discrepancy,
        "priority_reason",
    ] = "ERP discrepancy meets the demo HIGH-priority threshold"

    # Always retain the simulation and human-review caveats.
    cases["evidence_limitation"] = (
        "Sensor-to-ERP association is simulated; "
        "sensor anomalies do not prove an ERP discrepancy or fraud. "
        "Human review is required."
    )

    # --------------------------------------------------------
    # 6. Save and display results
    # --------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases.to_csv(OUTPUT_PATH, index=False)

    print(f"\nERP orders processed: {len(erp)}")
    print(f"Available sensor recordings: {len(sensor)}")
    print(
        "ERP discrepancies: "
        f"{int(erp_discrepancy.sum())}"
    )
    print(
        "Cases with elevated sensor anomaly rates: "
        f"{int(elevated_sensor.sum())}"
    )

    print("\nInvestigation priority counts:")
    print(
        cases["investigation_priority"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
        .to_string()
    )

    print("\nFirst 10 investigation cases:")
    display_columns = [
        "order_id",
        "machine_id",
        "discrepancy_percentage",
        "sensor_source_file",
        "sensor_anomaly_rate_pct",
        "investigation_priority",
        "sensor_link_type",
    ]
    print(cases[display_columns].head(10).to_string(index=False))

    print(f"\nSaved investigation cases to:\n{OUTPUT_PATH}")
    print("\nIMPORTANT:")
    print("- Sensor-to-ERP assignments are simulated.")
    print("- Sensor anomalies do not establish quantity discrepancies.")
    print("- Priority scores are triage indicators, not proof of fraud.")
    print("- All cases require appropriate human review.")


if __name__ == "__main__":
    main()
