
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

# --------------------------------------------------
# PHANTOM DETECTION AI - FASTAPI BACKEND
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

CASES_PATH = (
    PROJECT_DIR / "data" / "synthetic" / "investigation_cases.csv"
)

app = FastAPI(
    title="PHANTOM DETECTION AI",
    description=(
        "ERP discrepancy investigation with sensor anomaly evidence. "
        "Sensor-to-ERP mappings in this prototype are simulated."
    ),
    version="1.0.0",
)


def load_cases():
    """Load the latest generated investigation cases."""

    if not CASES_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Cases file not found. Run investigation_cases.py first.",
        )

    try:
        return pd.read_csv(CASES_PATH)

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read investigation cases: {exc}",
        ) from exc


@app.get("/")
def home():
    return {
        "project": "PHANTOM DETECTION AI",
        "status": "running",
        "message": "ERP and sensor investigation API is ready.",
        "documentation": "/docs",
        "sensor_erp_mapping": "SIMULATED_DEMO_MAPPING",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "cases_file_exists": CASES_PATH.exists(),
    }


@app.get("/cases")
def get_cases(
    priority: str | None = Query(
        default=None,
        description="Filter by HIGH, MEDIUM, or LOW",
    ),
    erp_discrepancy_only: bool = False,
    limit: int = Query(default=50, ge=1, le=500),
):
    df = load_cases()

    if priority is not None:
        priority = priority.strip().upper()

        if priority not in {"HIGH", "MEDIUM", "LOW"}:
            raise HTTPException(
                status_code=400,
                detail="priority must be HIGH, MEDIUM, or LOW",
            )

        df = df[df["investigation_priority"] == priority]

    if erp_discrepancy_only:
        df = df[df["erp_discrepancy_flag"].astype(str).str.lower().isin(
            ["true", "1"]
        )]

    df = df.head(limit)

    # Replace missing values so the JSON response is valid.
    records = (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )

    return {
        "count": len(records),
        "sensor_erp_mapping": "SIMULATED_DEMO_MAPPING",
        "cases": records,
    }


@app.get("/summary")
def get_summary():
    df = load_cases()

    priorities = df["investigation_priority"].value_counts().to_dict()

    erp_flags = df["erp_discrepancy_flag"].astype(str).str.lower()
    sensor_flags = df["elevated_sensor_anomaly_flag"].astype(str).str.lower()

    return {
        "total_orders": int(len(df)),
        "erp_discrepancy_cases": int(
            erp_flags.isin(["true", "1"]).sum()
        ),
        "elevated_sensor_indicator_cases": int(
            sensor_flags.isin(["true", "1"]).sum()
        ),
        "priority_counts": {
            "HIGH": int(priorities.get("HIGH", 0)),
            "MEDIUM": int(priorities.get("MEDIUM", 0)),
            "LOW": int(priorities.get("LOW", 0)),
        },
        "fraud_confirmed_cases": 0,
        "review_status": "Human review required",
        "sensor_erp_mapping": "SIMULATED_DEMO_MAPPING",
    }


@app.get("/cases/{order_id}")
def get_case(order_id: str):
    df = load_cases()

    matches = df[
        df["order_id"].astype(str).str.upper() == order_id.upper()
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation case found for {order_id}",
        )

    record = (
        matches.iloc[0:1]
        .astype(object)
        .where(pd.notna(matches.iloc[0:1]), None)
        .to_dict(orient="records")[0]
    )

    return record
