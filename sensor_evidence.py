
import csv
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "sensor_anomaly_model.joblib"
OUTPUT_PATH = (
    PROJECT_DIR / "data" / "synthetic" / "sensor_evidence_summary.csv"
)


def main():
    # Check that the trained model exists
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found: {MODEL_PATH}\n"
            "Run sensor_model.py first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    sensor_columns = saved["sensor_columns"]

    summaries = []
    files = sorted(DATA_DIR.rglob("*.csv"))

    print("=" * 60)
    print("PHANTOM DETECTION AI - SENSOR EVIDENCE")
    print("=" * 60)

    for path in files:
        # Read headers using the SKAB semicolon delimiter
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            headers = next(csv.reader(file, delimiter=";"), [])

        if not all(column in headers for column in sensor_columns):
            print(f"Skipping incompatible file: {path.name}")
            continue

        df = pd.read_csv(path, sep=";")

        # Convert sensor features to numeric values
        X = df[sensor_columns].apply(pd.to_numeric, errors="coerce")

        # Keep rows containing at least one usable sensor value
        valid_rows = X.notna().any(axis=1)
        X = X.loc[valid_rows]

        if X.empty:
            print(f"Skipping file with no usable readings: {path.name}")
            continue

        # Predict normal (0) or anomalous (1) readings
        predictions = model.predict(X)

        total_readings = len(predictions)
        anomaly_count = int((predictions == 1).sum())
        normal_count = total_readings - anomaly_count
        anomaly_rate = anomaly_count / total_readings * 100

        result = {
            "source_file": path.relative_to(DATA_DIR).as_posix(),
            "total_readings": total_readings,
            "predicted_anomalies": anomaly_count,
            "predicted_normal": normal_count,
            "predicted_anomaly_rate_pct": round(anomaly_rate, 2),
        }

        # Compare with dataset labels when available
        if "anomaly" in df.columns:
            labels = pd.to_numeric(
                df.loc[X.index, "anomaly"], errors="coerce"
            )
            labelled = labels.isin([0, 1])

            if labelled.any():
                true_labels = labels.loc[labelled].astype(int)
                predicted_labels = predictions[labelled.to_numpy()]

                result["labelled_readings"] = int(labelled.sum())
                result["ground_truth_anomaly_rate_pct"] = round(
                    true_labels.mean() * 100, 2
                )
                result["file_accuracy_pct"] = round(
                    accuracy_score(
                        true_labels, predicted_labels
                    ) * 100,
                    2,
                )
            else:
                result["labelled_readings"] = 0
                result["ground_truth_anomaly_rate_pct"] = None
                result["file_accuracy_pct"] = None
        else:
            result["labelled_readings"] = 0
            result["ground_truth_anomaly_rate_pct"] = None
            result["file_accuracy_pct"] = None

        summaries.append(result)

    if not summaries:
        raise SystemExit(
            "No compatible sensor files found. "
            "Check the files in data/processed."
        )

    # Save the results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFiles summarized: {len(summary_df)}")
    print("Total readings:", int(summary_df["total_readings"].sum()))
    print(
        "Predicted anomalies:",
        int(summary_df["predicted_anomalies"].sum()),
    )

    print("\nFirst 10 recording summaries:")
    print(summary_df.head(10).to_string(index=False))

    print(f"\nSummary saved to:\n{OUTPUT_PATH}")
    print("\nNote: Anomalies indicate unusual sensor behavior,")
    print("not confirmed production quantity discrepancies.")


if __name__ == "__main__":
    main()
