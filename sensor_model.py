
import csv
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

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

TARGET_COLUMN = "anomaly"


def load_labelled_data():
    frames = []

    for path in sorted(DATA_DIR.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            headers = next(csv.reader(f, delimiter=";"), [])

        # The anomaly-free file does not have ground-truth labels.
        if TARGET_COLUMN not in headers:
            continue

        frame = pd.read_csv(path, sep=";")
        required = SENSOR_COLUMNS + [TARGET_COLUMN]

        if not all(column in frame.columns for column in required):
            print(f"Skipping unexpected schema: {path.name}")
            continue

        frame = frame[required].copy()
        frame["source_file"] = path.relative_to(DATA_DIR).as_posix()

        for column in SENSOR_COLUMNS + [TARGET_COLUMN]:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            )

        frame = frame.dropna(subset=[TARGET_COLUMN])
        frame = frame[frame[TARGET_COLUMN].isin([0, 1])]
        frames.append(frame)

    if not frames:
        raise SystemExit(
            "No labelled CSV files found. Check data/processed."
        )

    return pd.concat(frames, ignore_index=True)


def main():
    data = load_labelled_data()

    X = data[SENSOR_COLUMNS]
    y = data[TARGET_COLUMN].astype(int)
    groups = data["source_file"]

    # Hold out entire files for testing rather than random rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + "=" * 55)
    print("PHANTOM DETECTION AI — SENSOR MODEL BASELINE")
    print("=" * 55)
    print(f"Labelled rows loaded: {len(data)}")
    print(f"Labelled files loaded: {groups.nunique()}")
    print(f"Training rows: {len(train_idx)}")
    print(f"Testing rows: {len(test_idx)}")
    print(f"Training files: {groups.iloc[train_idx].nunique()}")
    print(f"Testing files: {groups.iloc[test_idx].nunique()}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save a portable model file for later API integration.
    import joblib

    model_path = MODEL_DIR / "sensor_anomaly_model.joblib"
    joblib.dump(
        {
            "model": model,
            "sensor_columns": SENSOR_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")
    print("This model predicts sensor anomalies, not fraud itself.")


if __name__ == "__main__":
    main()
