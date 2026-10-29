from pathlib import Path
import csv
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "sensor_anomaly_model.joblib"

FEATURES = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
TARGET = "anomaly"
THRESHOLD = 0.40


def load_data():
    frames = []

    for path in sorted(DATA_DIR.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            headers = next(csv.reader(file, delimiter=";"), [])

        if not all(c in headers for c in FEATURES + [TARGET]):
            continue

        df = pd.read_csv(path, sep=";")
        df = df[FEATURES + [TARGET]].copy()

        for col in FEATURES + [TARGET]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET].isin([0, 1])]

        if not df.empty:
            df["source_file"] = path.relative_to(DATA_DIR).as_posix()
            frames.append(df)

    if not frames:
        raise SystemExit("No labelled CSV files found.")

    data = pd.concat(frames, ignore_index=True)

    if data[TARGET].nunique() != 2:
        raise SystemExit("Both Normal and Anomaly labels are required.")

    return data


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )


def main():
    data = load_data()
    X = data[FEATURES]
    y = data[TARGET].astype(int)
    groups = data["source_file"]

    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )
    train_idx, test_idx = next(outer.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=21
    )
    fit_idx, val_idx = next(
        inner.split(X_train, y_train, groups=train_groups)
    )

    X_fit, y_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        raise SystemExit(
            "Fit/validation split lacks one class. Stop and adjust the split."
        )

    # Fit only on fit files to select the threshold on validation files.
    model = make_model()
    model.fit(X_fit, y_fit)

    anomaly_index = list(model.classes_).index(1)
    val_scores = model.predict_proba(X_val)[:, anomaly_index]
    val_predictions = (val_scores >= THRESHOLD).astype(int)

    print("=" * 60)
    print("PHANTOM DETECTION AI - VALIDATION AND FINAL TEST")
    print("=" * 60)
    print(f"Threshold selected from prior exploration: {THRESHOLD:.2f}")
    print("\nValidation metrics:")
    print(f"Precision: {precision_score(y_val, val_predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_val, val_predictions, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(y_val, val_predictions, zero_division=0):.3f}")

    # Retrain using all outer training files, then evaluate once on held-out files.
    final_model = make_model()
    final_model.fit(X_train, y_train)

    if y_test.nunique() < 2:
        raise SystemExit(
            "Reserved test files lack one class. Cannot report a meaningful final test."
        )

    test_scores = final_model.predict_proba(X_test)[:, anomaly_index]
    test_predictions = (test_scores >= THRESHOLD).astype(int)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("-" * 60)
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Training files: {train_groups.nunique()}")
    print(f"Testing files: {groups.iloc[test_idx].nunique()}")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, test_predictions, zero_division=0):.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

    # Save the final model and threshold for later API integration.
    joblib.dump(
        {
            "model": final_model,
            "sensor_columns": FEATURES,
            "target_column": TARGET,
            "threshold": THRESHOLD,
        },
        MODEL_PATH,
    )

    print(f"\nFinal model and threshold saved to: {MODEL_PATH}")
    print("This model detects labelled sensor anomalies, not fraud itself.")


if __name__ == "__main__":
    main()
