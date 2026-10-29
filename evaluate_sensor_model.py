from pathlib import Path
import csv
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_PATH = PROJECT_DIR / "models" / "sensor_anomaly_model.joblib"

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
TARGET = "anomaly"


def load_data():
    frames = []

    for path in sorted(DATA_DIR.rglob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            headers = next(csv.reader(f, delimiter=";"), [])

        required = SENSOR_COLUMNS + [TARGET]
        if not all(column in headers for column in required):
            continue

        df = pd.read_csv(path, sep=";")
        df = df[required].copy()

        for column in required:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=[TARGET])
        df = df[df[TARGET].isin([0, 1])]

        if not df.empty:
            df["source_file"] = path.relative_to(DATA_DIR).as_posix()
            frames.append(df)

    if not frames:
        raise SystemExit("No labelled CSV data found.")

    data = pd.concat(frames, ignore_index=True)

    if data[TARGET].nunique() != 2:
        raise SystemExit("Dataset must contain both Normal and Anomaly labels.")

    return data


def main():
    data = load_data()
    X = data[SENSOR_COLUMNS]
    y = data[TARGET].astype(int)
    groups = data["source_file"]

    if groups.nunique() < 3:
        raise SystemExit("Need at least three labelled files for this evaluation.")

    # Outer split: reserve entire files for final testing.
    outer = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(outer.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    g_train = groups.iloc[train_idx]

    # Inner split: separate fit and validation files.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=21)
    fit_idx, val_idx = next(
        inner.split(X_train, y_train, groups=g_train)
    )

    X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
    y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]

    if y_fit.nunique() < 2 or y_val.nunique() < 2:
        raise SystemExit(
            "The fit or validation split has only one class. "
            "We need to adjust the file-level split."
        )

    # Fit a fresh model only on the fit files.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(X_fit, y_fit)

    probabilities = model.predict_proba(X_val)
    anomaly_index = list(model.classes_).index(1)
    scores = probabilities[:, anomaly_index]

    print("=" * 60)
    print("PHANTOM DETECTION AI - VALIDATION THRESHOLDS")
    print("=" * 60)
    print(f"Total labelled rows: {len(data)}")
    print(f"Fit rows: {len(X_fit)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Reserved test rows: {len(test_idx)}")
    print("\nThreshold | Precision | Recall | F1")
    print("-" * 45)

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        predictions = (scores >= threshold).astype(int)
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)

        print(
            f"{threshold:9.2f} | {precision:9.3f} | "
            f"{recall:6.3f} | {f1:.3f}"
        )

    print("\nValidation only: do not treat these results as final test scores.")
    print("The existing saved model has not been changed.")


if __name__ == "__main__":
    main()
