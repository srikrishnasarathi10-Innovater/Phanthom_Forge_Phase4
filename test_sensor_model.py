
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/sensor_anomaly_model.joblib")

print("=" * 55)
print("PHANTOM DETECTION AI — SAVED MODEL TEST")
print("=" * 55)

# Load the trained model
saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
sensor_columns = saved_data["sensor_columns"]

print("Model loaded successfully!")
print("Sensor features:", sensor_columns)

# Use one real sensor record from the processed dataset
data_path = Path("data/processed/valve1/0.csv")
df = pd.read_csv(data_path, sep=";")

# Select one record and prepare the required sensor features
sample = df[sensor_columns].iloc[[0]].apply(
    pd.to_numeric, errors="coerce"
)

# Predict: 0 = normal, 1 = anomaly
prediction = int(model.predict(sample)[0])

print("\nTest record:")
print(sample.to_string(index=False))

print("\nPredicted result:",
      "ANOMALY" if prediction == 1 else "NORMAL")

# Show model confidence scores if supported
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample)[0]
    for label, probability in zip(model.classes_, probabilities):
        name = "Anomaly" if label == 1 else "Normal"
        print(f"{name} probability: {probability:.2%}")

print("\nSaved-model test completed.")
