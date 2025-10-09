import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("data/sensor_data.csv")

# Remove non-feature columns
X = df.drop(columns=["timestamp", "label"], errors='ignore')

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "models/isolation_forest_model.pkl")
print("✅ Model trained and saved to models/isolation_forest_model.pkl")
