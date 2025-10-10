import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Equipment list
equipment = ["Pump-01", "Turbine-07", "Transformer-03"]

# Generate timestamps (every 4 hours for 3 days)
timestamps = [datetime(2025, 10, 1, 0, 0) + timedelta(hours=4*i) for i in range(18)]

data = []
np.random.seed(42)

for eq in equipment:
    for ts in timestamps:
        temp = np.random.normal(loc=75 if "Pump" in eq else 90 if "Turbine" in eq else 65, scale=10)
        vib = np.random.normal(loc=0.5 if "Pump" in eq else 1.0 if "Turbine" in eq else 0.3, scale=0.2)
        pres = np.random.normal(loc=1.2 if "Pump" in eq else 1.5 if "Turbine" in eq else 1.0, scale=0.1)
        data.append([eq, ts.strftime("%Y-%m-%d %H:%M"), round(temp,2), round(vib,2), round(pres,2)])

df = pd.DataFrame(data, columns=["equipment_id", "timestamp", "temperature", "vibration", "pressure"])

# Introduce some anomalies to simulate risk
df.loc[5, "temperature"] = 120  # Turbine-07 overheating
df.loc[12, "vibration"] = 2.0   # Turbine-07 high vibration
df.loc[15, "pressure"] = 1.8    # Turbine-07 pressure spike

# Save CSV
df.to_csv("demo_sensor_logs.csv", index=False)
print("Demo CSV generated: demo_sensor_logs.csv")
