import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
from streamlit_autorefresh import st_autorefresh

from langchain_openai import ChatOpenAI
import httpx

# Set up the LLM client
client = httpx.Client(verify=False)

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",  # or your LLM endpoint
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="YOUR_API_KEY_HERE",  # Replace with your key
    http_client=client
)

def generate_fault_explanation(temp, vib, pres, severity):
    prompt = f"""
    The system detected a potential equipment fault.
    Sensor Readings:
    - Temperature: {temp:.2f} °C
    - Vibration: {vib:.2f} mm/s
    - Pressure: {pres:.2f} psi
    - Severity: {severity}

    Please explain the likely cause of the fault and recommend an action in simple terms.
    """
    response = llm.invoke(prompt)
    return response.content



    
# Load your trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="IoT Fault Detection", layout="wide")
st.title("🔧 IoT AI Agent: Real-Time Predictive Fault Detection")
st.markdown("Simulated real-time monitoring of industrial equipment using AI-powered fault prediction.")

# Initialize history DataFrame in session state
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Temperature', 'Vibration', 'Pressure', 'Fault', 'Severity'])

# Auto-refresh every 2 seconds (2000 ms), up to 50 times
count = st_autorefresh(interval=2000, limit=50, key="refresh")

# Function to simulate sensor data
def get_simulated_sensor_data():
    temperature = np.random.normal(70, 10)   # °C
    vibration = np.random.normal(12, 6)      # mm/s
    pressure = np.random.normal(30, 4)       # psi
    return temperature, vibration, pressure

# Function to determine severity
def get_severity(temp, vib, pres):
    if vib > 25 or temp > 95 or pres > 40:
        return "Critical"
    elif vib > 20 or temp > 90 or pres > 37:
        return "High"
    elif vib > 15 or temp > 80 or pres > 35:
        return "Moderate"
    else:
        return "Low"

# Function to get alert message
def get_alert_message(severity):
    messages = {
        "Critical": "🔴 Immediate shutdown and inspection required!",
        "High": "🟠 Schedule maintenance ASAP to prevent failure.",
        "Moderate": "🟡 Monitor closely and prepare for maintenance.",
        "Low": "🟢 System is stable, no immediate action needed."
    }
    return messages[severity]

# Generate new simulated sensor reading
temperature, vibration, pressure = get_simulated_sensor_data()
timestamp = datetime.datetime.now().strftime("%H:%M:%S")

# Predict fault using model
fault = model.predict([[temperature, vibration, pressure]])[0]
severity = get_severity(temperature, vibration, pressure)
alert_msg = get_alert_message(severity)

# Append new data to history
new_data = pd.DataFrame([{
    'Time': timestamp,
    'Temperature': temperature,
    'Vibration': vibration,
    'Pressure': pressure,
    'Fault': fault,
    'Severity': severity
}])
st.session_state.history = pd.concat([st.session_state.history, new_data], ignore_index=True)

# Display current sensor values
col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Temperature (°C)", f"{temperature:.2f}")
col2.metric("💢 Vibration (mm/s)", f"{vibration:.2f}")
col3.metric("⚙️ Pressure (psi)", f"{pressure:.2f}")

# Show fault detection results
st.markdown("### 🛑 Fault Detection Result")
if fault == 1:
    st.error(f"🚨 Fault Detected — Severity: **{severity}**")
    st.warning(alert_msg)
else:
    st.success("✅ System Normal - No fault detected.")

# Show sensor trends
st.markdown("### 📈 Sensor Trends (Last 50 Readings)")
chart_data = st.session_state.history[['Temperature', 'Vibration', 'Pressure']]
chart_data.index = st.session_state.history['Time']
st.line_chart(chart_data.tail(50))

print(f"Current sensor values: Temp={temperature}, Vib={vibration}, Pressure={pressure}")

def suggest_fix_with_genai(temp, vib, pres, severity):
    prompt = f"""
    An industrial equipment system reported a fault.

    Sensor Readings:
    - Temperature: {temp:.1f} °C
    - Vibration: {vib:.1f} mm/s
    - Pressure: {pres:.1f} psi
    - Fault severity: {severity}

    Based on this data:
    1. Suggest the most likely root cause of the fault.
    2. Recommend an action the maintenance team should take.
    3. Keep the answer concise and clear.
    """

    response = llm.invoke(prompt)
    return response.content.strip()
st.warning(alert_msg)

# Add this:
with st.spinner("🧠 Generating maintenance suggestion..."):
    solution = suggest_fix_with_genai(temperature, vibration, pressure, severity)
st.markdown("### 🛠️ Suggested Resolution")
st.info(solution)
