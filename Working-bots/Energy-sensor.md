This will be a **Streamlit app** that:

1. Accepts **equipment sensor logs** (CSV/JSON)
2. Uses **simple anomaly detection / predictive logic**
3. Passes anomalies + summaries to an **LLM** to generate a **maintenance schedule**
4. Displays results in a **human-readable table**

For hackathon purposes, we’ll focus on **simplicity + clarity**.

---

### ⚡ Full Prototype Code

```python
import streamlit as st
import pandas as pd
import numpy as np
import httpx
import os
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Energy AI Agent — Predictive Maintenance", layout="wide")
st.title("⚡ Energy Sector AI Agent — Predictive Maintenance Scheduling")

client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# ---------------- LLM Client ----------------
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        base_url="url",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="****",
        http_client=client
    )

llm = get_llm()

# ---------------- File Upload ----------------
st.sidebar.header("Inputs")
sensor_file = st.sidebar.file_uploader("Upload Equipment Sensor Logs (CSV)", type="csv")
mode = st.sidebar.radio("Mode", ["Analyze & Predict", "Chat Mode"])

if not sensor_file:
    st.info("Upload a CSV of sensor logs to start.")
    st.stop()

# ---------------- Load Sensor Data ----------------
df = pd.read_csv(sensor_file)
st.subheader("📊 Sensor Data Preview")
st.dataframe(df.head(10))

# ---------------- Predictive Logic ----------------
st.subheader("🔍 Predictive Maintenance Analysis")

# Assume the CSV has columns: 'equipment_id', 'timestamp', 'temperature', 'vibration', 'pressure', etc.
# For prototype: detect anomalies if sensor exceeds threshold (simple rule)
sensor_columns = [c for c in df.columns if c not in ['equipment_id', 'timestamp']]

def detect_anomalies(df, sensor_columns):
    df_anomaly = df.copy()
    anomaly_flags = []
    thresholds = {col: df[col].mean() + 2*df[col].std() for col in sensor_columns}  # simple statistical threshold
    for _, row in df.iterrows():
        flag = any(row[col] > thresholds[col] for col in sensor_columns)
        anomaly_flags.append(flag)
    df_anomaly['anomaly'] = anomaly_flags
    return df_anomaly, thresholds

df_anomaly, thresholds = detect_anomalies(df, sensor_columns)

# Show anomalies summary
anomaly_summary = df_anomaly[df_anomaly['anomaly'] == True].groupby('equipment_id').size().reset_index(name='anomaly_count')
st.write("### ⚠️ Anomaly Summary by Equipment")
st.dataframe(anomaly_summary)

# ---------------- LLM Prompt to Generate Maintenance Schedule ----------------
st.subheader("📋 Generating Maintenance Schedule (LLM)")

# Prepare context: summarize anomalies for LLM
context_text = ""
for _, row in anomaly_summary.iterrows():
    context_text += f"Equipment {row['equipment_id']} has {row['anomaly_count']} recent anomalies.\n"

system_prompt = f"""
You are an expert maintenance engineer AI. Given the following anomaly summary for energy equipment, generate a prioritized predictive maintenance schedule.
For each equipment with anomalies, provide:
- Equipment ID
- Predicted Risk Level (High, Medium, Low)
- Recommended Maintenance Action
- Suggested Maintenance Date (within next week)
Present the output in a **markdown table**. Keep answers concise.

Anomaly Summary:
{context_text}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Generate the predictive maintenance schedule.")
])

docs_chain = create_stuff_documents_chain(llm, prompt)
result = docs_chain.invoke({"input": "Generate schedule"})

st.subheader("🗓 Predicted Maintenance Schedule")
st.markdown(result["output_text"])

# ---------------- Optional Chat Mode ----------------
if mode == "Chat Mode":
    st.subheader("💬 Ask questions about the maintenance")
    query = st.chat_input("Ask something about equipment anomalies or maintenance")
    if query:
        chat_prompt = f"Answer based on the following anomalies summary:\n{context_text}\nQuestion: {query}"
        chat_response = llm(chat_prompt)
        st.chat_message("user").markdown(query)
        st.chat_message("assistant").markdown(chat_response)
```

---

### ✅ Features in This Prototype

| Feature                        | Description                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------ |
| 📈 **Sensor Data Analysis**    | Loads CSV logs and computes simple anomaly flags                               |
| ⚠️ **Anomaly Detection**       | Flags equipment where sensor readings exceed statistical thresholds            |
| 🧠 **LLM Schedule Generation** | Uses GPT/DeepSeek to produce a **human-readable predictive maintenance table** |
| 📊 **Interactive Dashboard**   | Shows sensor data preview, anomalies, and LLM-generated schedule               |
| 💬 **Chat Mode**               | Ask questions like “Which equipment has the highest risk this week?”           |

---

### 🧩 Notes for Hackathon

1. **Simplified for demo** — thresholds are statistical (mean + 2σ).
2. LLM converts numeric anomalies into **priority risk and actionable schedule**.
3. Easy to extend:

   * More sensors, multiple time-series analysis
   * Real ML predictive models (e.g., Random Forest, Prophet, LSTM)
   * Integration with ERP/Maintenance systems

---

If you want, I can **also create a sample CSV format and demo dataset** for this Energy AI Agent — perfect for showing a working prototype in your hackathon.

Do you want me to do that next?
