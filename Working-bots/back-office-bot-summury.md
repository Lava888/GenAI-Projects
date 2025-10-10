pip install streamlit httpx pypdf langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pandas xlsxwriter

**GenAI Hackathon use case: “Office AI Agent for Automated Invoice Processing & Validation.”**

---

# 📄 **Office AI Agent – Automated Invoice Processing & Validation**

## 🧠 **Overview**

Manual invoice validation is time-consuming, error-prone, and repetitive.
Our **Office AI Agent** automates this workflow using **Generative AI + Document Intelligence**.

The system:

* Extracts key fields from invoice PDFs using an LLM
* Validates them against uploaded Purchase Orders
* Flags mismatches automatically
* Generates structured, exportable reports (JSON / Excel)

This project demonstrates a **real-world GenAI automation use case** for office document workflows.

---

## 🚀 **Key Features**

| Feature                      | Description                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| 📚 **Context-Aware Chatbot** | Users can chat naturally and ask questions about invoice content                 |
| 🧾 **Data Extraction**       | LLM extracts key invoice details (vendor, invoice no, PO no, amount, etc.)       |
| 🔍 **Automated Validation**  | Compares extracted invoice fields with uploaded Purchase Orders                  |
| ⚠️ **Discrepancy Detection** | Highlights mismatches and missing fields instantly                               |
| 📊 **Export Reports**        | Generates downloadable JSON and Excel reports with structured validation results |
| 🕹️ **Dual Mode**            | Switch between *Chat Mode* and *Extract & Validate Mode*                         |
| 💡 **LLM + Vector Search**   | Uses embeddings and FAISS for intelligent retrieval and contextual reasoning     |

---

## 🧩 **Architecture**

```
                ┌──────────────────────┐
                │  User Uploads Files  │
                │ (Invoice PDF, PO CSV)│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Document Processing │
                │ (PDF → Text Splitter)│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Vector Store (FAISS)│
                │ Embeddings via OpenAI│
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   LLM Reasoning (RAG)│
                │   Extract / Validate │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Streamlit Frontend   │
                │  - Chat Interface    │
                │  - JSON/Excel Export │
                └──────────────────────┘
```

---

## ⚙️ **Tech Stack**

| Layer                   | Technology                                 |
| ----------------------- | ------------------------------------------ |
| **Frontend/UI**         | Streamlit                                  |
| **Backend Logic**       | Python                                     |
| **LLM Integration**     | Azure OpenAI (via `langchain-openai`)      |
| **Document Processing** | `pypdf`, `langchain-text-splitters`        |
| **Vector Store**        | FAISS (via `langchain-community`)          |
| **Validation Logic**    | Python (JSON/CSV field comparison)         |
| **Export Formats**      | JSON & Excel (via `pandas` + `xlsxwriter`) |

---

## 📦 **Setup Instructions**

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd invoice-ai-agent
```

### 2. Install Dependencies

```bash
pip install streamlit httpx pypdf langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pandas xlsxwriter python-dotenv tiktoken
```

### 3. Configure API Key

Set your **Azure/OpenAI API key** in the code or use an `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧠 **How It Works**

1. **File Upload**

   * Upload an **invoice PDF** and optionally a **Purchase Order (JSON/CSV)**.

2. **Extraction & Understanding**

   * The LLM extracts key information from the PDF text.

3. **Validation**

   * Extracted invoice data is compared with PO data (if uploaded).
   * Discrepancies like missing or mismatched values are flagged.

4. **Report Generation**

   * Results are presented in an interactive dashboard and can be exported as JSON or Excel.

5. **Conversational Mode**

   * Users can chat with the system in natural language for quick insights about the invoice.

---

## 📊 **Example Output**

**Extracted Invoice JSON:**

```json
{
  "Vendor Name": "Tech Supplies Ltd.",
  "Invoice Number": "INV-2025-1042",
  "Invoice Date": "2025-10-01",
  "PO Number": "PO-8932",
  "Tax Amount": "₹2,500",
  "Total Amount": "₹45,000"
}
```

**Validation Result:**

```
✅ All fields validated successfully – Invoice matches PO!
```

Or if discrepancies:

```
❌ Mismatch in Total Amount → Invoice: ₹45,000 | PO: ₹44,000
⚠️ Missing field in invoice: PO Number
```

**Export Options:**

* `invoice_validation_20251010_154233.json`
* `invoice_validation_20251010_154233.xlsx`

---

## 🧩 **Impact & Use Cases**

| Industry         | Application                            |
| ---------------- | -------------------------------------- |
| 🏢 Finance       | Invoice validation, billing automation |
| 🧾 Procurement   | Purchase Order cross-verification      |
| 🏥 Healthcare    | Claims and billing document checks     |
| ⚙️ Manufacturing | Supplier invoice & PO reconciliation   |
| 🧑‍💼 HR/Admin   | Expense reimbursement validation       |

**Impact:**

* ⚡ Reduces manual validation time by ~80%
* 🧠 Minimizes human errors in invoice processing
* 💰 Enables faster, more accurate payments

---

## 🏁 **Future Enhancements**

* 🔍 OCR integration for scanned invoices (Tesseract / Azure Form Recognizer)
* 🧠 Multi-document validation (Batch processing)
* 💬 Slack or Teams agent integration
* 📈 Analytics dashboard for total spend, vendor trends, etc.
* 🪄 Autonomous approval workflow integration with ERP APIs

---

