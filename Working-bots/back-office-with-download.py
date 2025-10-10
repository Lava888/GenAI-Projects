import streamlit as st
import httpx
import os
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ------------------ UI Setup ------------------
st.set_page_config(page_title="Invoice AI Agent", layout="wide")
st.title("📄 Office AI Agent – Smart Invoice Processing & Validation")

client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# ------------------ Helper Functions ------------------
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        base_url="url",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="****",
        http_client=client
    )

def get_eli_chat_model():
    return ChatOpenAI(
        base_url="url",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="****",
        http_client=client
    )

def parse_po_file(po_file):
    if not po_file:
        return None
    if po_file.name.endswith(".json"):
        return json.load(po_file)
    elif po_file.name.endswith(".csv"):
        df = pd.read_csv(po_file)
        return df.to_dict(orient="records")[0] if not df.empty else None
    return None

# ------------------ Sidebar ------------------
pdf_file = st.sidebar.file_uploader("Upload Invoice PDF", type="pdf")
po_file = st.sidebar.file_uploader("Upload Purchase Order (JSON/CSV optional)", type=["json", "csv"])
mode = st.sidebar.radio("Select Mode", ["Chat Mode", "Extract & Validate"])

purchase_order_data = parse_po_file(po_file)

# ------------------ Core Logic ------------------
if pdf_file:
    @st.cache_resource
    def get_retriever(file):
        reader = PdfReader(file)
        docs = [Document(page_content=p.extract_text(), metadata={"page": i})
                for i, p in enumerate(reader.pages) if p.extract_text()]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embedding=get_embeddings())
        return vector_store.as_retriever()

    retriever = get_retriever(pdf_file)
    chat = get_eli_chat_model()

    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase user's question to be specific and context-aware."),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(chat, retriever, rephrase_prompt)

    system_prompt = """
    You are an AI assistant that extracts structured invoice data.
    Extract and output data as a JSON object with the following fields if possible:
    ["Vendor Name", "Invoice Number", "Invoice Date", "PO Number", "Tax Amount", "Total Amount"].
    Be concise and ensure correct field labels. Include only factual data from the invoice text.
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    docs_chain = create_stuff_documents_chain(chat, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, docs_chain)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -------------- Chat Mode --------------
    if mode == "Chat Mode":
        query = st.chat_input("Ask something about the invoice")
        if query:
            response = rag_chain.invoke({"input": query, "chat_history": st.session_state.messages})
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "ai", "content": response["answer"]})

    # -------------- Extraction & Validation --------------
    else:
        st.subheader("Automated Extraction & Validation")

        extract_prompt = "Extract key invoice fields as JSON with correct labels."
        response = rag_chain.invoke({"input": extract_prompt, "chat_history": st.session_state.messages})

        # Step 1: Extraction
        st.write("### Extracted Data")
        try:
            invoice_data = json.loads(response["answer"])
        except:
            invoice_data = {"raw_text": response["answer"]}
        st.json(invoice_data)

        # Step 2: Validation
        discrepancies = []
        if purchase_order_data:
            st.write("### Validation Report")
            for key, po_value in purchase_order_data.items():
                inv_value = invoice_data.get(key)
                if inv_value is None:
                    discrepancies.append(f"Missing field in invoice: **{key}**")
                elif str(inv_value).strip() != str(po_value).strip():
                    discrepancies.append(f"Mismatch in **{key}** → Invoice: `{inv_value}` | PO: `{po_value}`")

            if discrepancies:
                st.error("Discrepancies found:")
                for d in discrepancies:
                    st.markdown(f"- {d}")
            else:
                st.success("All fields validated successfully – Invoice matches PO!")
        else:
            st.info("No purchase order uploaded — validation skipped.")

        # Step 3: Summary
        st.write("### 📋 Validation Summary")
        summary = (
            f"Vendor: **{invoice_data.get('Vendor Name', 'N/A')}**\n\n"
            f"Invoice #: **{invoice_data.get('Invoice Number', 'N/A')}** | "
            f"PO #: **{invoice_data.get('PO Number', 'N/A')}**\n\n"
            f"Total Amount: **{invoice_data.get('Total Amount', 'N/A')}**"
        )
        st.markdown(summary)
        st.success("Validation Complete")

        # Step 4: Export (Download Buttons)
        st.divider()
        st.subheader("Export Results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON Export
        json_output = {
            "invoice_data": invoice_data,
            "purchase_order_data": purchase_order_data,
            "discrepancies": discrepancies,
            "summary": summary,
            "timestamp": timestamp
        }
        json_bytes = json.dumps(json_output, indent=4).encode("utf-8")
        st.download_button(
            label="Download Validation Report (JSON)",
            data=json_bytes,
            file_name=f"invoice_validation_{timestamp}.json",
            mime="application/json"
        )

        # Excel Export
        try:
            df_invoice = pd.DataFrame([invoice_data])
            df_po = pd.DataFrame([purchase_order_data]) if purchase_order_data else pd.DataFrame()
            df_disc = pd.DataFrame({"Discrepancies": discrepancies})
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_invoice.to_excel(writer, index=False, sheet_name="Invoice Data")
                df_po.to_excel(writer, index=False, sheet_name="Purchase Order")
                df_disc.to_excel(writer, index=False, sheet_name="Validation Report")
            excel_data = output.getvalue()

            st.download_button(
                label="Download Validation Report (Excel)",
                data=excel_data,
                file_name=f"invoice_validation_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.warning(f"Excel export skipped: {e}")

    # -------------- Chat History UI --------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
