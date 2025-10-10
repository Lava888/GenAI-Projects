import streamlit as st
import httpx
import os
import json
import pandas as pd
from pypdf import PdfReader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Invoice AI Agent", layout="wide")
st.title("Office AI Agent – Smart Invoice Processing & Validation")

client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# --- Embeddings
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        base_url="url",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="****",
        http_client=client
    )

# --- LLM
def get_eli_chat_model():
    return ChatOpenAI(
        base_url="url",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="****",
        http_client=client
    )

# --- File Uploads
pdf_file = st.sidebar.file_uploader("Upload Invoice PDF", type="pdf")
po_file = st.sidebar.file_uploader("Upload Purchase Order (JSON/CSV optional)", type=["json", "csv"])
mode = st.sidebar.radio("Select Mode", ["Chat Mode", "Extract & Validate"])

# --- Parse PO file (if available)
def parse_po_file(po_file):
    if not po_file:
        return None
    if po_file.name.endswith(".json"):
        return json.load(po_file)
    elif po_file.name.endswith(".csv"):
        df = pd.read_csv(po_file)
        return df.to_dict(orient="records")[0] if not df.empty else None
    return None

purchase_order_data = parse_po_file(po_file)

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

    # --- Prompts
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

    # --- Chat Mode
    if mode == "Chat Mode":
        query = st.chat_input("Ask something about the invoice")
        if query:
            response = rag_chain.invoke({"input": query, "chat_history": st.session_state.messages})
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "ai", "content": response["answer"]})
    else:
        st.subheader("Automated Extraction & Validation")

        # Step 1: Extract fields via LLM
        extract_prompt = "Extract key invoice fields as JSON with correct labels."
        response = rag_chain.invoke({"input": extract_prompt, "chat_history": st.session_state.messages})

        st.write("### Extracted Data")
        try:
            invoice_data = json.loads(response["answer"])
        except:
            invoice_data = {"raw_text": response["answer"]}
        st.json(invoice_data)

        # Step 2: Validate against PO (if provided)
        if purchase_order_data:
            st.write("### Validation Report")
            discrepancies = []
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

        # Step 3: Summary output
        st.write("### Validation Summary")
        summary = (
            f"Vendor: **{invoice_data.get('Vendor Name', 'N/A')}**\n\n"
            f"Invoice #: **{invoice_data.get('Invoice Number', 'N/A')}** | "
            f"PO #: **{invoice_data.get('PO Number', 'N/A')}**\n\n"
            f"Total Amount: **{invoice_data.get('Total Amount', 'N/A')}**"
        )
        st.markdown(summary)
        st.success("Validation Complete")

    # --- Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
