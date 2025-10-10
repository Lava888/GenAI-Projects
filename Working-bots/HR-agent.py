import streamlit as st
import httpx
import os
import json
from pypdf import PdfReader
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------- UI Setup ----------------
st.set_page_config(page_title="HR AI Agent — Resume Screener", layout="wide")
st.title("HR AI Agent — Resume Screening & Candidate Validation")

client = httpx.Client(verify=False)
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# --------------- Credentials / Clients ---------------
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        base_url="url",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key="****",       # <-- replace with your key or use env var
        http_client=client
    )

def get_llm():
    return ChatOpenAI(
        base_url="url",
        model="azure_ai/genailab-maas-DeepSeek-V3-0324",
        api_key="****",       # <-- replace with your key or use env var
        http_client=client
    )

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Inputs")
pdf_resume = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")
job_json = st.sidebar.file_uploader("Upload Job Description (JSON) — optional", type="json")
job_text = st.sidebar.text_area("Or paste Job Description / Required Skills (optional)", height=150)
mode = st.sidebar.radio("Mode", ["Chat Mode", "Extract & Validate"])

# parse job JSON if provided
def parse_job_json(job_file):
    if not job_file:
        return None
    try:
        return json.load(job_file)
    except Exception:
        return None

job_data = parse_job_json(job_json)
# if job_text is present, prefer that for requirements
if job_text and job_text.strip():
    # simple structure: put the free text under "job_text"
    job_data = job_data or {}
    job_data["job_text"] = job_text.strip()

# ---------------- Core logic ----------------
if not pdf_resume:
    st.info("Upload a resume PDF in the sidebar to start. Optionally upload a job JSON or paste job text.")
    st.stop()

@st.cache_resource
def get_retriever(file):
    reader = PdfReader(file)
    docs = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text()
        if text and text.strip():
            docs.append(Document(page_content=text, metadata={"page": i+1}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embedding=get_embeddings())
    return vector_store.as_retriever()

retriever = get_retriever(pdf_resume)
llm = get_llm()

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase user's question to be specific and context-aware about the resume content."),
    ("human", "{input}")
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

# HR-specific system prompt for extraction
system_prompt = """
You are an HR assistant AI. Given resume content, extract structured candidate information as JSON.
Return only valid JSON when extracting. Include these fields when available:
["Candidate Name", "Email", "Phone", "Years of Experience", "Primary Skills", "Education", "Certifications", "Last Employer"].

When asked to evaluate against a job description or required skills:
 - Identify matching skills and missing skills.
 - Produce an "Overall Fit Score" (0-100) based on matched vs required skills.
 - Keep the JSON concise and factual; use only information present in the resume or job text.

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, docs_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Chat Mode ----------------
if mode == "Chat Mode":
    st.subheader("Chat with the Resume")
    query = st.chat_input("Ask something about the resume (e.g., 'What are the candidate's top skills?')")
    if query:
        response = rag_chain.invoke({"input": query, "chat_history": st.session_state.messages})
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": response["answer"]})

    # show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --------------- Extract & Validate Mode ---------------
else:
    st.subheader("🧾 Extract & Validate — Resume Screening")
    # 1) Extract candidate fields as JSON
    extract_prompt = "Extract candidate fields as JSON using required field labels. Return JSON only."
    response = rag_chain.invoke({"input": extract_prompt, "chat_history": st.session_state.messages})

    st.write("### Extracted Candidate Data")
    try:
        candidate_data = json.loads(response["answer"])
    except Exception:
        # fallback: include raw LLM response if JSON failed
        candidate_data = {"raw_extraction": response["answer"]}
    st.json(candidate_data)

    # 2) Validation / Matching logic
    st.write("### 🔍 Job Match Analysis")
    # if job criteria present, try to parse required skills from job_data
    required_skills = set()
    job_title = ""
    if job_data:
        # if structured JSON contains a specific field, prefer it; else try to parse free text
        if isinstance(job_data, dict):
            job_title = job_data.get("Job Title", "") or job_data.get("job_title", "")
            # try "Required Skills" field if present (comma-separated or list)
            req = job_data.get("Required Skills") or job_data.get("required_skills") or job_data.get("skills")
            if isinstance(req, list):
                required_skills.update([s.strip().lower() for s in req if s])
            elif isinstance(req, str):
                # assume comma or newline separated
                parts = [p.strip() for p in req.replace("\n", ",").split(",") if p.strip()]
                required_skills.update([p.lower() for p in parts])
            # fallback: if job_text present, we'll parse it below
            if not required_skills and job_data.get("job_text"):
                job_text_for_parse = job_data.get("job_text")
            else:
                job_text_for_parse = job_data.get("job_text") if isinstance(job_data, dict) else ""
        else:
            job_text_for_parse = str(job_data)
    else:
        job_text_for_parse = job_text or ""

    # naive parsing of required skills from free text job description (split on commas/newlines)
    if job_text_for_parse and not required_skills:
        parts = [p.strip() for p in job_text_for_parse.replace("\n", ",").split(",") if p.strip()]
        # heuristics: assume skills are short tokens (<=4 words)
        for p in parts:
            if len(p.split()) <= 5:
                required_skills.add(p.lower())

    # extract candidate skills (try to read Primary Skills field)
    candidate_skills_raw = candidate_data.get("Primary Skills") or candidate_data.get("Skills") or candidate_data.get("primary_skills") or ""
    candidate_skills = set()
    if isinstance(candidate_skills_raw, list):
        candidate_skills = set([s.strip().lower() for s in candidate_skills_raw if s])
    elif isinstance(candidate_skills_raw, str):
        parts = [p.strip() for p in candidate_skills_raw.replace("\n", ",").split(",") if p.strip()]
        candidate_skills = set([p.lower() for p in parts])

    # compute match
    matched = sorted(list(candidate_skills.intersection(required_skills)))
    missing = sorted(list(required_skills.difference(candidate_skills)))
    fit_score = 0
    if required_skills:
        fit_score = int((len(matched) / len(required_skills)) * 100)

    # friendly display
    if job_title:
        st.markdown(f"**Job:** {job_title}")
    st.markdown(f"**Required skills (detected):** {', '.join(sorted(required_skills)) if required_skills else 'N/A'}")
    st.markdown(f"**Candidate skills (detected):** {', '.join(sorted(candidate_skills)) if candidate_skills else 'N/A'}")
    st.markdown(f"**Matched:** {', '.join(matched) if matched else 'None'}")
    st.markdown(f"**Missing:** {', '.join(missing) if missing else 'None'}")
    st.markdown(f"**Fit Score:** **{fit_score}%**")

    # summary card
    st.write("### Candidate Summary")
    summary_md = (
        f"- **Name:** {candidate_data.get('Candidate Name', 'N/A')}\n"
        f"- **Email:** {candidate_data.get('Email', 'N/A')}\n"
        f"- **Years Experience:** {candidate_data.get('Years of Experience', 'N/A')}\n"
        f"- **Primary Skills:** {candidate_data.get('Primary Skills', 'N/A')}\n"
        f"- **Overall Fit Score:** {fit_score}%\n"
    )
    st.markdown(summary_md)

    # Append to chat history for later conversation context if you want
    st.session_state.messages.append({"role": "system", "content": "Last extraction & validation completed."})
    st.session_state.messages.append({"role": "ai", "content": f"Fit Score: {fit_score}%, Matched Skills: {matched}"})

# Show chat history (if any)
if st.session_state.messages:
    st.write("------")
    st.write("### Chat / Activity History")
    for msg in st.session_state.messages[-20:]:
        who = "User" if msg["role"] == "user" else ("Assistant" if msg["role"] == "ai" else msg["role"])
        st.markdown(f"**{who}:** {msg['content']}")
