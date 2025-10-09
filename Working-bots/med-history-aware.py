import streamlit as st
import pandas as pd
import os
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Title
st.title("🩺 Medical Assistant")

# HTTP client
client = httpx.Client(verify=False)

# Load LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-kFX3rQYe-dHtSEbfudJyHg",
    http_client=client
)

# Load Embeddings
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-kFX3rQYe-dHtSEbfudJyHg",
    http_client=client
)

# Optional: Tokenizer cache
os.environ["TIKTOKEN_CACHE_DIR"] = "./token"

# Load medical data
df = pd.read_csv("medicals.csv")
documents = [
    Document(
        page_content=row["Symptoms"],
        metadata={
            "disease": row["Disease"],
            "treatment": row["Treatment"]
        }
    )
    for _, row in df.iterrows()
]

# Build FAISS vector store
vector_store = FAISS.from_documents(documents, embedding_model)

# History-aware setup
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the user's question to be more specific and context-aware."),
    ("human", "{input}")
])
retriever = vector_store.as_retriever()
history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

system_prompt = (
    "You are a helpful and knowledgeable medical assistant. "
    "Use the retrieved context below to answer the user's question accurately. "
    "If the user greets you, respond with a friendly greeting. "
    "If the question is unrelated to the context or you don't know the answer, say 'I don't know.' "
    "Only use information from the vector store to ensure accuracy. "
    "Keep your answers clear, concise, and medically appropriate.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

docs_chain = create_stuff_documents_chain(llm, prompt)
history_chain = create_retrieval_chain(history_aware_retriever, docs_chain)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input
query = st.chat_input("Enter your medical condition. I can help you.")

# Response logic
if query:
    with st.chat_message("user"):
        st.write(query)

    # Check relevance before invoking chain
    results = vector_store.similarity_search_with_score(query, k=1)
    if not results or results[0][1] < 0.7:
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": "I'm sorry, I don't recognize these symptoms. Please consult a medical professional."})
    else:
        response = history_chain.invoke({
            "input": query,
            "chat_history": st.session_state.messages
        })
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "ai", "content": response["answer"]})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
