import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import httpx
from openai import OpenAI
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile
import streamlit as st
import pandas as pd

def get_eli_chat_model(temperature: float = 0.0, model_name: str = "qwen2.5-7b"):
    # Create an instance of the OpenAI client
    client = OpenAI(
        api_key="xxxx",
        base_url="url",
        http_client=httpx.Client(verify=False),
    )
    # Create an instance of ChatOpenAI
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key="xxxx",
        base_url="url",
    )
    # Now we plug the OpenAI client into our langchain-openai interface
    llm.client = client.chat.completions
    return llm

chat = get_eli_chat_model()

# Step 1: Upload PDF
st.sidebar.title("Upload Document")
st.sidebar.markdown("### \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])


st.title("PDF Chatbot")
st.write("Ask questions based on your uploaded PDF.")

# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_pdf_path = tmp_file.name

#     # Step 2: Load PDF
#     loader = PyPDFLoader(tmp_pdf_path)
#     documents = loader.load()
emmbeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(persist_directory="./munindb", embedding_function=emmbeddings)
retriever = vectorstore.as_retriever()


    # Step 4: RAG Prompt Setup
system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If User is greeting then greet the user. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )

prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

qa_chain = create_stuff_documents_chain(chat, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Step 5: Ask user input
query = st.chat_input("Ask something")
if query:
    st.write("User: ", query)
    response = rag_chain.invoke({"input": query})
    st.write(response['answer'])
