import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.ingest import load_documents_from_pdf, split_documents
from scripts.retriever import create_vector_store, load_vector_store
from scripts.rag_pipeline import create_qa_chain, ask_question
import os

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄🔍 RAG Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
persist_path = "models/faiss_index"

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Loading and chunking document...")
    docs = load_documents_from_pdf("temp.pdf")
    chunks = split_documents(docs)

    st.success(f"Loaded {len(chunks)} chunks.")

    st.info("Creating vector store...")
    vectorstore = create_vector_store(chunks, persist_path)
    st.success("Vector store created and saved.")

    qa_chain = create_qa_chain(vectorstore)

    question = st.text_input("Ask a question based on the document:")
    if question:
        with st.spinner("Generating answer..."):
            answer = ask_question(qa_chain, question)
            st.write("**Answer:**", answer)

elif os.path.exists(persist_path):
    vectorstore = load_vector_store(persist_path)
    qa_chain = create_qa_chain(vectorstore)
    question = st.text_input("Ask a question based on the existing vectorstore:")
    if question:
        with st.spinner("Generating answer..."):
            answer = ask_question(qa_chain, question)
            st.write("**Answer:**", answer)
else:
    st.warning("Please upload a PDF to get started.")

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
