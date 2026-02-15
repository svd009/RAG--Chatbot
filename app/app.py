import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 RAG Chatbot")
st.sidebar.header("📄 Document Management")

# Test if OpenAI key works
api_key = os.getenv("OPENAI_API_KEY")
st.sidebar.write("**API Key loaded:**", "✅ Yes" if api_key else "❌ No")

# PDF Upload (this will definitely show)
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", accept_multiple_files=True, type="pdf"
)

if uploaded_files:
    st.sidebar.success(f"✅ {len(uploaded_files)} file(s) uploaded!")
    for file in uploaded_files:
        st.sidebar.write(f"- {file.name} ({file.size} bytes)")
else:
    st.sidebar.info("👆 Upload PDFs here")

st.info("🎉 PDF uploader working! Fix imports next.")
