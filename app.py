import streamlit as st
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot PROTOTYPE")
st.markdown("✅ Working! Upload PDF → Extract text → Chat simulation")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    
    # Simple PDF text extraction (no ML)
    try:
        import pypdf
        with open(tmp_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        st.success("✅ PDF text extracted!")
        st.text_area("Document Preview:", text[:2000], height=300)
        
        # Mock RAG response
        query = st.chat_input("Ask about this document...")
        if query:
            st.chat_message("user").write(query)
            st.chat_message("assistant").markdown(f"""
            **Answer:** Based on the document content, here's what I found:
            
            🔍 **Key points detected:** {len(text.split())} words, {text[:100]}...
            
            💡 **Answer to "{query}":** 
            The document discusses [topic]. Relevant sections include...
            """)
            
    except Exception as e:
        st.error(f"PDF error: {e}")
    finally:
        os.unlink(tmp_path)
else:
    st.info("👆 Upload a PDF in the sidebar to start!")
    st.balloons()
