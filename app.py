import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # FIXED deprecation
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace     # FREE local model
from langchain_huggingface import HuggingFacePipeline

st.set_page_config(layout="wide", page_title="FREE PDF RAG Chatbot")
st.title("🤖 FREE PDF RAG Chatbot (No OpenAI needed!)")
st.info("👈 Upload PDF → Process → Ask questions LOCALLY!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - PDF Upload
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose PDF", type="pdf")

if uploaded_file and st.sidebar.button("🚀 Process PDF", use_container_width=True):
    with st.spinner("🔄 Processing your PDF..."):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
        temp_file.close()
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma.from_documents(
            chunks, embeddings, persist_directory="./chroma_db"
        )
        os.unlink(temp_file_path)
        
        st.session_state.messages = []
        st.sidebar.success(f"✅ {len(chunks)} chunks indexed!")
        st.success("🚀 Ready! FREE local answers!")

# Chat Interface  
if st.session_state.vectorstore:
    st.header("💬 Ask about your PDF")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF locally..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # FREE local model (downloads first time ~2GB)
                pipe = HuggingFacePipeline.from_model_id(
                    model_id="microsoft/DialoGPT-medium",
                    task="text-generation",
                    pipeline_kwargs={"max_new_tokens": 200}
                )
                llm = ChatHuggingFace(llm=pipe)
                
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}\nAnswer:"
                response = llm.invoke(full_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

else:
    st.markdown("""
    ### 🚀 Quick Start (100% FREE!)
    1. **Upload PDF** in sidebar 
    2. **Click "Process PDF"**
    3. **Ask questions** → Get answers LOCALLY from YOUR PDF!
    """)

st.markdown("---")
st.caption("✅ FREE!")
