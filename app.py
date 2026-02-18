import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline

st.set_page_config(layout="wide", page_title="PDF RAG Chatbot v2.0")
st.title("🤖 PDF RAG Chatbot")
st.info("Upload PDF → Process → Ask focused questions!")

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
        st.success("🚀 Ready for questions!")
        st.balloons()

# Chat Interface - FIXED VERSION
if st.session_state.vectorstore:
    st.header("💬 Ask about your PDF")
    
    # Show chat history
    for message in st.session_state.messages[-6:]:  # Last 6 messages
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with validation
    if prompt := st.chat_input("Ask a specific question about your PDF..."):
        # Query validation
        prompt = prompt.strip()
        if len(prompt) < 5:
            st.error("Please ask a more specific question (5+ characters)")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            continue
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Progress bar + status updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Retrieving relevant chunks...")
        progress_bar.progress(0.3)
        
        # Retrieve TOP 3 chunks only (FIXES entire PDF dump)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(prompt)
        
        # Limit each chunk to 300 chars + top 3 only
        context = "\n\n".join([doc.page_content[:300] for doc in relevant_docs])
        
        status_text.text("🤖 Generating focused answer...")
        progress_bar.progress(0.8)
        
        # Generate response
        try:
            pipe = HuggingFacePipeline.from_model_id(
                model_id="microsoft/DialoGPT-medium",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 150, "truncation": True}
            )
            llm = ChatHuggingFace(llm=pipe)
            
            full_prompt = f"""Context from PDF (top 3 chunks only):
{context}

Question: {prompt}

Answer briefly using only the context above:"""
            
            response = llm.invoke(full_prompt)
            
            # Truncate response (FIXES long outputs)
            answer = response.content[:800]
            if len(response.content) > 800:
                answer += "... (response truncated)"
                
            status_text.text("✅ Done!")
            progress_bar.progress(1.0)
            
            st.markdown(answer)
            st.caption(f"📊 Based on {len(relevant_docs)} chunks from your PDF")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

else:
    st.markdown("""
    ### 🚀 Quick Start
    1. Upload PDF in sidebar (try your ML/ECE notes)
    2. Click "Process PDF" 
    3. Ask SPECIFIC questions like:
       - "main algorithms discussed?"
       - "Fourier transform applications?"
       - "key findings?"
    """)

st.markdown("---")
st.caption("Production RAG Chatbot- Optimized for accuracy")
