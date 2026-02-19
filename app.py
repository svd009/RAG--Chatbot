import streamlit as st
import tempfile
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline

st.set_page_config(layout="wide", page_title="PDF RAG Chatbot v2.0")
st.title("PDF RAG Chatbot")
st.info("Upload PDF → Process → Ask specific questions!")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "k" not in st.session_state:
    st.session_state.k = 3
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200

# Sidebar - PDF Upload + Settings
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose PDF", type="pdf")

st.sidebar.header("Retrieval Settings")
st.session_state.k = st.sidebar.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=3)
st.session_state.chunk_size = st.sidebar.slider("Chunk size", min_value=300, max_value=1500, step=100, value=1000)
st.session_state.chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=400, step=50, value=200)

if uploaded_file and st.sidebar.button("🚀 Process PDF", use_container_width=True):
    with st.spinner("🔄 Processing your PDF..."):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
        temp_file.close()
        
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size, 
            chunk_overlap=st.session_state.chunk_overlap
        )
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

# Chat Interface
if st.session_state.vectorstore:
    st.header("💬 Ask about your PDF")
    
    # Show recent chat history (last 6 messages)
    for message in st.session_state.messages[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with validation
    if prompt := st.chat_input("Ask a specific question about your PDF..."):
        prompt = prompt.strip()
        
        # Query validation
        if len(prompt) < 5:
            st.error("Please ask a more specific question (5+ characters)")
            continue
        
        # Log query
        with open("query_log.txt", "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            f.write(f"[{ts}] {prompt}\n")
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Retrieving relevant chunks...")
        progress_bar.progress(0.3)
        
        # Retrieve with configurable k
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.k})
        relevant_docs = retriever.invoke(prompt)
        
        # Limit context: top 3 chunks, 300 chars each
        context = "\n\n".join([doc.page_content[:300] for doc in relevant_docs[:3]])
        
        status_text.text("🤖 Generating focused answer...")
        progress_bar.progress(0.8)
        
        try:
            # Local model pipeline
            pipe = HuggingFacePipeline.from_model_id(
                model_id="microsoft/DialoGPT-medium",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 150, "truncation": True}
            )
            llm = ChatHuggingFace(llm=pipe)
            
            full_prompt = f"""Context from PDF (top 3 chunks):
{context}

Question: {prompt}

Answer briefly using only the context above:"""
            
            response = llm.invoke(full_prompt)
            
            # Truncate response to 800 chars
            answer = response.content[:800]
            if len(response.content) > 800:
                answer += "... (response truncated for readability)"
            
            status_text.text("✅ Complete!")
            progress_bar.progress(1.0)
            
            st.markdown(answer)
            
            # Source citations
            st.markdown("---")
            st.subheader("📚 Sources used")
            for i, doc in enumerate(relevant_docs[:3], start=1):
                preview = doc.page_content[:200].replace("\n", " ")
                st.markdown(f"**Source {i}:** {preview}...")
            
            st.caption(f"Retrieved {len(relevant_docs)} chunks | Used top 3")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

else:
    st.markdown("""
    ### 🚀 Quick Start
    1. **Upload PDF** in sidebar (ML papers, ECE notes, textbooks)
    2. **Adjust settings** (chunks, size, overlap)
    3. **Click "Process PDF"** 
    4. **Ask SPECIFIC questions**:
       • "What algorithms are discussed?"
       • "Key findings of this paper?"
       • "Fourier transform applications?"
    """)

st.markdown("---")
st.caption("RAG Chatbot - Production optimized")
