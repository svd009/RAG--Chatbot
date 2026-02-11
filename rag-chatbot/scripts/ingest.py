from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def load_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
