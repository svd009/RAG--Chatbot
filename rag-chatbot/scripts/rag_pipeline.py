from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # ✅ pull from .env

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0},
        task="text2text-generation",
        huggingfacehub_api_token=token
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

def ask_question(chain, question):
    response = chain.invoke({"query": question})
    return response["result"]
